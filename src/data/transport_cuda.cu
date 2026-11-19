// CudaStreamTransport (Phase E3): stream-pool CUDA transport — one
// strided-kernel launch per op when both endpoints are SM-accessible,
// copy-engine memcpys otherwise (Phase 0 findings F2/F4; see
// include/stratakv/data/transport_cuda.h for the copy-path policy).
//
// Threading model: Enqueue validates and appends a WorkItem; a single
// worker thread owns ALL CUDA submission (scratch upload, launches,
// event records) and completion delivery (cudaEventQuery polling from
// its loop — one documented executor thread, ADR 0004 compatible).

#include "stratakv/data/transport_cuda.h"

#include <algorithm>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include "stratakv/common/exceptions.h"
#include "stratakv/data/tier_table.h"

namespace stratakv::data {

namespace {

using control::CompletionSink;
using control::TransferOp;

[[noreturn]] void ThrowCuda(const char* what, cudaError_t err) {
  throw std::runtime_error(std::string("CudaStreamTransport: ") + what +
                           ": " + cudaGetErrorString(err));
}

#define TRANSPORT_CUDA_CHECK(expr)              \
  do {                                          \
    cudaError_t err_ = (expr);                  \
    if (err_ != cudaSuccess) {                  \
      ThrowCuda(#expr, err_);                   \
    }                                           \
  } while (0)

constexpr std::uint32_t kMaxGridY = 65535;   ///< CUDA gridDim.y limit.
constexpr std::uint64_t kMaxSlotBytes =      ///< Keeps kernel math u32.
    1ull << 31;

/// \brief Device-visible parameters for one op (uploaded to scratch).
/// Both sides of an op move the same model-wide total per id; per-side
/// segmentation (LAYERWISE per-layer bases vs one BLOCKWISE base) is
/// normalized through flat word offsets in the kernel.
struct DeviceOpParams {
  const char* const* src_bases;   ///< [src_segments] device array.
  char* const* dst_bases;         ///< [dst_segments] device array.
  const std::uint32_t* src_ids;   ///< [ids] source slot ids.
  const std::uint32_t* dst_ids;   ///< [ids] destination slot ids.
  std::uint32_t ids;              ///< Id pairs (1 for state ops).
  std::uint32_t words_per_id;     ///< Model-wide slot bytes / 8.
  std::uint32_t src_slice_words;  ///< Source per-segment bytes / 8.
  std::uint32_t dst_slice_words;  ///< Destination per-segment bytes/8.
  std::uint32_t src_stride;       ///< Source per-id stride, bytes.
  std::uint32_t dst_stride;       ///< Destination per-id stride, bytes.
};

/// \brief One op = one launch: grid.y walks id pairs, grid.x walks the
/// flat word range; per-side segmentation via slice division (equal
/// per-segment slices are the tier-group invariant). The single-segment
/// case (BLOCKWISE pool) skips the divisions.
__global__ void TransferOpKernel(const DeviceOpParams* __restrict__ p) {
  const std::uint32_t w =
      blockIdx.x * static_cast<std::uint32_t>(blockDim.x) + threadIdx.x;
  if (w >= p->words_per_id) return;
  const std::uint32_t k = blockIdx.y;  // Id-pair index.
  const char* src;
  if (p->src_slice_words == p->words_per_id) {
    src = p->src_bases[0] +
          static_cast<std::uint64_t>(p->src_ids[k]) * p->src_stride +
          static_cast<std::uint64_t>(w) * 8;
  } else {
    const std::uint32_t s = w / p->src_slice_words;
    src = p->src_bases[s] +
          static_cast<std::uint64_t>(p->src_ids[k]) * p->src_stride +
          static_cast<std::uint64_t>(w - s * p->src_slice_words) * 8;
  }
  char* dst;
  if (p->dst_slice_words == p->words_per_id) {
    dst = p->dst_bases[0] +
          static_cast<std::uint64_t>(p->dst_ids[k]) * p->dst_stride +
          static_cast<std::uint64_t>(w) * 8;
  } else {
    const std::uint32_t d = w / p->dst_slice_words;
    dst = p->dst_bases[d] +
          static_cast<std::uint64_t>(p->dst_ids[k]) * p->dst_stride +
          static_cast<std::uint64_t>(w - d * p->dst_slice_words) * 8;
  }
  *reinterpret_cast<unsigned long long*>(dst) =
      *reinterpret_cast<const unsigned long long*>(src);
}

/// \brief Validated op geometry (Enqueue output, launch input).
struct OpPlan {
  const GroupAddressing* src = nullptr;
  const GroupAddressing* dst = nullptr;
  std::uint64_t total_bytes = 0;  ///< Per-id model-wide bytes (src==dst).
  bool kernel = false;            ///< Kernel path eligible (vs memcpys).
  std::vector<std::uint32_t> src_ids;
  std::vector<std::uint32_t> dst_ids;
};

std::string OpLabel(std::size_t index, const TransferOp& op) {
  return "op[" + std::to_string(index) + "] tier " +
         std::to_string(op.src_tier) + "->" + std::to_string(op.dst_tier) +
         " group " + std::to_string(op.src_group) + "->" +
         std::to_string(op.dst_group);
}

std::uint64_t RoundUp(std::uint64_t value, std::uint64_t align) {
  return ((value + align - 1) / align) * align;
}

}  // namespace

struct CudaStreamTransport::Impl {
  struct Config config;
  TierTable table;

  cudaStream_t upload_stream = nullptr;
  std::vector<cudaStream_t> streams;

  // Device scratch arena + pinned mirror; single writer (the worker):
  // bump-allocated per batch, reset whenever nothing is in flight.
  void* arena_dev = nullptr;
  void* arena_host = nullptr;
  std::uint64_t arena_bytes = 0;
  std::uint64_t arena_next = 0;

  /// \brief Per-batch bookkeeping; temporary batches (arena too full)
  /// own allocations freed when their last op completes.
  struct BatchRecord {
    bool temporary = false;
    void* temp_dev = nullptr;
    void* temp_host = nullptr;
    std::size_t remaining = 0;
  };

  struct WorkItem {
    std::vector<OpPlan> plans;
    std::vector<TransferOp> ops;  // op ids + object ids.
    CompletionSink* sink = nullptr;
    std::uint64_t scratch_bytes = 0;  ///< Blob size to reserve.
  };

  struct InFlight {
    std::uint64_t op_id = 0;
    bool ok = true;
    CompletionSink* sink = nullptr;
    cudaEvent_t event = nullptr;
    std::shared_ptr<BatchRecord> batch;
  };

  mutable std::mutex mu;
  std::condition_variable cv;
  std::deque<WorkItem> queue;
  std::deque<InFlight> in_flight;
  std::vector<cudaEvent_t> event_pool;
  bool draining = false;     // Enqueue rejects (DrainPending/dtor).
  bool stop_worker = false;  // Join request; worker drains first.
  bool worker_failed = false;

  std::thread worker;

  ~Impl() {
    if (worker.joinable()) {
      worker.join();
    }
    for (cudaEvent_t event : event_pool) {
      (void)cudaEventDestroy(event);
    }
    for (cudaStream_t stream : streams) {
      (void)cudaStreamDestroy(stream);
    }
    if (upload_stream != nullptr) {
      (void)cudaStreamDestroy(upload_stream);
    }
    (void)cudaFree(arena_dev);
    (void)cudaFreeHost(arena_host);
  }

  cudaEvent_t AcquireEvent() {
    {
      std::lock_guard<std::mutex> lock(mu);
      if (!event_pool.empty()) {
        cudaEvent_t event = event_pool.back();
        event_pool.pop_back();
        return event;
      }
    }
    cudaEvent_t event = nullptr;
    cudaError_t err =
        cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
    if (err != cudaSuccess) {
      ThrowCuda("cudaEventCreate", err);
    }
    return event;
  }

  void RecycleEvent(cudaEvent_t event) {
    std::lock_guard<std::mutex> lock(mu);
    event_pool.push_back(event);
  }

  /// \brief Worker-only: reserve `bytes` of scratch. Prefers the arena
  /// (bump; resets when nothing is in flight); falls back to a
  /// temporary pinned+device pair. Returns {device, host, record}.
  std::tuple<void*, void*, std::shared_ptr<BatchRecord>> ClaimScratch(
      std::uint64_t bytes) {
    auto batch = std::make_shared<BatchRecord>();
    {
      std::lock_guard<std::mutex> lock(mu);
      const bool idle = in_flight.empty();
      if (idle) {
        arena_next = 0;  // Nothing in flight: the arena is all free.
      }
      if (arena_next + bytes <= arena_bytes) {
        const std::uint64_t at = arena_next;
        arena_next = RoundUp(at + bytes, 256);
        return {static_cast<char*>(arena_dev) + at,
                static_cast<char*>(arena_host) + at, batch};
      }
    }
    // Oversized batch or arena busy: temporary allocation (blocking,
    // rare — bounded by queue pressure, not by the op stream).
    cudaError_t err = cudaMalloc(&batch->temp_dev, bytes);
    if (err != cudaSuccess) {
      ThrowCuda("cudaMalloc(scratch)", err);
    }
    err = cudaMallocHost(&batch->temp_host, bytes);
    if (err != cudaSuccess) {
      (void)cudaFree(batch->temp_dev);
      ThrowCuda("cudaMallocHost(scratch)", err);
    }
    batch->temporary = true;
    return {batch->temp_dev, batch->temp_host, batch};
  }

  /// \brief Worker-only, on each op completion: frees a temporary
  /// batch's buffers when its last op completed.
  void ReleaseBatch(const std::shared_ptr<BatchRecord>& batch) {
    if (!batch->temporary || batch->remaining == 0) {
      return;
    }
    if (--batch->remaining == 0) {
      (void)cudaFree(batch->temp_dev);
      (void)cudaFreeHost(batch->temp_host);
      batch->temp_dev = nullptr;
      batch->temp_host = nullptr;
    }
  }

  void FailEverything(const std::string& what) {
    std::fprintf(stderr, "CudaStreamTransport worker: %s\n", what.c_str());
    std::vector<InFlight> pending;
    {
      std::lock_guard<std::mutex> lock(mu);
      worker_failed = true;
      stop_worker = true;
      queue.clear();
      pending.assign(std::make_move_iterator(in_flight.begin()),
                     std::make_move_iterator(in_flight.end()));
      in_flight.clear();
    }
    for (InFlight& f : pending) {
      f.sink->OnOpCompleted(f.op_id, /*ok=*/false);
      (void)cudaEventDestroy(f.event);
    }
    cv.notify_all();
  }

  /// \brief The worker: launches queued batches, polls in-flight
  /// events, delivers completions. Owns all CUDA submission.
  void Run() {
    std::unique_lock<std::mutex> lock(mu);
    std::vector<InFlight> ready;
    while (true) {
      while (queue.empty() && in_flight.empty() && !stop_worker) {
        cv.wait(lock);
      }
      if (stop_worker && queue.empty() && in_flight.empty()) {
        break;
      }
      if (!queue.empty()) {
        WorkItem item = std::move(queue.front());
        queue.pop_front();
        lock.unlock();
        try {
          LaunchBatch(item);
        } catch (const std::exception& e) {
          FailEverything(e.what());  // Takes the lock itself.
        }
        lock.lock();
        cv.notify_all();  // DrainPending waits on emptiness.
        continue;
      }

      // Queue empty, in-flight non-empty: poll events.
      ready.clear();
      for (auto it = in_flight.begin(); it != in_flight.end();) {
        const cudaError_t q = cudaEventQuery(it->event);
        if (q == cudaSuccess) {
          ready.push_back(*it);
          it = in_flight.erase(it);
        } else if (q != cudaErrorNotReady) {
          it->ok = false;  // Sticky/async error: fail this op.
          ready.push_back(*it);
          it = in_flight.erase(it);
        } else {
          ++it;
        }
      }
      lock.unlock();
      for (InFlight& f : ready) {
        f.sink->OnOpCompleted(f.op_id, f.ok);
        ReleaseBatch(f.batch);
        if (f.ok) {
          RecycleEvent(f.event);
        } else {
          (void)cudaEventDestroy(f.event);  // Error state may be sticky.
        }
      }
      lock.lock();
      cv.notify_all();
      if (ready.empty() && queue.empty() && !stop_worker) {
        // Pace the poll: the event+completion round trip is ~20-30 us
        // (Phase 0 micro table); 50 us caps added latency without a
        // hard spin.
        cv.wait_for(lock, std::chrono::microseconds(50));
      }
    }
  }

  /// \brief Worker-only, no lock held: build the scratch blob, upload,
  /// then launch each op with its event.
  void LaunchBatch(const WorkItem& item) {
    const std::uint64_t region_bytes = RoundUp(item.scratch_bytes, 256);
    auto [region_dev, region_host, batch] = ClaimScratch(region_bytes);

    // Pass 1: build the whole blob host-side, patching device pointers
    // relative to the region's device base.
    std::byte* host = static_cast<std::byte*>(region_host);
    const std::uintptr_t dev_base =
        reinterpret_cast<std::uintptr_t>(region_dev);
    std::uint64_t off = 0;
    auto take = [&host, &off](std::uint64_t bytes,
                              std::uint64_t align) -> void* {
      off = RoundUp(off, align);
      void* at = host + off;
      off += bytes;
      return at;
    };

    struct Staged {
      DeviceOpParams* params;
      std::uint64_t params_off;
    };
    std::vector<Staged> staged;
    staged.reserve(item.ops.size());
    for (std::size_t i = 0; i < item.ops.size(); ++i) {
      const OpPlan& plan = item.plans[i];
      const std::size_t n = plan.src_ids.size();
      DeviceOpParams* params = static_cast<DeviceOpParams*>(
          take(sizeof(DeviceOpParams), alignof(DeviceOpParams)));
      const char** src_ptrs = static_cast<const char**>(
          take(plan.src->bases.size() * sizeof(void*), sizeof(void*)));
      char** dst_ptrs = static_cast<char**>(
          take(plan.dst->bases.size() * sizeof(void*), sizeof(void*)));
      std::uint32_t* src_ids = static_cast<std::uint32_t*>(
          take(n * sizeof(std::uint32_t), alignof(std::uint32_t)));
      std::uint32_t* dst_ids = static_cast<std::uint32_t*>(
          take(n * sizeof(std::uint32_t), alignof(std::uint32_t)));

      for (std::size_t j = 0; j < plan.src->bases.size(); ++j) {
        src_ptrs[j] = static_cast<const char*>(plan.src->bases[j]);
      }
      for (std::size_t j = 0; j < plan.dst->bases.size(); ++j) {
        dst_ptrs[j] = static_cast<char*>(plan.dst->bases[j]);
      }
      std::copy(plan.src_ids.begin(), plan.src_ids.end(), src_ids);
      std::copy(plan.dst_ids.begin(), plan.dst_ids.end(), dst_ids);
      params->src_bases = reinterpret_cast<const char* const*>(
          dev_base + (reinterpret_cast<std::byte*>(src_ptrs) - host));
      params->dst_bases = reinterpret_cast<char* const*>(
          dev_base + (reinterpret_cast<std::byte*>(dst_ptrs) - host));
      params->src_ids = reinterpret_cast<std::uint32_t*>(
          dev_base + (reinterpret_cast<std::byte*>(src_ids) - host));
      params->dst_ids = reinterpret_cast<std::uint32_t*>(
          dev_base + (reinterpret_cast<std::byte*>(dst_ids) - host));
      params->ids = static_cast<std::uint32_t>(n);
      params->words_per_id =
          static_cast<std::uint32_t>(plan.total_bytes / 8);
      params->src_slice_words =
          static_cast<std::uint32_t>(plan.src->slice_bytes / 8);
      params->dst_slice_words =
          static_cast<std::uint32_t>(plan.dst->slice_bytes / 8);
      params->src_stride =
          static_cast<std::uint32_t>(plan.src->block_stride);
      params->dst_stride =
          static_cast<std::uint32_t>(plan.dst->block_stride);
      staged.push_back(
          {params, static_cast<std::uint64_t>(
                       reinterpret_cast<std::byte*>(params) - host)});
    }

    // Pass 2: upload the blob, THEN launch (ops on other streams
    // dereference the blob's device pointers).
    TRANSPORT_CUDA_CHECK(cudaMemcpyAsync(region_dev, region_host,
                                         region_bytes,
                                         cudaMemcpyHostToDevice,
                                         upload_stream));
    TRANSPORT_CUDA_CHECK(cudaStreamSynchronize(upload_stream));

    // Pass 3: one launch + event per op.
    batch->remaining = item.ops.size();
    std::uint32_t stream_rr = 0;
    for (std::size_t i = 0; i < item.ops.size(); ++i) {
      const OpPlan& plan = item.plans[i];
      const TransferOp& op = item.ops[i];
      cudaStream_t stream = streams[stream_rr++ % streams.size()];
      cudaEvent_t event = AcquireEvent();

      if (plan.kernel) {
        const dim3 grid(
            (staged[i].params->words_per_id + config.thread_block - 1) /
                config.thread_block,
            staged[i].params->ids);
        TransferOpKernel<<<grid, config.thread_block, 0, stream>>>(
            reinterpret_cast<DeviceOpParams*>(dev_base +
                                              staged[i].params_off));
        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
          (void)cudaEventDestroy(event);
          ThrowCuda("TransferOpKernel launch", err);
        }
      } else {
        // Copy-engine fallback: per (id, layer-chunk) cudaMemcpyAsync
        // at gcd(src_slice, dst_slice) boundaries — correct for any
        // LAYERWISE/BLOCKWISE mix; slow by F2, pin the tier for speed.
        const std::uint64_t src_slice = plan.src->slice_bytes;
        const std::uint64_t dst_slice = plan.dst->slice_bytes;
        const std::uint64_t chunk = std::gcd(src_slice, dst_slice);
        const cudaMemcpyKind kind =
            plan.src->kind == common::DeviceKind::kGpu
                ? cudaMemcpyDeviceToHost
                : cudaMemcpyHostToDevice;
        const std::size_t n = plan.src_ids.size();
        for (std::size_t k = 0; k < n; ++k) {
          for (std::uint64_t c = 0; c < plan.total_bytes; c += chunk) {
            const std::uint64_t s = c / src_slice;
            const std::uint64_t s_off = c - s * src_slice;
            const std::uint64_t d = c / dst_slice;
            const std::uint64_t d_off = c - d * dst_slice;
            const cudaError_t err = cudaMemcpyAsync(
                static_cast<char*>(plan.dst->bases[d]) +
                    static_cast<std::uint64_t>(plan.dst_ids[k]) *
                        plan.dst->block_stride +
                    d_off,
                static_cast<const char*>(plan.src->bases[s]) +
                    static_cast<std::uint64_t>(plan.src_ids[k]) *
                        plan.src->block_stride +
                    s_off,
                chunk, kind, stream);
            if (err != cudaSuccess) {
              (void)cudaEventDestroy(event);
              ThrowCuda("cudaMemcpyAsync", err);
            }
          }
        }
      }
      const cudaError_t err = cudaEventRecord(event, stream);
      if (err != cudaSuccess) {
        (void)cudaEventDestroy(event);
        ThrowCuda("cudaEventRecord", err);
      }
      InFlight flight;
      flight.op_id = op.op_id;
      flight.sink = item.sink;
      flight.event = event;
      flight.batch = batch;
      {
        std::lock_guard<std::mutex> lock(mu);
        in_flight.push_back(flight);
      }
    }
  }
};

CudaStreamTransport::CudaStreamTransport(TierTable table)
    : CudaStreamTransport(std::move(table), Config()) {}

CudaStreamTransport::CudaStreamTransport(TierTable table, Config config)
    : impl_(std::make_unique<Impl>()) {
  int devices = 0;
  cudaError_t err = cudaGetDeviceCount(&devices);
  if (err != cudaSuccess || devices <= 0) {
    throw std::runtime_error(
        std::string("CudaStreamTransport: no usable CUDA device (") +
        cudaGetErrorString(err) + ")");
  }
  if (config.streams == 0 || config.thread_block == 0) {
    throw std::invalid_argument("CudaStreamTransport: bad config");
  }
  impl_->config = config;
  impl_->table = std::move(table);
  impl_->arena_bytes = std::max<uint64_t>(config.scratch_bytes, 1u << 20);

  TRANSPORT_CUDA_CHECK(cudaStreamCreate(&impl_->upload_stream));
  impl_->streams.resize(config.streams);
  for (cudaStream_t& stream : impl_->streams) {
    TRANSPORT_CUDA_CHECK(cudaStreamCreate(&stream));
  }
  TRANSPORT_CUDA_CHECK(cudaMalloc(&impl_->arena_dev, impl_->arena_bytes));
  TRANSPORT_CUDA_CHECK(
      cudaMallocHost(&impl_->arena_host, impl_->arena_bytes));

  impl_->worker = std::thread([impl = impl_.get()] { impl->Run(); });
}

CudaStreamTransport::~CudaStreamTransport() {
  {
    std::lock_guard<std::mutex> lock(impl_->mu);
    impl_->draining = true;
    impl_->stop_worker = true;
  }
  impl_->cv.notify_all();
  impl_.reset();  // ~Impl joins the worker, then frees CUDA state.
}

void CudaStreamTransport::Enqueue(std::vector<TransferOp> batch,
                                  CompletionSink& sink) {
  if (batch.empty()) {
    return;
  }
  Impl::WorkItem item;
  item.sink = &sink;
  item.ops = std::move(batch);
  item.plans.reserve(item.ops.size());

  for (std::size_t i = 0; i < item.ops.size(); ++i) {
    const TransferOp& op = item.ops[i];
    const bool snapshot = op.kind == TransferOp::Kind::kGpuToCpu ||
                          op.kind == TransferOp::Kind::kStateSnapshot;
    const bool restore = op.kind == TransferOp::Kind::kCpuToGpu ||
                         op.kind == TransferOp::Kind::kStateRestore;
    if (!snapshot && !restore) {
      throw LogicError("CudaStreamTransport: " + OpLabel(i, op) +
                       " targets an SSD tier (E5 backends not built)");
    }
    const bool state_op = op.kind == TransferOp::Kind::kStateSnapshot ||
                          op.kind == TransferOp::Kind::kStateRestore;

    const GroupAddressing* src =
        impl_->table.LookUp(op.src_tier, op.src_group);
    const GroupAddressing* dst =
        impl_->table.LookUp(op.dst_tier, op.dst_group);
    if (src == nullptr || dst == nullptr) {
      throw LogicError("CudaStreamTransport: " + OpLabel(i, op) +
                       " addresses an unknown (tier, group)");
    }
    if (src->is_state != state_op || dst->is_state != state_op) {
      throw LogicError("CudaStreamTransport: " + OpLabel(i, op) +
                       " op kind does not match group addressing kind");
    }

    OpPlan plan;
    plan.src = src;
    plan.dst = dst;
    if (state_op) {
      plan.src_ids = {op.src_object};
      plan.dst_ids = {op.dst_object};
    } else {
      if (op.src_blocks.size() != op.dst_blocks.size() ||
          op.src_blocks.empty()) {
        throw LogicError("CudaStreamTransport: " + OpLabel(i, op) +
                         " block lists must be parallel and non-empty");
      }
      plan.src_ids = op.src_blocks;
      plan.dst_ids = op.dst_blocks;
    }
    if (plan.src_ids.size() > kMaxGridY) {
      throw LogicError("CudaStreamTransport: " + OpLabel(i, op) + " has " +
                       std::to_string(plan.src_ids.size()) +
                       " ids (kernel grid-y limit 65535; split the op)");
    }
    const std::uint64_t src_total = src->slice_bytes * src->bases.size();
    const std::uint64_t dst_total = dst->slice_bytes * dst->bases.size();
    if (src_total != dst_total || src_total == 0 ||
        src_total > kMaxSlotBytes) {
      throw LogicError("CudaStreamTransport: " + OpLabel(i, op) +
                       " slot sizes differ or are invalid (" +
                       std::to_string(src_total) + " vs " +
                       std::to_string(dst_total) + " bytes)");
    }
    if (src->block_stride > kMaxSlotBytes || dst->block_stride > kMaxSlotBytes) {
      throw LogicError("CudaStreamTransport: " + OpLabel(i, op) +
                       " stride exceeds the u32 kernel math");
    }
    plan.total_bytes = src_total;
    const bool aligned = plan.total_bytes % 8 == 0 &&
                         src->slice_bytes % 8 == 0 &&
                         dst->slice_bytes % 8 == 0 &&
                         src->block_stride % 8 == 0 &&
                         dst->block_stride % 8 == 0;
    plan.kernel =
        src->sm_accessible && dst->sm_accessible && aligned;

    item.scratch_bytes +=
        RoundUp(sizeof(DeviceOpParams), 8) +
        src->bases.size() * sizeof(void*) +
        dst->bases.size() * sizeof(void*) +
        RoundUp(plan.src_ids.size() * sizeof(std::uint32_t), 8) +
        RoundUp(plan.dst_ids.size() * sizeof(std::uint32_t), 8);
    item.plans.push_back(std::move(plan));
  }

  {
    std::lock_guard<std::mutex> lock(impl_->mu);
    if (impl_->draining || impl_->worker_failed) {
      throw LogicError("CudaStreamTransport: drained");
    }
    impl_->queue.push_back(std::move(item));
  }
  impl_->cv.notify_one();
}

void CudaStreamTransport::DrainPending(std::chrono::milliseconds grace) {
  std::unique_lock<std::mutex> lock(impl_->mu);
  impl_->draining = true;
  impl_->cv.notify_all();
  if (impl_->worker_failed) {
    return;
  }
  const auto deadline = std::chrono::steady_clock::now() + grace;
  while (!impl_->queue.empty() || !impl_->in_flight.empty()) {
    if (impl_->cv.wait_until(lock, deadline) == std::cv_status::timeout) {
      break;
    }
  }
}

std::size_t CudaStreamTransport::num_in_flight() const {
  std::lock_guard<std::mutex> lock(impl_->mu);
  return impl_->in_flight.size();
}

}  // namespace stratakv::data
