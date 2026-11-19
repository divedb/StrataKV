// Phase E (E3): full-stack per-task cost benchmark — TaskEngine ->
// TransportExecutor -> TransferScheduler -> CudaStreamTransport on the
// Qwen2-0.5B-shaped geometry (24 layers x 8 KiB = 192 KiB block; the
// Phase 0 / E0 payload class).
//
// Measures the roadmap §5 metrics through the REAL stack (match and
// staging excluded from the timed window; reported separately):
//   sync       SubmitX -> Wait per task (the FlexKV per-task cost
//              definition; acceptance: <= 1 ms fixed at 4 blocks)
//   pipelined  K=8 tasks submitted back-to-back, one Wait (the
//              scheduler's drain-and-batch shape)
// plus the 0-byte extrapolated fixed overhead per direction.
//
// Byte verification (XXH3 golden pattern through the stack) precedes
// every timed scenario — no timing counts unless bytes verify.
// GPU-only (STRATAKV_GPU=ON), like gpu_baseline_bench NOT a ctest
// entry. Run: ./build-gpu/benchmarks/stratakv_gpu_transport_bench

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <numeric>
#include <span>
#include <utility>
#include <vector>

#include "xxhash.h"

#include "stratakv/common/model_layout.h"
#include "stratakv/common/types.h"
#include "stratakv/control/task_engine.h"
#include "stratakv/data/cpu_tier.h"
#include "stratakv/data/registry.h"
#include "stratakv/data/tier_table.h"
#include "stratakv/data/transport.h"
#include "stratakv/data/transport_cuda.h"
#include "stratakv/metadata/metadata_plane.h"

namespace {

using stratakv::common::BlockId;
using stratakv::common::BufferView;
using stratakv::common::DeviceKind;
using stratakv::common::FullAttnSpec;
using stratakv::common::LayerGroup;
using stratakv::common::ModelLayout;
using stratakv::common::TaskHandle;
using stratakv::common::TaskState;
using stratakv::common::TokenId;
using stratakv::control::TaskEngine;
using stratakv::data::CpuTier;
using stratakv::data::CpuTierConfig;
using stratakv::data::CudaStreamTransport;
using stratakv::data::DeviceMemoryRegistry;
using stratakv::data::GroupDescriptor;
using stratakv::data::GroupLayout;
using stratakv::data::HostBacking;
using stratakv::data::IsolationMode;
using stratakv::data::TierTable;
using stratakv::data::TransferScheduler;
using stratakv::data::TransportExecutor;
using stratakv::metadata::MetadataPlane;
using stratakv::metadata::MetadataPlaneConfig;
using stratakv::metadata::RadixIndex;

constexpr std::uint32_t kLayers = 24;
constexpr std::uint64_t kPerLayer = 8192;
constexpr std::uint64_t kSlotBytes = kLayers * kPerLayer;  // 192 KiB.
constexpr std::uint32_t kTpb = 16;
constexpr std::uint32_t kGpuSlots = 512;   // 8 pipelined x 64 blocks.
constexpr std::uint32_t kPoolBlocks = 600; // 8 x 64 + slack.
constexpr int kSyncIters = 200;
constexpr int kBatches = 30;  // Pipelined batches.
constexpr int kPipelinedK = 8;

stratakv::common::ModelLayout BenchLayout() {
  LayerGroup full;
  full.name = "full_attn";
  full.spec = FullAttnSpec{2, 64};
  for (int l = 0; l < static_cast<int>(kLayers); ++l) {
    full.layer_indices.push_back(l);
  }
  full.tokens_per_block = static_cast<int>(kTpb);
  return stratakv::common::ModelLayout::FromGroups({full});
}

#define CUDA_CHECK_BENCH(expr)                                        \
  do {                                                                \
    cudaError_t err_ = (expr);                                        \
    if (err_ != cudaSuccess) {                                        \
      std::fprintf(stderr, "CUDA error %s: %s\n", #expr,              \
                   cudaGetErrorString(err_));                         \
      std::exit(1);                                                   \
    }                                                                 \
  } while (0)

std::uint64_t SplitMix64(std::uint64_t& state) {
  state += 0x9E3779B97F4A7C15ULL;
  std::uint64_t z = state;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

double Percentile(std::vector<double> samples, double p) {
  std::sort(samples.begin(), samples.end());
  const std::size_t idx = static_cast<std::size_t>(
      p * static_cast<double>(samples.size() - 1));
  return samples[idx];
}

double NowUs() {
  return std::chrono::duration<double, std::micro>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

double GbPerSec(double bytes, double us) {
  return us <= 0.0 ? 0.0 : bytes / (us * 1000.0);
}

std::vector<TokenId> Tokens(std::uint32_t blocks, std::uint64_t salt) {
  std::uint64_t state = salt;
  std::vector<TokenId> tokens;
  tokens.reserve(static_cast<std::size_t>(blocks) * kTpb);
  for (std::uint32_t i = 0; i < blocks * kTpb; ++i) {
    tokens.push_back(static_cast<TokenId>(SplitMix64(state) % 100000));
  }
  return tokens;
}

/// \brief The full stack under test.
class Stack {
 public:
  Stack() : layout_(BenchLayout()) {
    LayerGroup full;
    full.name = "full_attn";
    full.spec = FullAttnSpec{2, 64};
    for (int l = 0; l < static_cast<int>(kLayers); ++l) {
      full.layer_indices.push_back(l);
    }
    full.tokens_per_block = static_cast<int>(kTpb);
    // layout_ initialized in the member-init list (ModelLayout's
    // default constructor is private by design).

    MetadataPlaneConfig plane_config;
    plane_config.blocks_per_group = kPoolBlocks;
    plane_ = std::make_unique<MetadataPlane>(
        layout_, std::make_unique<RadixIndex>(layout_), plane_config);

    CpuTierConfig tier_config;
    tier_config.blocks_per_group = kPoolBlocks;
    tier_config.state_objects_per_group = 8;
    tier_config.backing = HostBacking::kPinned;  // Kernel path (F4).
    tier_ = std::make_unique<CpuTier>(layout_, tier_config);

    for (std::uint32_t l = 0; l < kLayers; ++l) {
      void* buffer = nullptr;
      CUDA_CHECK_BENCH(
          cudaMalloc(&buffer, kGpuSlots * kPerLayer));
      gpu_layers_.push_back(buffer);
    }

    GroupDescriptor descriptor;
    descriptor.group_name = "full_attn";
    descriptor.group = 0;
    descriptor.layout = GroupLayout::kLayerwise;
    descriptor.device_id = 0;
    for (void* layer : gpu_layers_) {
      BufferView view;
      view.base = layer;
      view.kind = DeviceKind::kGpu;
      view.device_id = 0;
      view.size_bytes = kGpuSlots * kPerLayer;
      descriptor.per_layer.push_back(view);
    }
    const auto registration = registry_.RegisterGroup(std::move(descriptor));

    TierTable table;
    table.AddCpuTier(0, layout_, *tier_);
    table.AddGpuTier(1, registry_, registration, layout_, kGpuSlots);

    auto transport = std::make_unique<CudaStreamTransport>(std::move(table));
    scheduler_ = std::make_shared<TransferScheduler>(
        std::move(transport), IsolationMode::kInProcess,
        TransferScheduler::Config());
    engine_ = std::make_unique<TaskEngine>(
        plane_.get(), std::make_shared<TransportExecutor>(scheduler_));
  }

  ~Stack() {
    engine_->Shutdown(std::chrono::milliseconds(5000));
    scheduler_->Shutdown(std::chrono::milliseconds(5000));
    for (void* layer : gpu_layers_) {
      (void)cudaFree(layer);
    }
  }

  MetadataPlane& plane() { return *plane_; }
  CpuTier& tier() { return *tier_; }
  TaskEngine& engine() { return *engine_; }
  TransferScheduler& scheduler() { return *scheduler_; }

  /// \brief Stamp the golden block into GPU slots [slot, slot+blocks)
  /// (same content per slot — idempotent re-loads in the timed loops).
  void SeedGpu(std::uint32_t slot, std::uint32_t blocks,
               const std::vector<std::uint64_t>& golden) {
    for (std::uint32_t b = 0; b < blocks; ++b) {
      for (std::uint32_t l = 0; l < kLayers; ++l) {
        CUDA_CHECK_BENCH(cudaMemcpy(
            static_cast<char*>(gpu_layers_[l]) +
                static_cast<std::uint64_t>(slot + b) * kPerLayer,
            reinterpret_cast<const char*>(golden.data()) + l * kPerLayer,
            kPerLayer, cudaMemcpyHostToDevice));
      }
    }
  }

  /// \brief Hash GPU slots [slot, slot+blocks) gathered back.
  std::vector<std::uint64_t> GatherGpu(std::uint32_t slot,
                                       std::uint32_t blocks) {
    std::vector<std::uint64_t> gather(
        static_cast<std::size_t>(blocks) * kSlotBytes / 8);
    for (std::uint32_t b = 0; b < blocks; ++b) {
      for (std::uint32_t l = 0; l < kLayers; ++l) {
        CUDA_CHECK_BENCH(cudaMemcpy(
            reinterpret_cast<char*>(gather.data()) +
                (static_cast<std::size_t>(b) * kSlotBytes +
                 static_cast<std::size_t>(l) * kPerLayer),
            static_cast<const char*>(gpu_layers_[l]) +
                static_cast<std::uint64_t>(slot + b) * kPerLayer,
            kPerLayer, cudaMemcpyDeviceToHost));
      }
    }
    return gather;
  }

 private:
  stratakv::common::ModelLayout layout_;  std::unique_ptr<MetadataPlane> plane_;
  std::unique_ptr<CpuTier> tier_;
  std::vector<void*> gpu_layers_;
  DeviceMemoryRegistry registry_;
  std::shared_ptr<TransferScheduler> scheduler_;
  std::unique_ptr<TaskEngine> engine_;
};

TaskState WaitFor(TaskEngine& engine, TaskHandle handle) {
  const TaskHandle handles[1] = {handle};
  const auto result = engine.Wait(std::span<const TaskHandle>(handles, 1),
                                  std::chrono::milliseconds(10000));
  return result.items.at(0).state;
}

std::uint64_t HashSpan(const std::vector<std::uint64_t>& words,
                       std::size_t begin_word, std::size_t bytes) {
  return XXH3_64bits(words.data() + begin_word, bytes);
}

}  // namespace

int main() {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices <= 0) {
    std::fprintf(stderr, "no CUDA device\n");
    return 1;
  }  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, 0);
  int driver = 0, runtime = 0;
  cudaDriverGetVersion(&driver);
  cudaRuntimeGetVersion(&runtime);
  std::printf("== StrataKV full-stack transport bench ==\n");
  std::printf("device: %s (cc %d.%d) | driver %d.%d | runtime %d.%d\n",
              prop.name, prop.major, prop.minor, driver / 1000, driver % 1000,
              runtime / 1000, runtime % 1000);
  std::printf("block geometry: %u layers x %llu B = %llu B | pinned CPU tier "
              "(kernel path) | in-process mode\n\n",
              kLayers, static_cast<unsigned long long>(kPerLayer),
              static_cast<unsigned long long>(kSlotBytes));

  Stack stack;
  const std::vector<std::uint32_t> sizes = {1, 4, 16, 64};

  // Golden pattern + hashes for one block payload (reused per size).
  std::vector<std::uint64_t> golden(static_cast<std::size_t>(kSlotBytes) /
                                    8);
  {
    std::uint64_t state = 0x5EED;
    for (auto& word : golden) {
      word = SplitMix64(state);
    }
  }

  struct Row {
    std::uint32_t blocks;
    double p50_us;
    double p95_us;
    double p99_us;
  };
  std::vector<Row> put_rows, get_rows;

  // ---- PUT (D2H): fresh prefixes, evict between iterations ----
  std::printf("== SYNC mode: SubmitPut -> Wait per task ==\n");
  std::printf("%-6s %6s %10s %10s %10s %10s %8s\n", "task", "blocks",
              "bytes", "p50_us", "p95_us", "p99_us", "GB/s");
  std::uint32_t salt = 1000;
  for (std::uint32_t n : sizes) {
    stack.SeedGpu(0, n, golden);
    // Byte verification through the stack before timing.
    {
      const auto plan = stack.plane().StagePut(Tokens(n, salt));
      std::vector<BlockId> src(n);
      std::iota(src.begin(), src.end(), BlockId{0});
      const TaskHandle put = stack.engine().SubmitPut(plan, src);
      stack.scheduler().Flush();
      if (WaitFor(stack.engine(), put) != TaskState::kSucceeded) {
        std::fprintf(stderr, "verify put failed\n");
        return 1;
      }
      bool ok = true;
      for (std::uint32_t b = 0; b < n; ++b) {
        ok = ok && XXH3_64bits(stack.tier().BlockAddress(
                                     0, plan.blocks_to_store[b].block),
                               kSlotBytes) ==
                       HashSpan(golden, 0, kSlotBytes);
      }
      if (!ok) {
        std::fprintf(stderr, "PUT bytes corrupt (n=%u)\n", n);
        return 1;
      }
      (void)stack.plane().EvictBlocks(n);
    }
    std::vector<double> micros;
    micros.reserve(kSyncIters);
    for (int i = 0; i < kSyncIters; ++i) {
      const auto tokens_i = Tokens(n, salt++);
      const auto plan = stack.plane().StagePut(tokens_i);
      if (plan.blocks_to_store.size() != n) {
        std::fprintf(stderr,
                     "put %u iter %d: salt %u matched %u new %u stored %zu\n",
                     n, i, salt - 1, plan.matched_blocks, plan.new_blocks,
                     plan.blocks_to_store.size());
        return 1;
      }
      std::vector<BlockId> src(n);
      std::iota(src.begin(), src.end(), BlockId{0});
      const double t0 = NowUs();
      const TaskHandle put = stack.engine().SubmitPut(plan, src);
      stack.scheduler().Flush();
      const TaskState state = WaitFor(stack.engine(), put);
      micros.push_back(NowUs() - t0);
      if (state != TaskState::kSucceeded) {
        std::fprintf(stderr, "put task failed\n");
        return 1;
      }
      (void)stack.plane().EvictBlocks(n);
    }
    const Row row{n, Percentile(micros, 0.50), Percentile(micros, 0.95),
                  Percentile(micros, 0.99)};
    put_rows.push_back(row);
    std::printf("%-6s %6u %10llu %10.1f %10.1f %10.1f %8.2f\n", "put", n,
                static_cast<unsigned long long>(n) * kSlotBytes, row.p50_us,
                row.p95_us, row.p99_us,
                GbPerSec(n * kSlotBytes, row.p50_us));
  }

  // ---- GET (H2D): one primed prefix, repeated loads ----
  // Prime BEFORE the pipelined put section can evict it, and pin the
  // entry for the duration (pinned blocks are never evictable).
  const auto tokens = Tokens(64, 7);
  {
    const auto plan = stack.plane().StagePut(tokens);
    std::vector<BlockId> src(64);
    std::iota(src.begin(), src.end(), BlockId{0});
    const TaskHandle put = stack.engine().SubmitPut(plan, src);
    stack.scheduler().Flush();
    if (WaitFor(stack.engine(), put) != TaskState::kSucceeded) {
      std::fprintf(stderr, "prime put failed\n");
      return 1;
    }
  }
  const auto held_pin = stack.plane().MatchForGet(tokens);
  if (held_pin.matched_blocks != 64u) {
    std::fprintf(stderr, "prime pin failed (%u blocks)\n",
                 held_pin.matched_blocks);
    return 1;
  }
  {
    // Match latency context (metadata plane, same box as index_bench).
    for (std::uint32_t n : sizes) {
      std::vector<double> micros;
      micros.reserve(200);
      for (int i = 0; i < 200; ++i) {
        const double t0 = NowUs();
        const auto match =
            stack.plane().MatchForGet(std::span<const TokenId>(
                tokens.data(), static_cast<std::size_t>(n) * kTpb));
        micros.push_back(NowUs() - t0);
        stack.plane().Release(match);
      }
      std::printf("%-6s %6u %10s %10.1f %10.1f %10.1f (match only)\n",
                  "get", n, "-", Percentile(micros, 0.50),
                  Percentile(micros, 0.95), Percentile(micros, 0.99));
    }

    for (std::uint32_t n : sizes) {
      // Byte verification: load into slots [256, 256+n), gather, hash.
      {
        const auto match = stack.plane().MatchForGet(
            std::span<const TokenId>(tokens.data(),
                                     static_cast<std::size_t>(n) * kTpb));
        std::vector<BlockId> dst(n);
        for (std::uint32_t b = 0; b < n; ++b) {
          dst[b] = 256 + b;
        }
        const TaskHandle get = stack.engine().SubmitGet(match, dst);
        stack.scheduler().Flush();
        if (WaitFor(stack.engine(), get) != TaskState::kSucceeded) {
          std::fprintf(stderr, "verify get failed\n");
          return 1;
        }
        const auto gather = stack.GatherGpu(256, n);
        for (std::uint32_t b = 0; b < n; ++b) {
          if (HashSpan(gather,
                       static_cast<std::size_t>(b) * kSlotBytes / 8,
                       kSlotBytes) !=
              HashSpan(golden, 0, kSlotBytes)) {
            std::fprintf(stderr, "GET bytes corrupt (block %u)\n", b);
            return 1;
          }
        }
      }
      // The CPU pool was primed from the golden GPU pattern by the
      // prime put, so loading it back into the (still-golden) slots
      // 0..n-1 keeps every timed iteration idempotent.
      std::vector<double> micros;
      micros.reserve(kSyncIters);
      for (int i = 0; i < kSyncIters; ++i) {
        const auto match = stack.plane().MatchForGet(
            std::span<const TokenId>(tokens.data(),
                                     static_cast<std::size_t>(n) * kTpb));
        std::vector<BlockId> dst(n);
        for (std::uint32_t b = 0; b < n; ++b) {
          dst[b] = b;  // Overwrite the seeded slots.
        }
        const double t0 = NowUs();
        const TaskHandle get = stack.engine().SubmitGet(match, dst);
        stack.scheduler().Flush();
        const TaskState state = WaitFor(stack.engine(), get);
        micros.push_back(NowUs() - t0);
        if (state != TaskState::kSucceeded) {
          std::fprintf(stderr, "get task failed\n");
          return 1;
        }
      }
      const Row row{n, Percentile(micros, 0.50), Percentile(micros, 0.95),
                    Percentile(micros, 0.99)};
      get_rows.push_back(row);
      std::printf("%-6s %6u %10llu %10.1f %10.1f %10.1f %8.2f\n", "get", n,
                  static_cast<unsigned long long>(n) * kSlotBytes,
                  row.p50_us, row.p95_us, row.p99_us,
                  GbPerSec(n * kSlotBytes, row.p50_us));
    }
  }

  // ---- PIPELINED: K tasks submitted, one flush, one wait ----
  std::printf("\n== PIPELINED mode (K=%d tasks, one flush + Wait; "
              "drain-and-batch shape) ==\n",
              kPipelinedK);
  std::printf("%-6s %6s %12s %12s\n", "task", "blocks", "p50_us/task",
              "GB/s");
  for (std::uint32_t n : sizes) {
    // Puts: K staged plans, distinct prefixes, then reclaim the pool.
    {
      std::vector<double> per_task;
      per_task.reserve(kBatches);
      for (int batch = 0; batch < kBatches; ++batch) {
        std::vector<MetadataPlane::PutPlan> plans;
        std::vector<TaskHandle> handles;
        const double t0 = NowUs();
        for (int k = 0; k < kPipelinedK; ++k) {
          plans.push_back(stack.plane().StagePut(
              Tokens(n, salt + static_cast<std::uint32_t>(batch * 100 + k))));
          std::vector<BlockId> src(n);
          std::iota(src.begin(), src.end(), BlockId{0});
          handles.push_back(stack.engine().SubmitPut(plans.back(), src));
        }
        stack.scheduler().Flush();
        for (TaskHandle handle : handles) {
          if (WaitFor(stack.engine(), handle) != TaskState::kSucceeded) {
            std::fprintf(stderr, "pipelined put failed\n");
            return 1;
          }
        }
        per_task.push_back((NowUs() - t0) /
                           static_cast<double>(kPipelinedK));
        (void)stack.plane().EvictBlocks(kPipelinedK * n);
      }
      salt += kBatches * 100 + 1000;
      const double p50 = Percentile(per_task, 0.50);
      std::printf("%-6s %6u %12.1f %12.2f\n", "put", n, p50,
                  GbPerSec(n * kSlotBytes, p50));
    }
    // Gets: K matches pinned, K loads into disjoint slot ranges.
    {
      std::vector<double> per_task;
      per_task.reserve(kBatches);
      for (int batch = 0; batch < kBatches; ++batch) {
        std::vector<stratakv::metadata::MatchResult> matches;
        std::vector<TaskHandle> handles;
        const double t0 = NowUs();
        for (int k = 0; k < kPipelinedK; ++k) {
          matches.push_back(stack.plane().MatchForGet(
              std::span<const TokenId>(tokens.data(),
                                       static_cast<std::size_t>(n) * kTpb)));
          std::vector<BlockId> dst(n);
          for (std::uint32_t b = 0; b < n; ++b) {
            dst[b] = static_cast<BlockId>(k) * 64 + b;
          }
          handles.push_back(stack.engine().SubmitGet(matches.back(), dst));
        }
        stack.scheduler().Flush();
        for (TaskHandle handle : handles) {
          if (WaitFor(stack.engine(), handle) != TaskState::kSucceeded) {
            std::fprintf(stderr, "pipelined get failed\n");
            return 1;
          }
        }
        per_task.push_back((NowUs() - t0) /
                           static_cast<double>(kPipelinedK));
      }
      const double p50 = Percentile(per_task, 0.50);
      std::printf("%-6s %6u %12.1f %12.2f\n", "get", n, p50,
                  GbPerSec(n * kSlotBytes, p50));
    }
  }

  // Release the prime pin before the stack tears down.
  stack.plane().Release(held_pin);

  // ---- Fixed overhead: least squares over the sync rows ----
  auto fit = [](const std::vector<Row>& rows) {
    std::vector<double> xs, ys;
    for (const Row& row : rows) {
      xs.push_back(row.blocks * kSlotBytes);
      ys.push_back(row.p50_us);
    }
    const std::size_t n = xs.size();
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (std::size_t i = 0; i < n; ++i) {
      sx += xs[i];
      sy += ys[i];
      sxx += xs[i] * xs[i];
      sxy += xs[i] * ys[i];
    }
    const double denom = static_cast<double>(n) * sxx - sx * sx;
    const double b =
        denom == 0 ? 0 : (static_cast<double>(n) * sxy - sx * sy) / denom;
    const double a = (sy - b * sx) / static_cast<double>(n);
    return std::pair<double, double>(a, b);
  };
  std::printf("\n== Per-task fixed overhead (0-byte intercept, sync p50) ==\n");
  std::printf("%-6s %14s %16s\n", "task", "intercept_us", "marginal_GB/s");
  const auto [put_a, put_b] = fit(put_rows);
  const auto [get_a, get_b] = fit(get_rows);
  std::printf("%-6s %14.1f %16.2f\n", "put", put_a,
              put_b > 0 ? GbPerSec(1.0, put_b) : 0.0);
  std::printf("%-6s %14.1f %16.2f\n", "get", get_a,
              get_b > 0 ? GbPerSec(1.0, get_b) : 0.0);

  // ---- Reference lines (baseline: results/2026-11-19_wsl2_rtx4080super) ----
  std::printf("\n== References ==\n");
  std::printf("acceptance bar (roadmap E): per-task fixed overhead <= 1000 us "
              "at 4 blocks\n");
  const double* put4 = nullptr;
  const double* get4 = nullptr;
  for (std::size_t i = 0; i < put_rows.size(); ++i) {
    if (put_rows[i].blocks == 4) put4 = &put_rows[i].p50_us;
    if (get_rows[i].blocks == 4) get4 = &get_rows[i].p50_us;
  }
  if (put4 != nullptr && get4 != nullptr) {
    const double worst = std::max(*put4, *get4);
    std::printf("measured 4-block p50: put %.1f us | get %.1f us -> %s"
                " (bar 1000 us)\n",
                *put4, *get4, worst <= 1000.0 ? "PASS" : "FAIL");
  }
  std::printf("Phase 0 raw kernel floor (same box): h2d 74.9 / d2h 82.9 us "
              "at 4 blocks\n");
  std::printf("FlexKV v1.0.0 per-task (E0, remote 3080 Ti): 10850-21040 us "
              "at ~4 blocks\n");
  std::printf("verifications passed\n");
  return 0;
}
