// Phase 0: GPU data-plane capability baseline (roadmap §5 benchmark
// policy; the environment record every later phase compares against).
//
// What this measures — the raw CUDA floor under the StrataKV data plane,
// with no StrataKV layers above it:
//
//   1. Peak contiguous copy bandwidth (pinned/pageable host, H2D/D2H,
//      D2D) at large payload — the link ceiling.
//   2. Task-shaped transfers at the E0 block geometry: Qwen2-0.5B-shaped
//      block = 24 layers x 8 KiB = 192 KiB (4 blocks ~ 0.75 MB, the
//      payload FlexKV v1.0.0 spent 10-20 ms per task on). Three methods:
//        contig    one cudaMemcpyAsync of the whole payload (ceiling)
//        memcpy24  per-(block,layer) 8 KiB memcpy loop (the API-heavy
//                  decomposition Appendix B attributes to v1.0.0)
//        kernel    one strided gather/scatter kernel launch (the
//                  transfer.cu harvest design point)
//      x host memory kind {pinned (cudaMallocHost), registered
//      (cudaHostRegister), pageable} x direction x {1,4,16,64} blocks,
//      in two timing modes:
//        sync       submit -> event -> synchronize per task (the number
//                   comparable to FlexKV's per-task cost)
//        pipelined  K tasks enqueued back-to-back, one sync (the
//                   drain-and-batch throughput of the scheduler)
//   3. Per-task fixed overhead: least-squares fit of sync-mode task time
//      vs payload, extrapolated to 0 bytes (Phase E acceptance metric).
//   4. Micro costs: empty event task, no-op kernel task, submit-only
//      memcpy/kernel-launch API cost.
//
// Every timed row is verified with XXH3 against a deterministic golden
// pattern first; a row that fails verification aborts the run (no
// invented numbers). Timing executable only, NOT a ctest entry, per the
// benchmark policy in roadmap.md (latency tables are read by humans).
//
// Platform probe: on some environments (first observed: WSL2 GPU-PV,
// driver 610.88) SM/kernel access to cudaHostRegister'd or pageable
// host memory faults with an illegal memory access while copy-engine
// (cudaMemcpyAsync) access to the same memory works. The benchmark
// probes each memory kind in a CHILD process before the parent touches
// CUDA (a fault poisons the context for the process's lifetime, and on
// WSL2 even cudaDeviceReset recovery fails with "device busy"), then
// gates kernel scenarios accordingly, printing SKIP(platform) rows.
//
// Requires STRATAKV_GPU=ON. Run: ./build-gpu/benchmarks/stratakv_gpu_baseline

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

#include "xxhash.h"

namespace {

// ---- Block geometry: Qwen2-0.5B-shaped, matches E0 ground truth ----
// 2 (K,V) x 2 kv_heads x 64 head_size x 2 B fp16 x 16 tokens = 8 KiB
// per layer per block; 24 layers -> 192 KiB per model-wide block.
constexpr std::uint32_t kLayers = 24;
constexpr std::uint32_t kLayerSlice = 8192;
constexpr std::uint32_t kBlockBytes = kLayers * kLayerSlice;
constexpr std::uint32_t kMaxBlocks = 64;
// Largest payload (64 blocks = 12.58 MB) fits 32-bit arithmetic; the
// kernels index in 32-bit to keep per-word div/mod off the slow path.
static_assert(kMaxBlocks * kBlockBytes < (1u << 31), "32-bit payload");

constexpr int kThreadBlock = 256;
constexpr double kTargetScenarioMs = 200.0;  // Adaptive iteration target.
constexpr double kProbeSkipMs = 1000.0;      // Skip rows this slow on probe.

#define CUDA_CHECK(expr)                                              \
  do {                                                                \
    cudaError_t err_ = (expr);                                        \
    if (err_ != cudaSuccess) {                                        \
      std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #expr,      \
                   __FILE__, __LINE__, cudaGetErrorString(err_));     \
      std::exit(1);                                                   \
    }                                                                 \
  } while (0)

[[noreturn]] void Fail(const std::string& what) {
  std::fprintf(stderr, "FATAL: %s\n", what.c_str());
  std::exit(1);
}

// ---- Deterministic golden pattern (splitmix64 over 64-bit words) ----
std::uint64_t SplitMix64(std::uint64_t& state) {
  state += 0x9E3779B97F4A7C15ULL;
  std::uint64_t z = state;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

// ---- Kernels ----

__global__ void NoopKernel() {}

// Probe kernel: touch one word of host memory from an SM.
__global__ void ReadFirstWord(const unsigned long long* __restrict__ src,
                              unsigned long long* __restrict__ dst) {
  dst[0] = src[0];
}

// One task = one launch: copies `words` 64-bit words between the
// LAYERWISE GPU pools (per-layer base pointers) and a BLOCKWISE flat
// host pool laid out [block][layer][slice] (Appendix B heritage: the
// CPU tier stays one flat pool; GPU tensors arrive per layer).
__global__ void StridedCopyD2H(const char* const* __restrict__ layer_bases,
                               char* __restrict__ host_dst,
                               std::uint32_t words) {
  const std::uint32_t i =
      blockIdx.x * static_cast<std::uint32_t>(blockDim.x) + threadIdx.x;
  if (i >= words) return;
  const std::uint32_t byte = i * 8u;
  const std::uint32_t block = byte / kBlockBytes;
  const std::uint32_t in_block = byte - block * kBlockBytes;
  const std::uint32_t layer = in_block / kLayerSlice;
  const std::uint32_t off = in_block - layer * kLayerSlice;
  *reinterpret_cast<unsigned long long*>(host_dst + byte) =
      *reinterpret_cast<const unsigned long long*>(
          layer_bases[layer] + block * kLayerSlice + off);
}

__global__ void StridedCopyH2D(const char* __restrict__ host_src,
                               char* const* __restrict__ layer_bases,
                               std::uint32_t words) {
  const std::uint32_t i =
      blockIdx.x * static_cast<std::uint32_t>(blockDim.x) + threadIdx.x;
  if (i >= words) return;
  const std::uint32_t byte = i * 8u;
  const std::uint32_t block = byte / kBlockBytes;
  const std::uint32_t in_block = byte - block * kBlockBytes;
  const std::uint32_t layer = in_block / kLayerSlice;
  const std::uint32_t off = in_block - layer * kLayerSlice;
  *reinterpret_cast<unsigned long long*>(layer_bases[layer] +
                                         block * kLayerSlice + off) =
      *reinterpret_cast<const unsigned long long*>(host_src + byte);
}

// Overwrite the LAYERWISE GPU pools (pre-verify scrub so stale golden
// bytes cannot make a broken copy look correct).
__global__ void ScrubLayerwise(char* const* __restrict__ layer_bases,
                               std::uint32_t words) {
  const std::uint32_t i =
      blockIdx.x * static_cast<std::uint32_t>(blockDim.x) + threadIdx.x;
  if (i >= words) return;
  const std::uint32_t byte = i * 8u;
  const std::uint32_t block = byte / kBlockBytes;
  const std::uint32_t in_block = byte - block * kBlockBytes;
  const std::uint32_t layer = in_block / kLayerSlice;
  const std::uint32_t off = in_block - layer * kLayerSlice;
  *reinterpret_cast<unsigned long long*>(layer_bases[layer] +
                                         block * kLayerSlice + off) =
      0xA5A5A5A5A5A5A5A5ULL;
}

// ---- Small RAII owners ----

struct DeviceBuffer {
  void* ptr = nullptr;
  std::uint64_t bytes = 0;
  explicit DeviceBuffer(std::uint64_t n) : bytes(n) {
    CUDA_CHECK(cudaMalloc(&ptr, n));
  }
  ~DeviceBuffer() { cudaFree(ptr); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

struct HostBuffer {  // Pageable, 4096-aligned (register-compatible).
  void* ptr = nullptr;
  std::uint64_t bytes = 0;
  explicit HostBuffer(std::uint64_t n) : bytes(n) {
    ptr = std::aligned_alloc(4096, ((n + 4095) / 4096) * 4096);
    if (ptr == nullptr) Fail("aligned_alloc failed");
  }
  ~HostBuffer() { std::free(ptr); }
  HostBuffer(const HostBuffer&) = delete;
  HostBuffer& operator=(const HostBuffer&) = delete;
};

struct PinnedBuffer {
  void* ptr = nullptr;
  std::uint64_t bytes = 0;
  explicit PinnedBuffer(std::uint64_t n) : bytes(n) {
    CUDA_CHECK(cudaMallocHost(&ptr, n));
  }
  ~PinnedBuffer() { cudaFreeHost(ptr); }
  PinnedBuffer(const PinnedBuffer&) = delete;
  PinnedBuffer& operator=(const PinnedBuffer&) = delete;
};

// ---- Statistics helpers (index_bench lineage) ----

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

// One timed unit of work: enqueue onto the stream; the harness owns the
// event record + synchronize so sync/pipelined modes share one shape.
using TaskFn = std::function<void(cudaStream_t)>;

struct SyncStats {
  double p50_us = 0;
  double p95_us = 0;
  double p99_us = 0;
  bool skipped = false;
};

// Warm up, probe once (slow probes -> SKIP), then time `iters`
// submit -> completion rounds.
SyncStats MeasureSync(const TaskFn& task, cudaStream_t stream,
                      cudaEvent_t done) {
  for (int i = 0; i < 3; ++i) {
    task(stream);
    CUDA_CHECK(cudaEventRecord(done, stream));
    CUDA_CHECK(cudaEventSynchronize(done));
  }
  const double probe_t0 = NowUs();
  task(stream);
  CUDA_CHECK(cudaEventRecord(done, stream));
  CUDA_CHECK(cudaEventSynchronize(done));
  const double probe_ms = (NowUs() - probe_t0) / 1000.0;
  if (probe_ms > kProbeSkipMs) return SyncStats{.skipped = true};
  int iters = static_cast<int>(kTargetScenarioMs / probe_ms);
  iters = std::clamp(iters, 10, 400);
  std::vector<double> micros;
  micros.reserve(static_cast<std::size_t>(iters));
  for (int i = 0; i < iters; ++i) {
    const double t0 = NowUs();
    task(stream);
    CUDA_CHECK(cudaEventRecord(done, stream));
    CUDA_CHECK(cudaEventSynchronize(done));
    micros.push_back(NowUs() - t0);
  }
  return SyncStats{.p50_us = Percentile(micros, 0.50),
                   .p95_us = Percentile(micros, 0.95),
                   .p99_us = Percentile(micros, 0.99)};
}

// Warm up, then time `batches` rounds of K back-to-back tasks with one
// sync per round; returns per-task p50/p99 microseconds.
struct PipelinedStats {
  double p50_us = 0;
  double p99_us = 0;
  bool skipped = false;
};

PipelinedStats MeasurePipelined(const TaskFn& task, cudaStream_t stream,
                                cudaEvent_t done) {
  for (int i = 0; i < 3; ++i) {
    task(stream);
    CUDA_CHECK(cudaEventRecord(done, stream));
    CUDA_CHECK(cudaEventSynchronize(done));
  }
  const double probe_t0 = NowUs();
  task(stream);
  CUDA_CHECK(cudaEventRecord(done, stream));
  CUDA_CHECK(cudaEventSynchronize(done));
  const double probe_ms = (NowUs() - probe_t0) / 1000.0;
  if (probe_ms > kProbeSkipMs) return PipelinedStats{.skipped = true};
  int k = static_cast<int>(kTargetScenarioMs / probe_ms);
  k = std::clamp(k, 8, 256);
  constexpr int kBatches = 7;
  std::vector<double> per_task;
  per_task.reserve(kBatches);
  for (int b = 0; b < kBatches; ++b) {
    const double t0 = NowUs();
    for (int i = 0; i < k; ++i) {
      task(stream);
    }
    CUDA_CHECK(cudaEventRecord(done, stream));
    CUDA_CHECK(cudaEventSynchronize(done));
    per_task.push_back((NowUs() - t0) / static_cast<double>(k));
  }
  return PipelinedStats{.p50_us = Percentile(per_task, 0.50),
                        .p99_us = Percentile(per_task, 0.99)};
}

// Least-squares fit y = a + b*x; returns {a, b}.
std::pair<double, double> LinearFit(const std::vector<double>& xs,
                                    const std::vector<double>& ys) {
  const std::size_t n = xs.size();
  double sx = 0, sy = 0, sxx = 0, sxy = 0;
  for (std::size_t i = 0; i < n; ++i) {
    sx += xs[i];
    sy += ys[i];
    sxx += xs[i] * xs[i];
    sxy += xs[i] * ys[i];
  }
  const double denom = static_cast<double>(n) * sxx - sx * sx;
  const double b = denom == 0.0 ? 0.0
                                : (static_cast<double>(n) * sxy - sx * sy) /
                                      denom;
  const double a = (sy - b * sx) / static_cast<double>(n);
  return {a, b};
}

// ---- Benchmark state ----

enum class Method { kContig, kMemcpy24, kKernel };
enum class Dir { kH2D, kD2H };
enum class HostMem { kPinned, kRegistered, kPageable };

/// \brief Which host memory kinds the SM can touch on this platform.
struct SmCaps {
  bool registered = true;
  bool pageable = true;
};

// Probe one memory kind with a 1-word kernel read, in a forked child.
// Must run before the parent's first CUDA call (fork-before-CUDA is the
// safe direction); the child's CUDA state — including any sticky fault
// — dies with it.
bool ProbeSmAccess(bool register_it) {
  constexpr std::size_t kProbeBytes = 64 * 1024;
  int fds[2];
  if (pipe(fds) != 0) return true;  // Cannot probe: assume capable.
  std::fflush(stdout);
  std::fflush(stderr);
  const pid_t pid = fork();
  if (pid == 0) {  // Child: probe, report one byte, exit.
    bool ok = false;
    void* p = std::aligned_alloc(4096, kProbeBytes);
    if (p != nullptr) {
      std::memset(p, 0x11, kProbeBytes);
      if (!register_it ||
          cudaHostRegister(p, kProbeBytes, cudaHostRegisterDefault) ==
              cudaSuccess) {
        void* d = nullptr;
        if (cudaMalloc(&d, sizeof(unsigned long long)) == cudaSuccess) {
          ReadFirstWord<<<1, 32>>>(
              static_cast<const unsigned long long*>(p),
              static_cast<unsigned long long*>(d));
          ok = cudaDeviceSynchronize() == cudaSuccess;
        }
      }
    }
    const char byte = ok ? 1 : 0;
    const ssize_t sent = write(fds[1], &byte, 1);
    (void)sent;  // Best effort; parent treats a short read as "faulted".
    _exit(0);    // Skip atexit/static-destructor re-runs in the child.
  }
  char byte = 0;
  const ssize_t got = read(fds[0], &byte, 1);
  int status = 0;
  (void)waitpid(pid, &status, 0);
  (void)close(fds[0]);
  (void)close(fds[1]);
  return got == 1 && byte == 1;
}

const char* MethodName(Method m) {
  switch (m) {
    case Method::kContig: return "contig";
    case Method::kMemcpy24: return "memcpy24";
    case Method::kKernel: return "kernel";
  }
  return "?";
}

const char* MemName(HostMem m) {
  switch (m) {
    case HostMem::kPinned: return "pinned";
    case HostMem::kRegistered: return "registered";
    case HostMem::kPageable: return "pageable";
  }
  return "?";
}

struct World {
  cudaStream_t stream;
  cudaEvent_t done;

  // Host pools (BLOCKWISE flat [block][layer][slice]), all seeded golden.
  PinnedBuffer pinned{kMaxBlocks * kBlockBytes};
  HostBuffer registered{kMaxBlocks * kBlockBytes};
  HostBuffer pageable{kMaxBlocks * kBlockBytes};

  // LAYERWISE GPU pools: one allocation carved per layer, seeded golden.
  DeviceBuffer dev_layerwise{kMaxBlocks * kBlockBytes};
  char* layer_base[kLayers];
  DeviceBuffer layer_ptrs{sizeof(char*) * kLayers};

  // Flat GPU mirror (contig scenario), seeded golden.
  DeviceBuffer dev_flat{kMaxBlocks * kBlockBytes};

  // Trusted gather-back staging (H2D verification).
  PinnedBuffer staging{kMaxBlocks * kBlockBytes};

  // Golden pattern + prefix hash per block count.
  HostBuffer golden{kMaxBlocks * kBlockBytes};
  std::vector<std::uint64_t> golden_hash;

  char* HostPool(HostMem m) const {
    switch (m) {
      case HostMem::kPinned: return static_cast<char*>(pinned.ptr);
      case HostMem::kRegistered: return static_cast<char*>(registered.ptr);
      case HostMem::kPageable: return static_cast<char*>(pageable.ptr);
    }
    return nullptr;
  }

  void SeedAndHash() {
    std::uint64_t state = 0x1234567890ABCDEFULL;
    auto* words = static_cast<std::uint64_t*>(golden.ptr);
    const std::size_t n_words =
        static_cast<std::size_t>(kMaxBlocks) * kBlockBytes / 8;
    for (std::size_t i = 0; i < n_words; ++i) {
      words[i] = SplitMix64(state);
    }
    // Host pools start identical to golden (H2D transports golden bytes).
    std::memcpy(pinned.ptr, golden.ptr, pinned.bytes);
    std::memcpy(registered.ptr, golden.ptr, registered.bytes);
    std::memcpy(pageable.ptr, golden.ptr, pageable.bytes);
    // LAYERWISE pools: per-(block,layer) trusted copies from golden.
    auto* base = static_cast<char*>(dev_layerwise.ptr);
    for (std::uint32_t l = 0; l < kLayers; ++l) {
      layer_base[l] = base + static_cast<std::size_t>(l) * kMaxBlocks *
                                 kLayerSlice;
    }
    for (std::uint32_t b = 0; b < kMaxBlocks; ++b) {
      for (std::uint32_t l = 0; l < kLayers; ++l) {
        const void* src = static_cast<char*>(golden.ptr) +
                          static_cast<std::size_t>(b) * kBlockBytes +
                          static_cast<std::size_t>(l) * kLayerSlice;
        void* dst =
            layer_base[l] + static_cast<std::size_t>(b) * kLayerSlice;
        CUDA_CHECK(cudaMemcpy(dst, src, kLayerSlice,
                              cudaMemcpyHostToDevice));
      }
    }
    CUDA_CHECK(cudaMemcpy(layer_ptrs.ptr, layer_base, sizeof(layer_base),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_flat.ptr, golden.ptr, dev_flat.bytes,
                          cudaMemcpyHostToDevice));
    golden_hash.resize(kMaxBlocks);
    for (std::uint32_t b = 1; b <= kMaxBlocks; ++b) {
      golden_hash[b - 1] =
          XXH3_64bits(golden.ptr, static_cast<std::size_t>(b) * kBlockBytes);
    }
  }
};

TaskFn MakeTask(const World& w, Method method, Dir dir, HostMem mem,
                std::uint32_t blocks) {
  char* host = w.HostPool(mem);
  const std::uint32_t bytes = blocks * kBlockBytes;
  const std::uint32_t words = bytes / 8;
  const unsigned int grid = (words + kThreadBlock - 1) / kThreadBlock;
  const auto* layer_ro =
      static_cast<const char* const*>(w.layer_ptrs.ptr);
  auto* layer_rw = static_cast<char* const*>(w.layer_ptrs.ptr);

  if (dir == Dir::kD2H) {
    switch (method) {
      case Method::kContig:
        return [bytes, flat = static_cast<char*>(w.dev_flat.ptr), host](
                   cudaStream_t s) {
          CUDA_CHECK(cudaMemcpyAsync(host, flat, bytes,
                                     cudaMemcpyDeviceToHost, s));
        };
      case Method::kMemcpy24:
        return [blocks, host, &w](cudaStream_t s) {
          for (std::uint32_t b = 0; b < blocks; ++b) {
            for (std::uint32_t l = 0; l < kLayers; ++l) {
              CUDA_CHECK(cudaMemcpyAsync(
                  host + static_cast<std::size_t>(b) * kBlockBytes +
                      static_cast<std::size_t>(l) * kLayerSlice,
                  w.layer_base[l] + static_cast<std::size_t>(b) * kLayerSlice,
                  kLayerSlice, cudaMemcpyDeviceToHost, s));
            }
          }
        };
      case Method::kKernel:
        return [words, grid, layer_ro, host](cudaStream_t s) {
          StridedCopyD2H<<<grid, kThreadBlock, 0, s>>>(layer_ro, host,
                                                       words);
          CUDA_CHECK(cudaGetLastError());
        };
    }
  } else {
    switch (method) {
      case Method::kContig:
        return [bytes, flat = static_cast<char*>(w.dev_flat.ptr), host](
                   cudaStream_t s) {
          CUDA_CHECK(cudaMemcpyAsync(flat, host, bytes,
                                     cudaMemcpyHostToDevice, s));
        };
      case Method::kMemcpy24:
        return [blocks, host, &w](cudaStream_t s) {
          for (std::uint32_t b = 0; b < blocks; ++b) {
            for (std::uint32_t l = 0; l < kLayers; ++l) {
              CUDA_CHECK(cudaMemcpyAsync(
                  w.layer_base[l] + static_cast<std::size_t>(b) * kLayerSlice,
                  host + static_cast<std::size_t>(b) * kBlockBytes +
                      static_cast<std::size_t>(l) * kLayerSlice,
                  kLayerSlice, cudaMemcpyHostToDevice, s));
            }
          }
        };
      case Method::kKernel:
        return [words, grid, layer_rw, host](cudaStream_t s) {
          StridedCopyH2D<<<grid, kThreadBlock, 0, s>>>(host, layer_rw,
                                                       words);
          CUDA_CHECK(cudaGetLastError());
        };
    }
  }
  Fail("unreachable task shape");
}

// Run one op, synchronize, and compare destination bytes against the
// golden prefix hash. H2D destinations are gathered back with trusted
// synchronous memcpys before hashing.
void Verify(const World& w, Method method, Dir dir, HostMem mem,
            std::uint32_t blocks) {
  char* host = w.HostPool(mem);
  const std::uint32_t bytes = blocks * kBlockBytes;
  const std::uint32_t words = bytes / 8;
  const unsigned int grid = (words + kThreadBlock - 1) / kThreadBlock;
  const std::uint64_t want = w.golden_hash[blocks - 1];

  if (dir == Dir::kD2H) {
    std::memset(host, 0xA5, bytes);  // Scrub; cannot equal golden.
    if (method == Method::kContig) {
      CUDA_CHECK(cudaMemcpyAsync(host, w.dev_flat.ptr, bytes,
                                 cudaMemcpyDeviceToHost, w.stream));
    } else if (method == Method::kMemcpy24) {
      for (std::uint32_t b = 0; b < blocks; ++b) {
        for (std::uint32_t l = 0; l < kLayers; ++l) {
          CUDA_CHECK(cudaMemcpyAsync(
              host + static_cast<std::size_t>(b) * kBlockBytes +
                  static_cast<std::size_t>(l) * kLayerSlice,
              w.layer_base[l] + static_cast<std::size_t>(b) * kLayerSlice,
              kLayerSlice, cudaMemcpyDeviceToHost, w.stream));
        }
      }
    } else {
      StridedCopyD2H<<<grid, kThreadBlock, 0, w.stream>>>(
          static_cast<const char* const*>(w.layer_ptrs.ptr), host, words);
      CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaEventRecord(w.done, w.stream));
    CUDA_CHECK(cudaEventSynchronize(w.done));
    if (XXH3_64bits(host, bytes) != want) {
      Fail("D2H verify FAILED for " + std::string(MethodName(method)) +
           "/" + MemName(mem));
    }
    return;
  }

  // H2D: scrub GPU destination, run the op, gather back, hash.
  if (method == Method::kContig) {
    CUDA_CHECK(cudaMemsetAsync(w.dev_flat.ptr, 0xA5, bytes, w.stream));
    CUDA_CHECK(cudaMemcpyAsync(w.dev_flat.ptr, host, bytes,
                               cudaMemcpyHostToDevice, w.stream));
    CUDA_CHECK(cudaEventRecord(w.done, w.stream));
    CUDA_CHECK(cudaEventSynchronize(w.done));
    CUDA_CHECK(cudaMemcpy(w.staging.ptr, w.dev_flat.ptr, bytes,
                          cudaMemcpyDeviceToHost));
  } else {
    ScrubLayerwise<<<grid, kThreadBlock, 0, w.stream>>>(
        static_cast<char* const*>(w.layer_ptrs.ptr), words);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(w.done, w.stream));
    CUDA_CHECK(cudaEventSynchronize(w.done));
    MakeTask(w, method, dir, mem, blocks)(w.stream);
    CUDA_CHECK(cudaEventRecord(w.done, w.stream));
    CUDA_CHECK(cudaEventSynchronize(w.done));
    // Trusted gather-back into BLOCKWISE staging layout.
    for (std::uint32_t b = 0; b < blocks; ++b) {
      for (std::uint32_t l = 0; l < kLayers; ++l) {
        CUDA_CHECK(cudaMemcpy(
            static_cast<char*>(w.staging.ptr) +
                static_cast<std::size_t>(b) * kBlockBytes +
                static_cast<std::size_t>(l) * kLayerSlice,
            w.layer_base[l] + static_cast<std::size_t>(b) * kLayerSlice,
            kLayerSlice, cudaMemcpyDeviceToHost));
      }
    }
  }
  if (XXH3_64bits(w.staging.ptr, bytes) != want) {
    Fail("H2D verify FAILED for " + std::string(MethodName(method)) +
         "/" + MemName(mem));
  }
}

double GbPerSec(double bytes, double us) {
  return us <= 0.0 ? 0.0 : bytes / (us * 1000.0);  // Decimal GB/s.
}

void PrintEnvironment(const SmCaps& caps) {
  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  int driver = 0, runtime = 0;
  CUDA_CHECK(cudaDriverGetVersion(&driver));
  CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
  std::printf("== Environment ==\n");
  std::printf("device %d: %s (compute capability %d.%d)\n", dev, prop.name,
              prop.major, prop.minor);
  std::printf("driver API %d.%d | runtime API %d.%d\n", driver / 1000,
              driver % 1000, runtime / 1000, runtime % 1000);
  std::printf("sm clock max %d MHz | mem clock max %d MHz | bus %d-bit | "
              "L2 %d KiB\n",
              prop.clockRate / 1000, prop.memoryClockRate / 1000,
              prop.memoryBusWidth,
              static_cast<int>(prop.l2CacheSize / 1024));
  std::printf("ECC %s | global mem %.2f GiB\n",
              prop.ECCEnabled != 0 ? "on" : "off",
              static_cast<double>(prop.totalGlobalMem) /
                  (1024.0 * 1024.0 * 1024.0));
  std::printf("block geometry: %u layers x %u B = %u B per block "
              "(Qwen2-0.5B-shaped)\n",
              kLayers, kLayerSlice, kBlockBytes);
  std::printf("SM access probed: pinned=OK registered=%s pageable=%s"
              " (copy-engine access works for all kinds)\n",
              caps.registered ? "OK" : "FAULTS",
              caps.pageable ? "OK" : "FAULTS");
  std::printf("harness: benchmarks/gpu_baseline_bench.cu (methodology: "
              "benchmarks/phase0_gpu_baseline.md)\n\n");
}

// Peak contiguous reference: large pinned/pageable copies + D2D.
void RunPeakBandwidth() {
  std::printf("== Peak contiguous bandwidth (decimal GB/s, p50 of "
              "sync'd async copies) ==\n");
  constexpr std::uint64_t kBytes = 256ULL << 20;
  PinnedBuffer big_pinned{kBytes};
  HostBuffer big_pageable{kBytes};
  DeviceBuffer big_dev{kBytes};
  DeviceBuffer d2d_dst{kBytes};
  cudaStream_t s;
  CUDA_CHECK(cudaStreamCreate(&s));
  cudaEvent_t done;
  CUDA_CHECK(cudaEventCreate(&done));

  struct Row {
    const char* name;
    cudaMemcpyKind kind;
    const void* src;
    void* dst;
  };
  const std::vector<Row> rows = {
      {"h2d_pinned_256MB", cudaMemcpyHostToDevice, big_pinned.ptr,
       big_dev.ptr},
      {"d2h_pinned_256MB", cudaMemcpyDeviceToHost, big_dev.ptr,
       big_pinned.ptr},
      {"h2d_pageable_256MB", cudaMemcpyHostToDevice, big_pageable.ptr,
       big_dev.ptr},
      {"d2h_pageable_256MB", cudaMemcpyDeviceToHost, big_dev.ptr,
       big_pageable.ptr},
      {"d2d_256MB", cudaMemcpyDeviceToDevice, big_dev.ptr, d2d_dst.ptr},
  };
  std::printf("%-22s %12s %12s\n", "scenario", "p50_us", "GB/s");
  for (const Row& r : rows) {
    std::vector<double> micros;
    constexpr int kIters = 20;
    micros.reserve(kIters);
    for (int i = 0; i < kIters + 3; ++i) {
      const double t0 = NowUs();
      CUDA_CHECK(cudaMemcpyAsync(r.dst, r.src, kBytes, r.kind, s));
      CUDA_CHECK(cudaEventRecord(done, s));
      CUDA_CHECK(cudaEventSynchronize(done));
      if (i >= 3) micros.push_back(NowUs() - t0);
    }
    const double p50 = Percentile(micros, 0.50);
    std::printf("%-22s %12.1f %12.2f\n", r.name, p50,
                GbPerSec(static_cast<double>(kBytes), p50));
  }
  std::printf("\n");
  CUDA_CHECK(cudaEventDestroy(done));
  CUDA_CHECK(cudaStreamDestroy(s));
}

void RunTaskShaped(const World& w, const SmCaps& caps) {
  const std::vector<std::uint32_t> sizes = {1, 4, 16, 64};
  struct Combo {
    Method method;
    HostMem mem;
  };
  const std::vector<Combo> combos = {
      {Method::kContig, HostMem::kPinned},
      {Method::kKernel, HostMem::kPinned},
      {Method::kMemcpy24, HostMem::kPinned},
      {Method::kContig, HostMem::kRegistered},
      {Method::kKernel, HostMem::kRegistered},
      {Method::kContig, HostMem::kPageable},
      {Method::kKernel, HostMem::kPageable},
      {Method::kMemcpy24, HostMem::kPageable},
  };

  // Kernel scenarios need SM access to the host pool; probed at startup
  // (WSL2-class platforms fault on registered/pageable, see header).
  auto sm_ok = [caps](HostMem m) {
    switch (m) {
      case HostMem::kPinned: return true;
      case HostMem::kRegistered: return caps.registered;
      case HostMem::kPageable: return caps.pageable;
    }
    return false;
  };

  // Sync-mode pinned/kernel rows feed the fixed-overhead fit.
  std::vector<double> fit_bytes[2];
  std::vector<double> fit_us[2];

  std::printf("== Task-shaped transfers, SYNC mode (submit -> completion "
              "per task) ==\n");
  std::printf("%-34s %6s %10s %10s %10s %10s %8s\n", "scenario", "blocks",
              "bytes", "p50_us", "p95_us", "p99_us", "GB/s");
  for (Dir dir : {Dir::kH2D, Dir::kD2H}) {
    for (const Combo& c : combos) {
      for (std::uint32_t blocks : sizes) {
        const std::string name =
            std::string(dir == Dir::kH2D ? "h2d" : "d2h") + "/" +
            MemName(c.mem) + "/" + MethodName(c.method);
        if (c.method == Method::kKernel && !sm_ok(c.mem)) {
          std::printf("%-34s %6u %10u %10s %10s %10s %8s\n", name.c_str(),
                      blocks, blocks * kBlockBytes, "SKIP", "-", "-", "-");
          continue;
        }
        Verify(w, c.method, dir, c.mem, blocks);
        const TaskFn task = MakeTask(w, c.method, dir, c.mem, blocks);
        const SyncStats st = MeasureSync(task, w.stream, w.done);
        if (st.skipped) {
          std::printf("%-34s %6u %10u %10s %10s %10s %8s\n", name.c_str(),
                      blocks, blocks * kBlockBytes, "SKIP", "-", "-", "-");
          continue;
        }
        std::printf("%-34s %6u %10u %10.1f %10.1f %10.1f %8.2f\n",
                    name.c_str(), blocks, blocks * kBlockBytes, st.p50_us,
                    st.p95_us, st.p99_us,
                    GbPerSec(blocks * kBlockBytes, st.p50_us));
        if (c.mem == HostMem::kPinned && c.method == Method::kKernel) {
          const int d = dir == Dir::kH2D ? 0 : 1;
          fit_bytes[d].push_back(blocks * kBlockBytes);
          fit_us[d].push_back(st.p50_us);
        }
      }
    }
  }

  std::printf("\n== Task-shaped transfers, PIPELINED mode (K tasks "
              "back-to-back, one sync; drain-and-batch shape) ==\n");
  std::printf("%-34s %6s %10s %12s %12s\n", "scenario", "blocks", "bytes",
              "p50_us/task", "GB/s");
  for (Dir dir : {Dir::kH2D, Dir::kD2H}) {
    for (const Combo& c : combos) {
      for (std::uint32_t blocks : sizes) {
        const std::string name =
            std::string(dir == Dir::kH2D ? "h2d" : "d2h") + "/" +
            MemName(c.mem) + "/" + MethodName(c.method);
        if (c.method == Method::kKernel && !sm_ok(c.mem)) {
          std::printf("%-34s %6u %10u %12s %12s\n", name.c_str(), blocks,
                      blocks * kBlockBytes, "SKIP", "-");
          continue;
        }
        Verify(w, c.method, dir, c.mem, blocks);
        const TaskFn task = MakeTask(w, c.method, dir, c.mem, blocks);
        const PipelinedStats st = MeasurePipelined(task, w.stream, w.done);
        if (st.skipped) {
          std::printf("%-34s %6u %10u %12s %12s\n", name.c_str(), blocks,
                      blocks * kBlockBytes, "SKIP", "-");
          continue;
        }
        std::printf("%-34s %6u %10u %12.1f %12.2f\n", name.c_str(), blocks,
                    blocks * kBlockBytes, st.p50_us,
                    GbPerSec(blocks * kBlockBytes, st.p50_us));
      }
    }
  }

  std::printf("\n== Per-task fixed overhead (least squares on "
              "pinned/kernel SYNC rows; intercept = 0-byte task) ==\n");
  std::printf("%-6s %14s %16s\n", "dir", "intercept_us", "marginal_GB/s");
  for (int d = 0; d < 2; ++d) {
    if (fit_bytes[d].size() < 2) continue;
    const auto [a, b] = LinearFit(fit_bytes[d], fit_us[d]);
    std::printf("%-6s %14.1f %16.2f\n", d == 0 ? "h2d" : "d2h", a,
                b > 0.0 ? GbPerSec(1.0, b) : 0.0);
  }

  // Empty task: pure submit -> completion round trip (event only).
  const TaskFn empty = [](cudaStream_t s) { (void)s; };
  const SyncStats ev = MeasureSync(empty, w.stream, w.done);
  std::printf("measured empty task (event record + sync only): p50 %.1f "
              "us\n",
              ev.p50_us);
}

void RunMicro(const World& w) {
  std::printf("\n== Micro costs (p50 us) ==\n");
  const TaskFn empty = [](cudaStream_t s) { (void)s; };
  const SyncStats ev = MeasureSync(empty, w.stream, w.done);
  std::printf("%-40s %10.1f\n", "empty_event_task", ev.p50_us);

  const TaskFn noop = [](cudaStream_t s) {
    NoopKernel<<<1, kThreadBlock, 0, s>>>();
    CUDA_CHECK(cudaGetLastError());
  };
  const SyncStats nk = MeasureSync(noop, w.stream, w.done);
  std::printf("%-40s %10.1f\n", "noop_kernel_task", nk.p50_us);

  // Submit-only costs: time the API call loop, sync outside the timer.
  constexpr int kSubmits = 4096;
  {
    char* host = static_cast<char*>(w.pinned.ptr);
    for (int i = 0; i < 16; ++i) {
      CUDA_CHECK(cudaMemcpyAsync(w.dev_flat.ptr, host, kLayerSlice,
                                 cudaMemcpyHostToDevice, w.stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(w.stream));
    const double t0 = NowUs();
    for (int i = 0; i < kSubmits; ++i) {
      CUDA_CHECK(cudaMemcpyAsync(w.dev_flat.ptr, host, kLayerSlice,
                                 cudaMemcpyHostToDevice, w.stream));
    }
    const double submit_us = (NowUs() - t0) / kSubmits;
    CUDA_CHECK(cudaStreamSynchronize(w.stream));
    std::printf("%-40s %10.3f\n", "memcpy_async_8k_submit_only",
                submit_us);
  }
  {
    for (int i = 0; i < 16; ++i) {
      NoopKernel<<<1, kThreadBlock, 0, w.stream>>>();
      CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaStreamSynchronize(w.stream));
    const double t0 = NowUs();
    for (int i = 0; i < kSubmits; ++i) {
      NoopKernel<<<1, kThreadBlock, 0, w.stream>>>();
      CUDA_CHECK(cudaGetLastError());
    }
    const double launch_us = (NowUs() - t0) / kSubmits;
    CUDA_CHECK(cudaStreamSynchronize(w.stream));
    std::printf("%-40s %10.3f\n", "kernel_launch_submit_only", launch_us);
  }
}

}  // namespace

int main() {
  // Probes first: they fork (see ProbeSmAccess), which is only safe
  // before this process's first CUDA call.
  SmCaps caps;
  caps.registered = ProbeSmAccess(true);
  caps.pageable = ProbeSmAccess(false);

  PrintEnvironment(caps);

  World w;
  CUDA_CHECK(cudaStreamCreate(&w.stream));
  CUDA_CHECK(cudaEventCreate(&w.done));
  w.SeedAndHash();
  CUDA_CHECK(cudaHostRegister(w.registered.ptr, w.registered.bytes,
                              cudaHostRegisterDefault));

  RunPeakBandwidth();
  RunTaskShaped(w, caps);
  RunMicro(w);

  CUDA_CHECK(cudaHostUnregister(w.registered.ptr));
  CUDA_CHECK(cudaEventDestroy(w.done));
  CUDA_CHECK(cudaStreamDestroy(w.stream));
  std::printf("\nall verifications passed\n");
  return 0;
}
