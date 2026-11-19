#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <cuda_runtime.h>

#include "stratakv/common/types.h"
#include "stratakv/data/registry.h"
#include "stratakv/data/tier_table.h"
#include "stratakv/data/transport.h"

namespace stratakv::data {

/// \brief Stream-pool CUDA transport with event-based completion
/// (design.md Phase E CudaStreamTransport; transfer.cu harvest — one
/// strided-kernel launch per op, ids uploaded to a scratch arena).
///
/// Requires STRATAKV_GPU (pulls <cuda_runtime.h>); CPU-only builds use
/// the transport.h fakes and never include this header.
///
/// Copy-path policy (Phase 0 baseline findings, benchmarks/
/// phase0_gpu_baseline.md):
/// - Both endpoints SM-accessible (device memory; cudaMallocHost CPU
///   tier): ONE kernel launch per op regardless of block count and
///   layer segmentation (F2: per-(block,layer) memcpy decompositions
///   cap at ~2 GB/s; the kernel reaches link class).
/// - Otherwise (cudaHostRegister/pageable CPU tiers — SM access is
///   platform-dependent, faults on WSL2 per F4): copy-engine
///   cudaMemcpyAsync per (block, layer-chunk) — correct everywhere,
///   slower; page-locked cudaMallocHost tiers are the fast path.
///
/// Completion: one CUDA event per op; a single worker thread launches
/// batches and polls events, delivering OnOpCompleted(op_id, ok) from
/// that thread (exactly once per op — the TaskEngine tolerates any
/// executor thread). CancelPending: ops are launched promptly on
/// arrival; there is no transport-level queue to drop, so it returns 0
/// (cancellation happens scheduler-side, before Enqueue).
///
/// Thread safety: Enqueue safe from any engine thread; the TierTable
/// snapshot is taken at construction and read without locks afterwards.
class CudaStreamTransport final : public ITransport {
 public:
  struct Config {
    /// \brief Streams in the pool (round-robin per op). >1 lets
    /// independent ops overlap on the device.
    std::uint32_t streams = 2;
    /// \brief Kernel launch geometry (threads per block).
    std::uint32_t thread_block = 256;
    /// \brief Device scratch arena for op parameters/ids, bytes. Ops
    /// beyond it fall back to per-batch temporary allocations.
    std::uint64_t scratch_bytes = 8ull << 20;
  };

  /// \param table Tier-local id addressing (copied; the memory it
  /// points at — CpuTier pools, registered GPU buffers — must outlive
  /// the transport).
  ///
  /// \throws std::runtime_error when no usable CUDA device exists.
  ///
  /// \note The default-config path is a separate overload (not a
  /// default argument): nvcc 12.0's host front end rejects NSDMIs in
  /// same-class default arguments.
  explicit CudaStreamTransport(TierTable table);
  /// \brief Construct a stream transport with explicit pool sizing.
  /// \param table Tier-local addressing; backing memory must outlive transport.
  /// \param config Pool sizing.
  CudaStreamTransport(TierTable table, Config config);
  ~CudaStreamTransport() override;

  CudaStreamTransport(const CudaStreamTransport&) = delete;
  CudaStreamTransport& operator=(const CudaStreamTransport&) = delete;

  /// \brief Non-blocking: validates the batch (throwing
  /// stratakv::LogicError on unknown tiers/groups, mismatched geometry,
  /// or SSD ops — E5 backends), then queues it for the worker.
  /// Completions arrive on the worker thread.
  void Enqueue(std::vector<control::TransferOp> batch,
               control::CompletionSink& sink) override;

  /// \brief Stop accepting work and wait (up to grace) for in-flight
  /// ops to complete. Idempotent; Enqueue throws afterwards.
  void DrainPending(std::chrono::milliseconds grace) override;

  std::size_t CancelPending(common::TaskHandle owner) override {
    (void)owner;
    return 0;  // Everything handed to us starts immediately.
  }

  /// \brief Ops launched but not yet completion-reported.
  std::size_t num_in_flight() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/// \brief CUDA IPC handle source (roadmap E1): real cudaIpcGetMemHandle
/// behind the registry seam.
///
/// \note cudaIpcGetMemHandle requires the view base to be the exact
/// cudaMalloc allocation (not an offset into one) — engine contract.
/// CUDA IPC is unsupported on some platforms (WSL2 among them); Export
/// fails loudly there (stratakv::LogicError) rather than stubbing.
class CudaIpcHandleSource final : public IIpcHandleSource {
 public:
  IpcHandle Export(const common::BufferView& view) override;
};

}  // namespace stratakv::data
