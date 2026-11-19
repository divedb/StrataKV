#pragma once

#include <chrono>

#include "stratakv/common/handles.h"
#include "stratakv/control/task.h"

namespace stratakv::control {

/// \brief Receiver of op completions from an executor (design.md Phase
/// E CompletionSink, lifted into control so the seam is Phase-D
/// testable).
///
/// Implementations must be safe to call from executor threads.
class CompletionSink {
 public:
  virtual ~CompletionSink() = default;

  /// \brief One op finished. `op_id` was assigned by TaskEngine;
  /// `ok` false signals an executor-side failure for the owning task.
  virtual void OnOpCompleted(std::uint64_t op_id, bool ok) = 0;
};

/// \brief Execution seam between the control plane and a data plane.
///
/// Phase E's data::TransferScheduler (CUDA streams, or the isolated
/// process per ADR 0002) implements this; Phase D ships InlineExecutor
/// (simulation, below) plus test fakes. Contract: Execute returns
/// quickly (enqueue only, never blocks); op completions are delivered
/// asynchronously to the sink from any executor thread, exactly once
/// per op; buffers/refs addressed by the graph stay valid until the
/// last completion is delivered. Cancel is best-effort; the engine
/// tolerates late completions for cancelled tasks.
class ITaskExecutor {
 public:
  virtual ~ITaskExecutor() = default;

  /// \brief Enqueue a graph for execution on behalf of `owner`.
  virtual void Execute(const TransferGraph& graph,
                       common::TaskHandle owner, CompletionSink& sink) = 0;

  /// \brief Best-effort cancellation of a launched task. Returns true
  /// if the executor will drop the task's remaining ops. Late
  /// completions are always tolerated by the engine.
  virtual bool Cancel(common::TaskHandle owner) {
    (void)owner;
    return false;
  }
};

/// \brief Synchronous simulation executor (no data movement).
///
/// Completes every op inline on the calling thread, optionally after a
/// fixed per-op delay (transfer-latency simulation for scheduler-side
/// integration tests and the GIL heartbeat check). Intended for
/// control-plane tests and the CPU-only Python facade: the metadata
/// lifecycle (staging, readiness, pins, recycling) is exercised end to
/// end, but NO bytes are copied — the Phase E CUDA transport is the
/// first executor that moves data.
class InlineExecutor final : public ITaskExecutor {
 public:
  explicit InlineExecutor(
      std::chrono::milliseconds per_op_delay = std::chrono::milliseconds(0))
      : per_op_delay_(per_op_delay) {}

  void Execute(const TransferGraph& graph, common::TaskHandle owner,
               CompletionSink& sink) override;

  bool Cancel(common::TaskHandle owner) override {
    (void)owner;
    return false;  // Everything already completed inline.
  }

 private:
  std::chrono::milliseconds per_op_delay_;
};

}  // namespace stratakv::control
