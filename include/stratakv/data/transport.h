#pragma once

#include <chrono>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "stratakv/common/exceptions.h"
#include "stratakv/common/handles.h"
#include "stratakv/control/executor.h"
#include "stratakv/control/task.h"

namespace stratakv::data {

using stratakv::control::CompletionSink;
using stratakv::control::TransferOp;

/// \brief Mode selection (ADR 0002 — decided by the Phase E experiment
/// at gate G2; both modes share all scheduler logic).
enum class IsolationMode : std::uint8_t {
  kInProcess,
  kIsolatedProcess,
};

/// \brief Transport abstraction (design.md Phase E; CUDA streams v1,
/// GDS/NIXL deferred). Harvests the FlexKV worker contract.
///
/// Contract: Enqueue is non-blocking and coalesces the batch into the
/// transport's in-flight work; buffers/refs addressed by the ops stay
/// valid until completion is delivered to the sink (exactly once per
/// op, from any transport thread). CancelPending drops not-yet-started
/// ops and reports how many were dropped (their completions will never
/// arrive); started ops always complete.
class ITransport {
 public:
  virtual ~ITransport() = default;

  virtual void Enqueue(std::vector<TransferOp> batch,
                       CompletionSink& sink) = 0;

  /// \brief Shutdown helper: wait (up to grace) for started ops to
  /// complete; then return.
  virtual void DrainPending(std::chrono::milliseconds grace) = 0;

  /// \brief Best-effort drop of a task's not-yet-started ops.
  /// \return Number of dropped ops (0 = everything already started).
  virtual std::size_t CancelPending(common::TaskHandle owner) {
    (void)owner;
    return 0;
  }
};

/// \brief Scheduler wiring task graphs onto transports with
/// drain-and-batch coalescing (harvest of FlexKV worker.py's loop).
///
/// Batching policy: ops accumulate per owner; when the pending op count
/// reaches `max_batch_ops`, or Flush() is called, or the queue holds
/// ops from `max_pending_tasks` distinct owners, all pending ops are
/// coalesced into ONE transport Enqueue (one submission round trip per
/// batch — the lever that killed FlexKV v1.0.0's per-task costs).
///
/// Thread safety: all methods safe from any engine thread.
class TransferScheduler {
 public:
  struct Config {
    std::uint32_t max_batch_ops = 512;    ///< Flush threshold.
    std::uint32_t max_pending_tasks = 8;  ///< Owner-diversity flush.
  };

  TransferScheduler(std::unique_ptr<ITransport> transport,
                    IsolationMode mode, Config config);

  ~TransferScheduler();

  TransferScheduler(const TransferScheduler&) = delete;
  TransferScheduler& operator=(const TransferScheduler&) = delete;

  /// \brief Queue one task's graph (non-blocking).
  void Submit(const stratakv::control::TransferGraph& graph,
              common::TaskHandle owner, CompletionSink& sink);

  /// \brief Force a batch flush (event-loop tick hook).
  void Flush();

  /// \brief Drop a task's queued (unflushed) ops.
  /// \return True when anything was dropped (those ops never reach the
  /// transport, so their completions never arrive).
  bool Cancel(common::TaskHandle owner);

  IsolationMode mode() const { return mode_; }

  /// \brief Cancel queued work, drain the transport, idempotent.
  void Shutdown(std::chrono::milliseconds grace);

  /// \brief Ops queued but not yet flushed (introspection/tests).
  std::size_t pending_ops() const;

 private:
  void FlushLocked(CompletionSink* sink);

  const std::unique_ptr<ITransport> transport_;
  const IsolationMode mode_;
  const Config config_;

  mutable std::mutex mu_;
  struct Queued {
    common::TaskHandle owner;
    std::vector<TransferOp> ops;
    CompletionSink* sink;
  };
  std::deque<Queued> queue_;
  std::size_t pending_ops_ = 0;
  std::unordered_map<std::uint64_t, std::size_t> owners_waiting_;
  bool shutting_down_ = false;
};

/// \brief Bridges the control plane's execution seam onto the
/// scheduler (Phase E wiring: TaskEngine drives a real data plane).
///
/// Execute() queues the graph (ops may batch with other tasks');
/// completions arrive from transport threads via the engine's
/// CompletionSink. Cancel() drops still-queued ops; already-started
/// ones complete and the engine drops their late completions
/// (documented control-plane tolerance).
class TransportExecutor final : public stratakv::control::ITaskExecutor {
 public:
  explicit TransportExecutor(std::shared_ptr<TransferScheduler> scheduler)
      : scheduler_(std::move(scheduler)) {}

  void Execute(const stratakv::control::TransferGraph& graph,
               common::TaskHandle owner, CompletionSink& sink) override {
    scheduler_->Submit(graph, owner, sink);
  }

  bool Cancel(common::TaskHandle owner) override {
    return scheduler_->Cancel(owner);
  }

  /// \brief Launch queued work (engine poll cadence — see
  /// ITaskExecutor::Flush; without this, submitted tasks sit in the
  /// scheduler's batch queue until the next poll).
  void Flush() override { scheduler_->Flush(); }

 private:
  std::shared_ptr<TransferScheduler> scheduler_;
};

}  // namespace stratakv::data
