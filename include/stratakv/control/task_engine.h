#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <vector>

#include "stratakv/common/handles.h"
#include "stratakv/control/executor.h"
#include "stratakv/control/task.h"
#include "stratakv/metadata/metadata_plane.h"

namespace stratakv::control {

/// \brief Completion dispatch policy (ADR 0004): callbacks fire only on
/// the documented dispatcher context — the thread calling
/// DrainCompletions, or a single dedicated thread if such a dispatcher
/// is installed — never on arbitrary executor workers.
class CompletionDispatcher {
 public:
  using Callback = std::function<void()>;
  virtual ~CompletionDispatcher() = default;

  /// \brief May post to an event loop; must serialize callbacks.
  virtual void Dispatch(Callback cb) = 0;
};

/// \brief TaskEngine configuration.
struct TaskEngineConfig {
  /// \brief Backpressure bound: maximum tasks in Staged+Launched.
  /// Submissions beyond it throw NotEnoughSpaceError (reject, never
  /// queue unboundedly).
  std::uint32_t max_in_flight = 1024;

  /// \brief Max ops in one task graph (guards op-id packing).
  std::uint32_t max_ops_per_task = 1u << 20;
};

/// \brief Central task engine: staging, launch, completion,
/// cancellation (design.md Phase D).
///
/// Wiring: SubmitGet/SubmitPut translate metadata-plane results into
/// TransferGraphs and hand them to the ITaskExecutor; op completions
/// arrive on OnOpCompleted (any executor thread); terminal transitions
/// fire the completion fd and the DrainCompletions callback context.
/// Metadata side effects on terminal: get -> Release(match) (pins
/// dropped); put success -> CommitPut(entry) (bundle becomes servable);
/// put failure/cancel -> entry stays staged (caller aborts via
/// MetadataPlane::AbortPut).
///
/// Thread safety: all public methods safe from any engine thread.
/// The Python facade releases the GIL around every blocking entry
/// point (ADR 0004).
class TaskEngine final : public CompletionSink {
 public:
  /// \param meta Metadata plane (must outlive the engine).
  /// \param executor Execution seam (shared: tests/py face install
  /// InlineExecutor; Phase E installs the transfer scheduler).
  /// \param config Bounds.
  TaskEngine(metadata::MetadataPlane* meta,
             std::shared_ptr<ITaskExecutor> executor,
             TaskEngineConfig config = {});

  ~TaskEngine() override;

  TaskEngine(const TaskEngine&) = delete;
  TaskEngine& operator=(const TaskEngine&) = delete;

  /// \brief Stage + launch a load of matched blocks into GPU
  /// destinations.
  ///
  /// \param match Pinned match from MetadataPlane::MatchForGet; the
  /// engine releases its pins when the task reaches any terminal state.
  /// \param gpu_slots Destination ids, position-major / group-minor —
  /// exactly parallel to match.block_refs.
  /// \param gpu_state_slots Destination GPU state-object ids, parallel
  /// to match.state_refs; REQUIRED when the match carries sidecar
  /// states (the engine never invents GPU ids) and empty otherwise.
  /// \throws NotEnoughSpaceError on backpressure; std::invalid_argument
  /// on slot-count mismatch; LogicError after Shutdown.
  common::TaskHandle SubmitGet(
      metadata::MatchResult match,
      std::span<const common::BlockId> gpu_slots,
      std::span<const std::uint32_t> gpu_state_slots = {});

  /// \brief Stage + launch an offload of a finished request's
  /// KV/states. Commits the staged entry on success.
  ///
  /// \param plan From MetadataPlane::StagePut.
  /// \param gpu_blocks Source ids, position-major / group-minor —
  /// exactly parallel to plan.blocks_to_store.
  /// \param gpu_state_slots Source GPU state-object ids, parallel to
  /// plan.states_to_snapshot; REQUIRED when the plan snapshots sidecar
  /// states and empty otherwise.
  /// \throws NotEnoughSpaceError on backpressure; LogicError after
  /// Shutdown. No-op task when the plan staged nothing (matched 0 new
  /// blocks) — succeeds immediately without executor involvement.
  common::TaskHandle SubmitPut(
      metadata::MetadataPlane::PutPlan plan,
      std::span<const common::BlockId> gpu_blocks,
      std::span<const std::uint32_t> gpu_state_slots = {});

  /// \brief Cancel a staged/launched task; no-op after terminal state.
  /// Terminal completions arriving later are dropped.
  bool Cancel(common::TaskHandle handle);

  /// \brief Non-blocking poll of task states (replaces per-step
  /// try_wait polling loops — one native call per step, ADR 0004).
  WaitResult TryWait(std::span<const common::TaskHandle> handles);

  /// \brief Blocking wait with timeout. Returns when every handle is
  /// terminal or the timeout elapses (partial results then).
  WaitResult Wait(std::span<const common::TaskHandle> handles,
                  std::chrono::milliseconds timeout);

  /// \brief Pollable OS descriptor signalled (one byte) on every task
  /// terminal transition; integrates with event loops (vLLM EngineCore
  /// poll) without busy polling. Pipe-based for portability; Linux
  /// eventfd is a drop-in refinement.
  int CompletionFd() const;

  /// \brief Drain once after CompletionFd readable: clears the pipe and
  /// delivers completion events on THIS thread (or the installed
  /// dispatcher's context) — the documented single callback context.
  void DrainCompletions();

  /// \brief Install the completion dispatcher (ADR 0004); without one,
  /// callbacks run inline on the DrainCompletions caller.
  void SetDispatcher(std::unique_ptr<CompletionDispatcher> dispatcher);

  /// \brief Completion notification (handle, terminal state). Called
  /// from DrainCompletions' context only.
  using CompletionCallback =
      std::function<void(common::TaskHandle, common::TaskState)>;

  /// \brief Install the completion notification sink (optional).
  void SetCompletionCallback(CompletionCallback callback);

  /// \brief Shutdown: reject new submissions, cancel outstanding
  /// tasks, wait up to `grace` for terminal states. Idempotent.
  void Shutdown(std::chrono::milliseconds grace);

  /// \brief Tasks currently in Staged+Launched.
  std::uint32_t num_in_flight() const;

  /// \brief CompletionSink delivery (executor threads).
  void OnOpCompleted(std::uint64_t op_id, bool ok) override;

 private:
  /// \brief Shared staging path (validation happens in the public
  /// wrappers): assign op ids, launch via the executor, handle the
  /// zero-op fast path.
  common::TaskHandle Submit(TransferGraph graph, bool is_put,
                            metadata::MatchResult match,
                            metadata::MetadataPlane::PutPlan plan);

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace stratakv::control
