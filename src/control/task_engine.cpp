#include "stratakv/control/task_engine.h"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <condition_variable>
#include <deque>
#include <map>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "stratakv/common/exceptions.h"

namespace stratakv::control {

namespace {

/// \brief Metadata side effects of one terminal transition, applied by
/// the call site AFTER releasing the engine lock (the plane takes its
/// own locks; never nested under ours).
struct TerminalEffects {
  metadata::MatchResult match;              // Release pins (get).
  common::CacheEntry* commit_entry = nullptr;  // CommitPut (put success).
};

}  // namespace

/// \brief One tracked task. Records live until engine destruction
/// (simple lifetime story; reclamation of terminal records is a Phase E
/// growth item).
struct TaskRecord {
  std::uint64_t id = 0;
  common::TaskState state = common::TaskState::kStaged;
  std::string error;
  bool is_put = false;
  std::uint32_t remaining_ops = 0;
  metadata::MatchResult match;            // Get: pins released on
                                          // terminal.
  metadata::MetadataPlane::PutPlan plan;  // Put: committed on success.
};

struct TaskEngine::Impl {
  metadata::MetadataPlane* meta = nullptr;
  std::shared_ptr<ITaskExecutor> executor;
  TaskEngineConfig config;

  mutable std::mutex mu;
  std::condition_variable cv;
  std::map<std::uint64_t, TaskRecord> tasks;  // Keyed by task id.
  std::uint32_t in_flight = 0;
  std::uint64_t next_task_id = 1;
  bool shutting_down = false;

  std::unique_ptr<CompletionDispatcher> dispatcher;
  CompletionCallback completion_callback;

  // Completion pipe (portable eventfd stand-in; ADR 0004). Non-blocking
  // both ends: executor threads never block on write; drain loops never
  // block on read.
  int notify_fd[2] = {-1, -1};
  std::deque<common::TaskHandle> pending_events;

  // Op ids pack (task_id << 20) | op_index; bounded by
  // config.max_ops_per_task < 2^20.
  static std::uint64_t PackOpId(std::uint64_t task_id,
                                std::uint32_t op_index) {
    return (task_id << 20) | op_index;
  }
  static std::uint64_t TaskOfOp(std::uint64_t op_id) {
    return op_id >> 20;
  }

  void NotifyFdLocked() {
    if (notify_fd[1] >= 0) {
      const char byte = 1;
      const ssize_t rc = ::write(notify_fd[1], &byte, 1);
      (void)rc;  // EAGAIN when full is fine: a byte already pends.
    }
  }

  /// \brief Drive one task to a terminal state exactly once. Requires
  /// mu held; emits the fd byte + condvar signal and RETURNS the
  /// metadata side effects for the caller to apply unlocked.
  TerminalEffects FinishLocked(std::uint64_t id, common::TaskState state,
                               std::string error) {
    TerminalEffects effects;
    auto it = tasks.find(id);
    if (it == tasks.end()) {
      return effects;
    }
    TaskRecord& task = it->second;
    if (stratakv::common::IsTerminal(task.state)) {
      return effects;  // Already terminal; drop late completions.
    }
    task.state = state;
    task.error = std::move(error);
    --in_flight;
    pending_events.push_back(common::TaskHandle(task.id));
    NotifyFdLocked();
    cv.notify_all();

    effects.match = std::move(task.match);
    if (task.is_put && state == common::TaskState::kSucceeded) {
      effects.commit_entry = task.plan.entry;
    }
    return effects;
  }

  /// \brief Metadata side effects of one terminal transition.
  ///
  /// Applied WHILE mu IS HELD (the metadata plane is thread-safe and
  /// never re-enters the engine): the terminal state and its side
  /// effects become visible atomically. Wait/TryWait can therefore
  /// never observe a terminal task whose CommitPut/Release has not run
  /// — the previous unlocked ordering let callers evict between the
  /// state flip and CommitPut, shredding entries mid-transition (found
  /// by the Phase E transport benchmark's put-evict loop).
  void ApplyEffects(const TerminalEffects& effects) {
    if (effects.match.pin != nullptr) {
      meta->Release(effects.match);  // Get terminal: drop CPU pins.
    }
    if (effects.commit_entry != nullptr) {
      try {
        meta->CommitPut(effects.commit_entry);
      } catch (const LogicError&) {
        // Entry was evicted while the task ran; the transfer itself
        // still succeeded — nothing to commit onto.
      }
    }
  }
};

TaskEngine::TaskEngine(metadata::MetadataPlane* meta,
                       std::shared_ptr<ITaskExecutor> executor,
                       TaskEngineConfig config)
    : impl_(std::make_unique<Impl>()) {
  if (meta == nullptr || executor == nullptr) {
    throw std::invalid_argument(
        "TaskEngine: metadata plane and executor must be non-null");
  }
  if (config.max_ops_per_task > (1u << 20)) {
    throw std::invalid_argument(
        "TaskEngine: max_ops_per_task must stay within 2^20 (op-id "
        "packing)");
  }
  impl_->meta = meta;
  impl_->executor = std::move(executor);
  impl_->config = config;
  if (::pipe(impl_->notify_fd) != 0) {
    throw std::runtime_error("TaskEngine: pipe() failed");
  }
  for (int fd : impl_->notify_fd) {
    const int flags = ::fcntl(fd, F_GETFL, 0);
    ::fcntl(fd, F_SETFL, flags | O_NONBLOCK);
  }
}

TaskEngine::~TaskEngine() {
  Shutdown(std::chrono::milliseconds(0));
  for (int& fd : impl_->notify_fd) {
    if (fd >= 0) {
      ::close(fd);
      fd = -1;
    }
  }
}

common::TaskHandle TaskEngine::SubmitGet(
    metadata::MatchResult match,
    std::span<const common::BlockId> gpu_slots,
    std::span<const std::uint32_t> gpu_state_slots) {
  if (match.matched_blocks > 0 && match.pin == nullptr) {
    throw std::invalid_argument(
        "TaskEngine::SubmitGet: match must carry a pin (lock_matched)");
  }
  if (gpu_slots.size() != match.block_refs.size()) {
    throw std::invalid_argument(
        "TaskEngine::SubmitGet: gpu_slots must parallel match.block_refs");
  }

  // One H2D op per token-addressable group over the matched blocks,
  // plus one state-restore op per boundary state (hybrid layouts).
  if (gpu_state_slots.size() != match.state_refs.size()) {
    throw std::invalid_argument(
        "TaskEngine::SubmitGet: gpu_state_slots must parallel "
        "match.state_refs");
  }
  std::map<std::uint32_t, std::vector<common::BlockId>> src_by_group;
  std::map<std::uint32_t, std::vector<common::BlockId>> dst_by_group;
  for (std::size_t i = 0; i < match.block_refs.size(); ++i) {
    const common::BlockRef& ref = match.block_refs[i];
    src_by_group[ref.group].push_back(ref.block);
    dst_by_group[ref.group].push_back(gpu_slots[i]);
  }
  TransferGraph graph;
  for (auto& [group, srcs] : src_by_group) {
    TransferOp op;
    op.kind = TransferOp::Kind::kCpuToGpu;
    op.src_tier = 0;  // CPU tier (Phase C); Phase E names real tiers.
    op.dst_tier = 1;  // GPU tier.
    op.src_group = group;
    op.dst_group = group;
    op.src_blocks = std::move(srcs);
    op.dst_blocks = std::move(dst_by_group[group]);
    graph.AddOp(std::move(op));
  }
  for (std::size_t i = 0; i < match.state_refs.size(); ++i) {
    const common::StateRef& ref = match.state_refs[i];
    TransferOp op;
    op.kind = TransferOp::Kind::kStateRestore;
    op.src_tier = 0;
    op.dst_tier = 1;
    op.src_group = ref.group;
    op.dst_group = ref.group;
    op.src_object = ref.object;
    op.dst_object = gpu_state_slots[i];  // Engine-declared GPU slot.
    graph.AddOp(std::move(op));
  }

  return Submit(std::move(graph), /*is_put=*/false, std::move(match),
                metadata::MetadataPlane::PutPlan{});
}

common::TaskHandle TaskEngine::SubmitPut(
    metadata::MetadataPlane::PutPlan plan,
    std::span<const common::BlockId> gpu_blocks,
    std::span<const std::uint32_t> gpu_state_slots) {
  if (plan.new_blocks > 0 && plan.entry == nullptr) {
    throw std::invalid_argument(
        "TaskEngine::SubmitPut: plan must carry a staged entry");
  }
  if (gpu_blocks.size() != plan.blocks_to_store.size()) {
    throw std::invalid_argument(
        "TaskEngine::SubmitPut: gpu_blocks must parallel "
        "plan.blocks_to_store");
  }

  // One D2H op per token-addressable group over the new blocks, plus
  // one state-snapshot op per LinearState group at the boundary.
  if (gpu_state_slots.size() != plan.states_to_snapshot.size()) {
    throw std::invalid_argument(
        "TaskEngine::SubmitPut: gpu_state_slots must parallel "
        "plan.states_to_snapshot");
  }
  std::map<std::uint32_t, std::vector<common::BlockId>> dst_by_group;
  std::map<std::uint32_t, std::vector<common::BlockId>> src_by_group;
  for (std::size_t i = 0; i < plan.blocks_to_store.size(); ++i) {
    const common::BlockRef& ref = plan.blocks_to_store[i];
    src_by_group[ref.group].push_back(gpu_blocks[i]);
    dst_by_group[ref.group].push_back(ref.block);
  }
  TransferGraph graph;
  for (auto& [group, dsts] : dst_by_group) {
    TransferOp op;
    op.kind = TransferOp::Kind::kGpuToCpu;
    op.src_tier = 1;
    op.dst_tier = 0;
    op.src_group = group;
    op.dst_group = group;
    op.src_blocks = std::move(src_by_group[group]);
    op.dst_blocks = std::move(dsts);
    graph.AddOp(std::move(op));
  }
  for (std::size_t i = 0; i < plan.states_to_snapshot.size(); ++i) {
    const common::StateRef& ref = plan.states_to_snapshot[i];
    TransferOp op;
    op.kind = TransferOp::Kind::kStateSnapshot;
    op.src_tier = 1;
    op.dst_tier = 0;
    op.src_group = ref.group;
    op.dst_group = ref.group;
    op.src_object = gpu_state_slots[i];  // Engine-declared GPU slot.
    op.dst_object = ref.object;
    graph.AddOp(std::move(op));
  }

  return Submit(std::move(graph), /*is_put=*/true, metadata::MatchResult{},
                std::move(plan));
}

common::TaskHandle TaskEngine::Submit(
    TransferGraph graph, bool is_put, metadata::MatchResult match,
    metadata::MetadataPlane::PutPlan plan) {
  TerminalEffects effects;
  bool has_effects = false;
  common::TaskHandle handle;
  {
    std::lock_guard<std::mutex> lock(impl_->mu);
    if (impl_->shutting_down) {
      throw LogicError("TaskEngine: shut down; submissions rejected");
    }
    if (impl_->in_flight >= impl_->config.max_in_flight) {
      throw NotEnoughSpaceError(
          "TaskEngine: saturated (" +
          std::to_string(impl_->config.max_in_flight) + " in flight)");
    }
    if (graph.ops().size() > impl_->config.max_ops_per_task) {
      throw std::invalid_argument("TaskEngine: graph too large");
    }

    const std::uint64_t id = impl_->next_task_id++;
    TaskRecord& task = impl_->tasks[id];
    task.id = id;
    task.is_put = is_put;
    task.match = std::move(match);
    task.plan = std::move(plan);
    task.remaining_ops = static_cast<std::uint32_t>(graph.ops().size());
    handle = common::TaskHandle(id);

    // Assign completion ids to the ops (copy-out; TransferGraph exposes
    // a const view by design).
    TransferGraph assigned;
    std::uint32_t index = 0;
    for (const TransferOp& op : graph.ops()) {
      TransferOp with_id = op;
      with_id.op_id = Impl::PackOpId(id, index++);
      assigned.AddOp(std::move(with_id));
    }
    for (const auto& [op_index, depends_on] : graph.dependencies()) {
      assigned.AddDependency(op_index, depends_on);
    }
    graph = std::move(assigned);

    ++impl_->in_flight;
    if (task.remaining_ops == 0) {
      // Nothing to execute (e.g., fully-matched get): succeed now,
      // without the executor.
      effects = impl_->FinishLocked(id, common::TaskState::kSucceeded,
                                     std::string());
      impl_->ApplyEffects(effects);
      has_effects = true;
    } else {
      task.state = common::TaskState::kLaunched;
    }
  }
  if (has_effects) {
    return handle;
  }
  impl_->executor->Execute(graph, handle, *this);
  return handle;
}

bool TaskEngine::Cancel(common::TaskHandle handle) {
  TerminalEffects effects;
  bool launched = false;
  {
    std::lock_guard<std::mutex> lock(impl_->mu);
    auto it = impl_->tasks.find(handle.id());
    if (it == impl_->tasks.end() ||
        stratakv::common::IsTerminal(it->second.state)) {
      return false;
    }
    launched = it->second.state == common::TaskState::kLaunched;
    effects = impl_->FinishLocked(handle.id(),
                                  common::TaskState::kCancelled,
                                  "cancelled by caller");
    impl_->ApplyEffects(effects);
  }
  if (launched) {
    impl_->executor->Cancel(handle);  // Best-effort; late ops dropped.
  }
  return true;
}

WaitResult TaskEngine::TryWait(std::span<const common::TaskHandle> handles) {
  std::lock_guard<std::mutex> lock(impl_->mu);
  WaitResult result;
  result.items.reserve(handles.size());
  for (const common::TaskHandle& handle : handles) {
    auto it = impl_->tasks.find(handle.id());
    if (it == impl_->tasks.end()) {
      result.items.push_back(WaitResult::Item{
          handle, common::TaskState::kStaged, "unknown handle"});
      continue;
    }
    result.items.push_back(
        WaitResult::Item{handle, it->second.state, it->second.error});
  }
  return result;
}

WaitResult TaskEngine::Wait(std::span<const common::TaskHandle> handles,
                            std::chrono::milliseconds timeout) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  {
    std::unique_lock<std::mutex> lock(impl_->mu);
    const bool all_known =
        std::all_of(handles.begin(), handles.end(), [&](auto h) {
          return impl_->tasks.count(h.id()) > 0;
        });
    if (all_known) {
      impl_->cv.wait_until(lock, deadline, [&] {
        return std::all_of(handles.begin(), handles.end(), [&](auto h) {
          auto it = impl_->tasks.find(h.id());
          return it != impl_->tasks.end() &&
                 stratakv::common::IsTerminal(it->second.state);
        });
      });
    }
  }
  return TryWait(handles);
}

int TaskEngine::CompletionFd() const { return impl_->notify_fd[0]; }

void TaskEngine::DrainCompletions() {
  // Clear the pipe first (non-blocking read end).
  char buffer[256];
  while (::read(impl_->notify_fd[0], buffer, sizeof(buffer)) > 0) {
  }

  std::deque<common::TaskHandle> events;
  CompletionCallback callback;
  CompletionDispatcher* dispatcher = nullptr;
  {
    std::lock_guard<std::mutex> lock(impl_->mu);
    events.swap(impl_->pending_events);
    callback = impl_->completion_callback;
    dispatcher = impl_->dispatcher.get();
  }
  if (callback == nullptr) {
    return;  // Nothing installed; events stay drained.
  }

  // Delivery context (ADR 0004): THIS thread, or the dispatcher's
  // single documented context when one is installed.
  for (const common::TaskHandle handle : events) {
    const WaitResult snapshot = TryWait(
        std::span<const common::TaskHandle>(&handle, 1));
    if (snapshot.items.empty()) {
      continue;
    }
    const common::TaskState state = snapshot.items[0].state;
    if (dispatcher != nullptr) {
      dispatcher->Dispatch(
          [callback, handle, state] { callback(handle, state); });
    } else {
      callback(handle, state);
    }
  }
}

void TaskEngine::SetDispatcher(
    std::unique_ptr<CompletionDispatcher> dispatcher) {
  std::lock_guard<std::mutex> lock(impl_->mu);
  impl_->dispatcher = std::move(dispatcher);
}

void TaskEngine::SetCompletionCallback(CompletionCallback callback) {
  std::lock_guard<std::mutex> lock(impl_->mu);
  impl_->completion_callback = std::move(callback);
}

void TaskEngine::Shutdown(std::chrono::milliseconds grace) {
  std::vector<TerminalEffects> effects;
  std::vector<common::TaskHandle> launched;
  {
    std::lock_guard<std::mutex> lock(impl_->mu);
    if (impl_->shutting_down) {
      return;
    }
    impl_->shutting_down = true;
    for (auto& [id, task] : impl_->tasks) {
      if (!stratakv::common::IsTerminal(task.state)) {
        if (task.state == common::TaskState::kLaunched) {
          launched.emplace_back(id);
        }
        effects.push_back(
            impl_->FinishLocked(id, common::TaskState::kCancelled,
                                "engine shutdown"));
      }
    }
    for (const TerminalEffects& effect : effects) {
      impl_->ApplyEffects(effect);
    }
  }
  for (const common::TaskHandle& handle : launched) {
    impl_->executor->Cancel(handle);
  }
  if (!launched.empty()) {
    // Grace window for executor work to observe the cancels.
    std::unique_lock<std::mutex> lock(impl_->mu);
    impl_->cv.wait_until(lock, std::chrono::steady_clock::now() + grace,
                         [&] { return impl_->in_flight == 0; });
  }
}

std::uint32_t TaskEngine::num_in_flight() const {
  std::lock_guard<std::mutex> lock(impl_->mu);
  return impl_->in_flight;
}

void TaskEngine::OnOpCompleted(std::uint64_t op_id, bool ok) {
  TerminalEffects effects;
  {
    std::lock_guard<std::mutex> lock(impl_->mu);
    auto it = impl_->tasks.find(Impl::TaskOfOp(op_id));
    if (it == impl_->tasks.end()) {
      return;  // Unknown op: tolerated (executor contract violation
               // must not crash engine threads).
    }
    TaskRecord& task = it->second;
    if (stratakv::common::IsTerminal(task.state)) {
      return;  // Cancelled; drop the late completion.
    }
    if (task.remaining_ops > 0) {
      --task.remaining_ops;
    }
    if (!ok && task.error.empty()) {
      task.error = "op " + std::to_string(op_id) + " failed";
    }
    if (task.remaining_ops == 0) {
      effects = impl_->FinishLocked(
          task.id,
          task.error.empty() ? common::TaskState::kSucceeded
                             : common::TaskState::kFailed,
          task.error);
      impl_->ApplyEffects(effects);
    }
  }
}

}  // namespace stratakv::control
