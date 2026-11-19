#include "stratakv/data/transport.h"

#include <utility>

#include "stratakv/common/exceptions.h"

namespace stratakv::data {

TransferScheduler::TransferScheduler(std::unique_ptr<ITransport> transport,
                                     IsolationMode mode, Config config)
    : transport_(std::move(transport)), mode_(mode), config_(config) {
  if (transport_ == nullptr) {
    throw std::invalid_argument("TransferScheduler: transport required");
  }
}

TransferScheduler::~TransferScheduler() {
  Shutdown(std::chrono::milliseconds(0));
}

void TransferScheduler::Submit(
    const stratakv::control::TransferGraph& graph, common::TaskHandle owner,
    CompletionSink& sink) {
  std::lock_guard<std::mutex> lock(mu_);
  if (shutting_down_) {
    throw LogicError("TransferScheduler: shut down");
  }
  Queued queued;
  queued.queued_at = std::chrono::steady_clock::now();
  queued.owner = owner;
  queued.sink = &sink;
  queued.ops.reserve(graph.ops().size());
  for (const TransferOp& op : graph.ops()) {
    queued.ops.push_back(op);
  }
  pending_ops_ += queued.ops.size();
  queue_.push_back(std::move(queued));
  ++owners_waiting_[owner.id()];

  // Drain-and-batch (FlexKV worker.py harvest): flush when big enough
  // or when enough distinct owners are waiting — one Enqueue round
  // trip per batch.
  if (pending_ops_ >= config_.max_batch_ops ||
      owners_waiting_.size() >= config_.max_pending_tasks) {
    FlushLocked(&sink);
  }
}

void TransferScheduler::Flush() {
  std::lock_guard<std::mutex> lock(mu_);
  FlushLocked(nullptr);
}

void TransferScheduler::FlushLocked(CompletionSink* /*caller_sink*/) {
  if (queue_.empty()) {
    return;
  }
  // Drain-and-batch (FlexKV worker.py harvest): coalesce ALL queued
  // ops into one transport submission per completion sink — in
  // practice one sink (the TaskEngine) for every owner, so a flush is
  // one Enqueue round trip for the whole batch. Distinct sinks split
  // the batch (completion routing is per-sink).
  // Enqueue is non-blocking by contract and transports never call back
  // into the scheduler, so calling it under mu_ cannot deadlock
  // (scheduler.mu_ -> transport.* is the only ordering).
  std::unordered_map<CompletionSink*, std::vector<TransferOp>> by_sink;
  while (!queue_.empty()) {
    Queued queued = std::move(queue_.front());
    queue_.pop_front();
    pending_ops_ -= queued.ops.size();
    owners_waiting_.erase(queued.owner.id());
    queue_latency_.Observe(std::chrono::steady_clock::now() - queued.queued_at);
    auto& ops = by_sink[queued.sink];
    for (TransferOp& op : queued.ops) {
      ops.push_back(std::move(op));
    }
  }
  for (auto& [sink, ops] : by_sink) {
    common::ScopedLatency timing(enqueue_latency_);
    transport_->Enqueue(std::move(ops), *sink);
  }
}

bool TransferScheduler::Cancel(common::TaskHandle owner) {
  std::lock_guard<std::mutex> lock(mu_);
  bool dropped = false;
  std::deque<Queued> kept;
  while (!queue_.empty()) {
    Queued queued = std::move(queue_.front());
    queue_.pop_front();
    if (queued.owner == owner) {
      pending_ops_ -= queued.ops.size();
      dropped = true;
      continue;
    }
    kept.push_back(std::move(queued));
  }
  queue_ = std::move(kept);
  owners_waiting_.erase(owner.id());
  // Unflushed ops are gone (no completions will arrive for them — the
  // control plane tolerates late/missing completions only after
  // terminal states; the engine marks cancelled tasks itself, so this
  // is bookkeeping, not completion fabrication).
  return dropped;
}

void TransferScheduler::Shutdown(std::chrono::milliseconds grace) {
  std::vector<Queued> dropped;
  {
    std::lock_guard<std::mutex> lock(mu_);
    if (shutting_down_) {
      return;
    }
    shutting_down_ = true;
    dropped.assign(std::make_move_iterator(queue_.begin()),
                   std::make_move_iterator(queue_.end()));
    queue_.clear();
    pending_ops_ = 0;
    owners_waiting_.clear();
  }
  (void)dropped;
  transport_->DrainPending(grace);
}

std::size_t TransferScheduler::pending_ops() const {
  std::lock_guard<std::mutex> lock(mu_);
  return pending_ops_;
}

}  // namespace stratakv::data
