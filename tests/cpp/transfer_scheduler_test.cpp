// Phase E (E3, offline portion): TransferScheduler + TransportExecutor
// tests — drain-and-batch coalescing (the FlexKV worker.py harvest),
// cancel of queued work, and the full D+E wiring: TaskEngine graphs
// flow through the scheduler into a (fake) transport whose completions
// drive engine terminal states and metadata side effects.

#ifdef STRATAKV_GTEST_SHIM
#include "gtest_shim.h"
#else
#include <gtest/gtest.h>
#endif

#include <chrono>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <span>
#include <vector>

#include "stratakv/common/exceptions.h"
#include "stratakv/common/handles.h"
#include "stratakv/common/model_layout.h"
#include "stratakv/control/task_engine.h"
#include "stratakv/data/transport.h"
#include "stratakv/metadata/metadata_plane.h"

namespace {

using stratakv::common::FullAttnSpec;
using stratakv::common::LayerGroup;
using stratakv::common::ModelLayout;
using stratakv::common::TaskHandle;
using stratakv::common::TaskState;
using stratakv::control::CompletionSink;
using stratakv::control::TaskEngine;
using stratakv::control::TransferGraph;
using stratakv::control::TransferOp;
using stratakv::data::IsolationMode;
using stratakv::data::ITransport;
using stratakv::data::TransferScheduler;
using stratakv::data::TransportExecutor;
using stratakv::metadata::MatchResult;
using stratakv::metadata::MetadataPlane;
using stratakv::metadata::MetadataPlaneConfig;
using stratakv::metadata::RadixIndex;

ModelLayout Homogeneous() {
  LayerGroup full;
  full.name = "full_attn";
  full.spec = FullAttnSpec{2, 64};
  full.layer_indices = {0, 1, 2, 3};
  full.tokens_per_block = 2;
  return ModelLayout::FromGroups({full});
}

MetadataPlaneConfig TestPools() {
  MetadataPlaneConfig config;
  config.blocks_per_group = 128;
  config.state_objects_per_group = 8;
  config.state_object_bytes = 4096;
  return config;
}

/// \brief Sink that ignores completions (batching-only tests).
struct NoOpSink final : public CompletionSink {
  void OnOpCompleted(std::uint64_t, bool) override {}
};

std::vector<stratakv::common::BlockId> Slots(
    std::initializer_list<stratakv::common::BlockId> in) {
  return std::vector<stratakv::common::BlockId>(in);
}

/// \brief Fake transport: records batches; completions fired manually.
class FakeTransport final : public ITransport {
 public:
  void Enqueue(std::vector<TransferOp> batch,
               CompletionSink& sink) override {
    std::lock_guard<std::mutex> lock(mu_);
    ++enqueue_count_;
    for (const TransferOp& op : batch) {
      ops_.push_back({op.op_id, &sink});
    }
  }

  void DrainPending(std::chrono::milliseconds) override {}

  std::size_t CancelPending(TaskHandle) override { return 0; }

  /// \brief Deliver `n` queued completions (oldest first).
  std::size_t FireCompletions(std::size_t n, bool ok) {
    std::size_t fired = 0;
    while (fired < n) {
      std::uint64_t op_id = 0;
      CompletionSink* sink = nullptr;
      {
        std::lock_guard<std::mutex> lock(mu_);
        if (ops_.empty()) break;
        op_id = ops_.front().op_id;
        sink = ops_.front().sink;
        ops_.pop_front();
      }
      sink->OnOpCompleted(op_id, ok);
      ++fired;
    }
    return fired;
  }

  std::size_t enqueued_ops() const {
    std::lock_guard<std::mutex> lock(mu_);
    return ops_.size();
  }

  std::size_t enqueue_count() const {
    std::lock_guard<std::mutex> lock(mu_);
    return enqueue_count_;
  }

 private:
  mutable std::mutex mu_;
  struct Pending {
    std::uint64_t op_id;
    CompletionSink* sink;
  };
  std::deque<Pending> ops_;
  std::size_t enqueue_count_ = 0;
};

TaskState State(TaskEngine& engine, const TaskHandle& handle) {
  return engine
      .TryWait(std::span<const TaskHandle>(&handle, 1))
      .items.at(0)
      .state;
}

TEST(TransferSchedulerTest, BatchingCoalescesOwners) {
  auto transport = std::make_unique<FakeTransport>();
  FakeTransport* raw = transport.get();
  TransferScheduler::Config config;
  config.max_batch_ops = 512;
  config.max_pending_tasks = 4;  // Flush at 4 distinct owners.
  TransferScheduler scheduler(std::move(transport),
                              IsolationMode::kInProcess, config);

  NoOpSink sink;
  for (int task = 0; task < 3; ++task) {
    TransferGraph graph;
    TransferOp op;
    op.kind = TransferOp::Kind::kCpuToGpu;
    op.src_blocks = {1, 2, 3};
    graph.AddOp(op);
    graph.AddOp(op);
    scheduler.Submit(graph, TaskHandle(static_cast<std::uint64_t>(task) + 1),
                     sink);
  }
  // Three distinct owners but threshold is 4: still queued, one op set.
  EXPECT_EQ(scheduler.pending_ops(), static_cast<std::size_t>(6));
  EXPECT_EQ(raw->enqueue_count(), static_cast<std::size_t>(0));

  scheduler.Flush();
  // All owners' ops coalesced into ONE transport submission.
  EXPECT_EQ(raw->enqueue_count(), static_cast<std::size_t>(1));
  EXPECT_EQ(raw->enqueued_ops(), static_cast<std::size_t>(6));
  EXPECT_EQ(scheduler.pending_ops(), static_cast<std::size_t>(0));

  // Threshold flush: 4 distinct owners trigger without explicit Flush.
  for (int task = 10; task < 14; ++task) {
    TransferGraph graph;
    graph.AddOp(TransferOp{});
    scheduler.Submit(graph, TaskHandle(static_cast<std::uint64_t>(task)),
                     sink);
  }
  EXPECT_EQ(raw->enqueue_count(), static_cast<std::size_t>(2));
  EXPECT_EQ(scheduler.QueueLatency().count, 7u);
  EXPECT_EQ(scheduler.EnqueueLatency().count, 2u);
  scheduler.Flush();
  EXPECT_EQ(scheduler.EnqueueLatency().count, 2u);
}

TEST(TransferSchedulerTest, CancelDropsQueuedOps) {
  auto transport = std::make_unique<FakeTransport>();
  FakeTransport* raw = transport.get();
  TransferScheduler::Config config;
  config.max_pending_tasks = 64;  // No auto-flush.
  TransferScheduler scheduler(std::move(transport),
                              IsolationMode::kIsolatedProcess, config);
  EXPECT_EQ(scheduler.mode(), IsolationMode::kIsolatedProcess);

  NoOpSink sink;
  TransferGraph graph;
  graph.AddOp(TransferOp{});
  const TaskHandle owner(7);
  scheduler.Submit(graph, owner, sink);
  EXPECT_EQ(scheduler.pending_ops(), static_cast<std::size_t>(1));

  EXPECT_TRUE(scheduler.Cancel(owner));
  EXPECT_EQ(scheduler.pending_ops(), static_cast<std::size_t>(0));
  EXPECT_FALSE(scheduler.Cancel(owner));  // Nothing left to drop.

  scheduler.Flush();
  EXPECT_EQ(raw->enqueue_count(),
            static_cast<std::size_t>(0));  // Never reached the
                                          // transport.
  scheduler.Shutdown(std::chrono::milliseconds(0));
}

TEST(DataPlaneWiringTest, TaskEngineDrivesTransportEndToEnd) {
  const ModelLayout layout = Homogeneous();
  MetadataPlane plane(layout, std::make_unique<RadixIndex>(layout),
                      TestPools());

  auto transport = std::make_unique<FakeTransport>();
  FakeTransport* raw = transport.get();
  TransferScheduler::Config config;
  config.max_batch_ops = 8;  // Small: batch after a couple of tasks.
  auto scheduler = std::make_shared<TransferScheduler>(
      std::move(transport), IsolationMode::kInProcess, config);
  TaskEngine engine(&plane,
                    std::make_shared<TransportExecutor>(scheduler));

  // Two puts batched into one transport submission.
  const MetadataPlane::PutPlan a =
      plane.StagePut(std::vector<stratakv::common::TokenId>{1, 2, 3, 4});
  const MetadataPlane::PutPlan b =
      plane.StagePut(std::vector<stratakv::common::TokenId>{5, 6, 7, 8});
  const TaskHandle put_a = engine.SubmitPut(a, Slots({0, 1}));
  const TaskHandle put_b = engine.SubmitPut(b, Slots({2, 3}));
  EXPECT_EQ(raw->enqueue_count(), static_cast<std::size_t>(0));
  scheduler->Flush();
  EXPECT_EQ(raw->enqueue_count(), static_cast<std::size_t>(1));
  EXPECT_EQ(raw->enqueued_ops(), static_cast<std::size_t>(2));

  // Completions from the transport drive the engine to terminal
  // states and commit the metadata.
  (void)raw->FireCompletions(1, true);
  EXPECT_EQ(State(engine, put_a), TaskState::kSucceeded);
  EXPECT_EQ(State(engine, put_b), TaskState::kLaunched);
  (void)raw->FireCompletions(1, true);
  EXPECT_EQ(State(engine, put_b), TaskState::kSucceeded);

  const MatchResult match = plane.MatchForGet(std::vector<stratakv::common::TokenId>{1, 2, 3, 4});
  ASSERT_EQ(match.matched_blocks, 2u);  // Put A committed by the
                                        // engine.

  // Get path: engine → scheduler → transport; pins released on
  // completion.
  const TaskHandle get = engine.SubmitGet(match, Slots({10, 11}));
  scheduler->Flush();
  EXPECT_EQ(raw->enqueue_count(), static_cast<std::size_t>(2));
  (void)raw->FireCompletions(1, true);
  EXPECT_EQ(State(engine, get), TaskState::kSucceeded);
  EXPECT_EQ(engine.num_in_flight(), 0u);
  const stratakv::metadata::Evicted evicted =
      plane.EvictBlocks(4);  // Unpinned after get terminal.
  EXPECT_EQ(evicted.blocks.size(), static_cast<std::size_t>(4));
}

TEST(DataPlaneWiringTest, CancelledTaskOpsNeverReachTransport) {
  const ModelLayout layout = Homogeneous();
  MetadataPlane plane(layout, std::make_unique<RadixIndex>(layout),
                      TestPools());
  auto transport = std::make_unique<FakeTransport>();
  FakeTransport* raw = transport.get();
  TransferScheduler::Config config;
  config.max_pending_tasks = 64;  // Hold everything queued.
  auto scheduler = std::make_shared<TransferScheduler>(
      std::move(transport), IsolationMode::kInProcess, config);
  TaskEngine engine(&plane,
                    std::make_shared<TransportExecutor>(scheduler));

  const MetadataPlane::PutPlan plan = plane.StagePut(std::vector<stratakv::common::TokenId>{1, 2, 3, 4});
  const TaskHandle handle = engine.SubmitPut(plan, Slots({0, 1}));
  EXPECT_TRUE(engine.Cancel(handle));
  scheduler->Flush();
  EXPECT_EQ(raw->enqueue_count(),
            static_cast<std::size_t>(0));  // Dropped before flush.
  EXPECT_EQ(State(engine, handle), TaskState::kCancelled);

  // The reservation is recoverable via abort.
  (void)plane.AbortPut(plan.entry);
  EXPECT_EQ(plane.num_free_blocks(0), TestPools().blocks_per_group);
}

}  // namespace
