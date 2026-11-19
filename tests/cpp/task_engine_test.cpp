// Phase D (D1/D2): task engine tests — lifecycle with metadata side
// effects, backpressure, cancellation + abort, failure propagation,
// wait timeouts, completion fd + drain dispatch context, a 10k-task
// stress with random cancels, and shutdown mid-flight.

#ifdef STRATAKV_GTEST_SHIM
#include "gtest_shim.h"
#else
#include <gtest/gtest.h>
#endif

#include <poll.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "stratakv/common/exceptions.h"
#include "stratakv/common/handles.h"
#include "stratakv/common/model_layout.h"
#include "stratakv/control/executor.h"
#include "stratakv/control/task_engine.h"
#include "stratakv/metadata/index.h"
#include "stratakv/metadata/metadata_plane.h"

namespace {

using stratakv::common::BlockId;
using stratakv::common::FullAttnSpec;
using stratakv::common::LayerGroup;
using stratakv::common::ModelLayout;
using stratakv::common::TaskHandle;
using stratakv::common::TaskState;
using stratakv::common::TokenId;
using stratakv::control::CompletionSink;
using stratakv::control::ITaskExecutor;
using stratakv::control::InlineExecutor;
using stratakv::control::TaskEngine;
using stratakv::control::TaskEngineConfig;
using stratakv::control::TransferGraph;
using stratakv::control::TransferOp;
using stratakv::control::WaitResult;
using stratakv::metadata::Evicted;
using stratakv::metadata::MatchResult;
using stratakv::metadata::MetadataPlane;
using stratakv::metadata::MetadataPlaneConfig;
using stratakv::metadata::RadixIndex;

ModelLayout Homogeneous(std::uint32_t tpb) {
  LayerGroup full;
  full.name = "full_attn";
  full.spec = FullAttnSpec{2, 64};
  full.layer_indices = {0, 1, 2, 3};
  full.tokens_per_block = static_cast<int>(tpb);
  return ModelLayout::FromGroups({full});
}

std::vector<TokenId> Tok(std::initializer_list<TokenId> in) {
  return std::vector<TokenId>(in);
}

std::vector<BlockId> Slots(std::initializer_list<BlockId> in) {
  return std::vector<BlockId>(in);
}

MetadataPlaneConfig TestPools() {
  MetadataPlaneConfig config;
  config.blocks_per_group = 256;
  config.state_objects_per_group = 16;
  config.state_object_bytes = 4096;
  return config;
}

TaskState State(TaskEngine& engine, const TaskHandle& handle) {
  return engine
      .TryWait(std::span<const TaskHandle>(&handle, 1))
      .items.at(0)
      .state;
}

TaskState WaitOne(TaskEngine& engine, const TaskHandle& handle,
                  std::chrono::milliseconds timeout) {
  return engine
      .Wait(std::span<const TaskHandle>(&handle, 1), timeout)
      .items.at(0)
      .state;
}

/// \brief Test fake: queues op completions until Pump() fires them.
/// The engine never calls into the executor while holding its lock, so
/// pumping from the test thread cannot deadlock.
class ManualExecutor final : public ITaskExecutor {
 public:
  void Execute(const TransferGraph& graph, TaskHandle /*owner*/,
               CompletionSink& /*sink*/) override {
    std::lock_guard<std::mutex> lock(mu_);
    for (const TransferOp& op : graph.ops()) {
      pending_.push_back(op.op_id);
    }
  }

  bool Cancel(TaskHandle /*owner*/) override { return true; }

  std::size_t Pump(std::size_t n, bool ok, CompletionSink& sink) {
    std::size_t fired = 0;
    while (fired < n) {
      std::uint64_t op_id = 0;
      {
        std::lock_guard<std::mutex> lock(mu_);
        if (pending_.empty()) break;
        op_id = pending_.front();
        pending_.pop_front();
      }
      sink.OnOpCompleted(op_id, ok);
      ++fired;
    }
    return fired;
  }

  std::size_t PumpAll(bool ok, CompletionSink& sink) {
    return Pump(pending(), ok, sink);
  }

  std::size_t pending() const {
    std::lock_guard<std::mutex> lock(mu_);
    return pending_.size();
  }

 private:
  mutable std::mutex mu_;
  std::deque<std::uint64_t> pending_;
};

// ---------------------------------------------------------------------------

TEST(TaskEngineTest, PutGetLifecycleWithMetadataEffects) {
  const ModelLayout layout = Homogeneous(2);
  MetadataPlane plane(layout, std::make_unique<RadixIndex>(layout),
                      TestPools());
  TaskEngine engine(&plane, std::make_shared<InlineExecutor>());
  const std::vector<TokenId> tokens = Tok({1, 2, 3, 4, 5, 6, 7, 8});

  // Put: staged -> executed inline -> committed.
  MetadataPlane::PutPlan plan = plane.StagePut(tokens);
  const TaskHandle put = engine.SubmitPut(plan, Slots({0, 1, 2, 3}));
  EXPECT_EQ(WaitOne(engine, put, std::chrono::milliseconds(1000)),
            TaskState::kSucceeded);

  const MatchResult match = plane.MatchForGet(tokens);
  EXPECT_EQ(match.matched_blocks, 4u);

  // Get: loads matched blocks, releases pins on terminal.
  const TaskHandle get = engine.SubmitGet(match, Slots({10, 11, 12, 13}));
  EXPECT_EQ(WaitOne(engine, get, std::chrono::milliseconds(1000)),
            TaskState::kSucceeded);

  // Pins released by the engine: everything is evictable again.
  const Evicted evicted = plane.EvictBlocks(4);
  EXPECT_EQ(evicted.blocks.size(), static_cast<std::size_t>(4));
  EXPECT_EQ(engine.num_in_flight(), 0u);
}

TEST(TaskEngineTest, FullyMatchedGetIsZeroOpSuccess) {
  const ModelLayout layout = Homogeneous(2);
  MetadataPlane plane(layout, std::make_unique<RadixIndex>(layout),
                      TestPools());
  auto executor = std::make_shared<ManualExecutor>();
  TaskEngine engine(&plane, executor);
  const MatchResult empty = plane.MatchForGet(Tok({5, 6}));
  ASSERT_EQ(empty.matched_blocks, 0u);
  const TaskHandle handle = engine.SubmitGet(empty, {});
  EXPECT_EQ(WaitOne(engine, handle, std::chrono::milliseconds(100)),
            TaskState::kSucceeded);
  EXPECT_EQ(executor->pending(), static_cast<std::size_t>(0));
}

TEST(TaskEngineTest, SubmitValidation) {
  const ModelLayout layout = Homogeneous(2);
  MetadataPlane plane(layout, std::make_unique<RadixIndex>(layout),
                      TestPools());
  TaskEngine engine(&plane, std::make_shared<InlineExecutor>());

  MetadataPlane::PutPlan plan = plane.StagePut(Tok({1, 2, 3, 4}));
  EXPECT_THROW((void)engine.SubmitPut(plan, Slots({0, 1, 2})),
               std::invalid_argument);  // Slot-count mismatch.

  // Unpinned match with matched blocks is rejected.
  stratakv::metadata::MatchOptions raw{/*lock_matched=*/false,
                                       /*require_ready=*/false};
  const MatchResult unpinned =
      plane.index().Match(Tok({1, 2, 3, 4}), 2, raw);
  ASSERT_EQ(unpinned.matched_blocks, 2u);  // Staged, not ready.
  EXPECT_THROW((void)engine.SubmitGet(unpinned, Slots({0, 1})),
               std::invalid_argument);
}

TEST(TaskEngineTest, BackpressureRejectsThenRecovers) {
  const ModelLayout layout = Homogeneous(2);
  MetadataPlane plane(layout, std::make_unique<RadixIndex>(layout),
                      TestPools());
  auto executor = std::make_shared<ManualExecutor>();
  TaskEngineConfig config;
  config.max_in_flight = 2;
  TaskEngine engine(&plane, executor, config);

  const MetadataPlane::PutPlan a = plane.StagePut(Tok({1, 2, 3, 4}));
  const MetadataPlane::PutPlan b = plane.StagePut(Tok({5, 6, 7, 8}));
  (void)engine.SubmitPut(a, Slots({0, 1}));
  (void)engine.SubmitPut(b, Slots({2, 3}));
  EXPECT_EQ(engine.num_in_flight(), 2u);

  const MetadataPlane::PutPlan c = plane.StagePut(Tok({9, 10, 11, 12}));
  EXPECT_THROW((void)engine.SubmitPut(c, Slots({4, 5})),
               stratakv::NotEnoughSpaceError);

  (void)executor->PumpAll(true, engine);
  EXPECT_EQ(engine.num_in_flight(), 0u);
  (void)engine.SubmitPut(c, Slots({4, 5}));  // Accepted again.
}

TEST(TaskEngineTest, CancelPutKeepsStagedEntryUntilAbort) {
  const ModelLayout layout = Homogeneous(2);
  MetadataPlane plane(layout, std::make_unique<RadixIndex>(layout),
                      TestPools());
  auto executor = std::make_shared<ManualExecutor>();
  TaskEngine engine(&plane, executor);

  MetadataPlane::PutPlan plan = plane.StagePut(Tok({1, 2, 3, 4}));
  const TaskHandle handle = engine.SubmitPut(plan, Slots({0, 1}));
  EXPECT_TRUE(engine.Cancel(handle));
  EXPECT_EQ(State(engine, handle), TaskState::kCancelled);
  EXPECT_FALSE(engine.Cancel(handle));  // Terminal: no-op.

  // Late completions for the cancelled task are dropped.
  (void)executor->PumpAll(true, engine);
  EXPECT_EQ(State(engine, handle), TaskState::kCancelled);

  // Entry was never committed; abort recycles the reservation.
  EXPECT_EQ(plane.MatchForGet(Tok({1, 2, 3, 4})).matched_blocks, 0u);
  const std::uint32_t free_before = plane.num_free_blocks(0);
  const Evicted reclaimed = plane.AbortPut(plan.entry);
  EXPECT_EQ(reclaimed.blocks.size(), static_cast<std::size_t>(2));
  EXPECT_EQ(plane.num_free_blocks(0) - free_before, 2u);
}

TEST(TaskEngineTest, OpFailureFailsTaskAndReleasesPins) {
  const ModelLayout layout = Homogeneous(2);
  MetadataPlane plane(layout, std::make_unique<RadixIndex>(layout),
                      TestPools());
  auto executor = std::make_shared<ManualExecutor>();
  TaskEngine engine(&plane, executor);

  // Failing put: task fails, entry stays staged, error is surfaced.
  MetadataPlane::PutPlan plan = plane.StagePut(Tok({1, 2, 3, 4}));
  const TaskHandle put = engine.SubmitPut(plan, Slots({0, 1}));
  (void)executor->Pump(1, /*ok=*/false, engine);
  EXPECT_EQ(WaitOne(engine, put, std::chrono::milliseconds(100)),
            TaskState::kFailed);
  const WaitResult done =
      engine.Wait(std::span<const TaskHandle>(&put, 1),
                  std::chrono::milliseconds(1));
  EXPECT_FALSE(done.items.at(0).error.empty());
  EXPECT_EQ(plane.MatchForGet(Tok({1, 2, 3, 4})).matched_blocks, 0u);
  (void)plane.AbortPut(plan.entry);

  // Successful put, then a failing get: pins must still be released.
  MetadataPlane::PutPlan ok_plan = plane.StagePut(Tok({5, 6, 7, 8}));
  const TaskHandle put2 = engine.SubmitPut(ok_plan, Slots({2, 3}));
  (void)executor->PumpAll(true, engine);
  ASSERT_EQ(State(engine, put2), TaskState::kSucceeded);

  const MatchResult match = plane.MatchForGet(Tok({5, 6, 7, 8}));
  ASSERT_EQ(match.matched_blocks, 2u);
  const TaskHandle get = engine.SubmitGet(match, Slots({7, 8}));
  (void)executor->Pump(1, /*ok=*/false, engine);
  ASSERT_EQ(State(engine, get), TaskState::kFailed);
  const Evicted evicted = plane.EvictBlocks(2);  // Unpinned again.
  EXPECT_EQ(evicted.blocks.size(), static_cast<std::size_t>(2));
}

TEST(TaskEngineTest, WaitTimeoutLeavesTaskPending) {
  const ModelLayout layout = Homogeneous(2);
  MetadataPlane plane(layout, std::make_unique<RadixIndex>(layout),
                      TestPools());
  auto executor = std::make_shared<ManualExecutor>();
  TaskEngine engine(&plane, executor);

  const MetadataPlane::PutPlan plan = plane.StagePut(Tok({1, 2, 3, 4}));
  const TaskHandle handle = engine.SubmitPut(plan, Slots({0, 1}));
  EXPECT_EQ(WaitOne(engine, handle, std::chrono::milliseconds(30)),
            TaskState::kLaunched);

  (void)executor->PumpAll(true, engine);
  EXPECT_EQ(WaitOne(engine, handle, std::chrono::milliseconds(1000)),
            TaskState::kSucceeded);
}

TEST(TaskEngineTest, CompletionFdSignalsAndDrainDispatches) {
  const ModelLayout layout = Homogeneous(2);
  MetadataPlane plane(layout, std::make_unique<RadixIndex>(layout),
                      TestPools());
  auto executor = std::make_shared<ManualExecutor>();
  TaskEngine engine(&plane, executor);

  std::vector<TaskHandle> handles;
  for (int i = 0; i < 3; ++i) {
    const std::vector<TokenId> tokens =
        Tok({1 + i, 2, 3 + i, 4, 5 + i, 6, 7 + i, 8});
    const MetadataPlane::PutPlan plan = plane.StagePut(tokens);
    std::vector<BlockId> gpu(plan.blocks_to_store.size(), 0);
    handles.push_back(engine.SubmitPut(plan, gpu));
  }
  (void)executor->PumpAll(true, engine);

  struct pollfd pfd{};
  pfd.fd = engine.CompletionFd();
  pfd.events = POLLIN;
  EXPECT_GT(::poll(&pfd, 1, /*timeout_ms=*/100), 0);

  std::vector<std::pair<TaskHandle, TaskState>> events;
  std::thread::id dispatch_thread{};
  engine.SetCompletionCallback([&](TaskHandle h, TaskState s) {
    if (dispatch_thread == std::thread::id()) {
      dispatch_thread = std::this_thread::get_id();
    }
    events.emplace_back(h, s);
  });
  engine.DrainCompletions();
  EXPECT_EQ(events.size(), static_cast<std::size_t>(3));
  EXPECT_TRUE(dispatch_thread == std::this_thread::get_id());

  // Unknown/late op ids from executors are tolerated (no crash).
  engine.OnOpCompleted(0xFFFFFFFFFFFFULL, true);

  // Pipe drained: poll reports not-readable now.
  struct pollfd pfd2{};
  pfd2.fd = engine.CompletionFd();
  pfd2.events = POLLIN;
  EXPECT_NE(::poll(&pfd2, 1, /*timeout_ms=*/0), 1);
}

TEST(TaskEngineTest, StressTenKTasksWithRandomCancels) {
  const ModelLayout layout = Homogeneous(2);
  MetadataPlane plane(layout, std::make_unique<RadixIndex>(layout),
                      TestPools());
  auto executor = std::make_shared<ManualExecutor>();
  TaskEngine engine(&plane, executor);

  std::mt19937_64 rng(0xD00DULL);
  std::vector<TaskHandle> handles;
  std::vector<MetadataPlane::PutPlan> plans;

  auto random_tokens = [&]() {
    std::uniform_int_distribution<int> len(1, 6);
    std::uniform_int_distribution<int> tok(1, 6);
    std::vector<TokenId> tokens;
    const std::uint32_t blocks = static_cast<std::uint32_t>(len(rng));
    for (std::uint32_t b = 0; b < blocks * 2; ++b) {
      tokens.push_back(tok(rng));
    }
    return tokens;
  };

  constexpr int kTasks = 10000;
  for (int i = 0; i < kTasks; ++i) {
    const int action = std::uniform_int_distribution<int>(0, 9)(rng);
    if (action < 5) {
      const std::vector<TokenId> tokens = random_tokens();
      MetadataPlane::PutPlan plan;
      try {
        plan = plane.StagePut(tokens);
      } catch (const stratakv::NotEnoughSpaceError&) {
        continue;  // Pools momentarily full.
      }
      if (plan.entry == nullptr) {
        continue;  // Fully present already.
      }
      std::vector<BlockId> gpu(plan.blocks_to_store.size(), 0);
      try {
        handles.push_back(engine.SubmitPut(plan, gpu));
      } catch (const stratakv::NotEnoughSpaceError&) {
        (void)plane.AbortPut(plan.entry);  // Saturated: release.
        continue;
      }
      plans.push_back(plan);
    } else if (action < 7 && !handles.empty()) {
      std::uniform_int_distribution<std::size_t> pick(0,
                                                      handles.size() - 1);
      (void)engine.Cancel(handles[pick(rng)]);
    } else if (action < 9) {
      (void)executor->Pump(
          std::uniform_int_distribution<std::size_t>(1, 8)(rng), true,
          engine);
    } else {
      const MatchResult match = plane.MatchForGet(random_tokens());
      std::vector<BlockId> slots_size(match.block_refs.size(), 0);
      try {
        handles.push_back(engine.SubmitGet(match, slots_size));
      } catch (const stratakv::NotEnoughSpaceError&) {
        plane.Release(match);  // Saturated: drop the pin we took.
      }
    }
  }
  (void)executor->PumpAll(true, engine);

  // Every task terminal, nothing in flight.
  EXPECT_EQ(engine.num_in_flight(), 0u);
  for (const TaskHandle& handle : handles) {
    EXPECT_TRUE(stratakv::common::IsTerminal(State(engine, handle)));
  }

  // Conservation: settle staged entries (abort; commit if extended),
  // evict everything, pools whole again.
  for (const MetadataPlane::PutPlan& plan : plans) {
    try {
      (void)plane.AbortPut(plan.entry);
    } catch (const stratakv::LogicError&) {
      plane.CommitPut(plan.entry);  // Extended by a later insert.
    }
  }
  for (int i = 0; i < 10000; ++i) {
    const Evicted evicted = plane.EvictBlocks(1u << 30);
    if (evicted.blocks.empty()) break;
  }
  EXPECT_EQ(plane.num_free_blocks(0), TestPools().blocks_per_group);
}

TEST(TaskEngineTest, ShutdownMidFlightCancelsAndIsIdempotent) {
  const ModelLayout layout = Homogeneous(2);
  MetadataPlane plane(layout, std::make_unique<RadixIndex>(layout),
                      TestPools());
  auto executor = std::make_shared<ManualExecutor>();
  TaskEngine engine(&plane, executor);

  std::vector<TaskHandle> handles;
  for (int i = 0; i < 8; ++i) {
    const std::vector<TokenId> tokens =
        Tok({1 + i, 2, 3 + i, 4, 5 + i, 6, 7 + i, 8});
    const MetadataPlane::PutPlan plan = plane.StagePut(tokens);
    std::vector<BlockId> gpu(plan.blocks_to_store.size(), 0);
    handles.push_back(engine.SubmitPut(plan, gpu));
  }
  EXPECT_GT(executor->pending(), static_cast<std::size_t>(0));

  engine.Shutdown(std::chrono::milliseconds(100));
  engine.Shutdown(std::chrono::milliseconds(100));  // Idempotent.
  EXPECT_EQ(engine.num_in_flight(), 0u);
  for (const TaskHandle& handle : handles) {
    EXPECT_EQ(State(engine, handle), TaskState::kCancelled);
  }

  const MetadataPlane::PutPlan plan = plane.StagePut(Tok({42, 1, 43, 2}));
  EXPECT_THROW((void)engine.SubmitPut(plan, Slots({0, 1})),
               stratakv::LogicError);

  // Late completions after shutdown are still dropped safely.
  (void)executor->PumpAll(true, engine);
}

}  // namespace
