// Unit tests for task handles and the CacheEntry no-partial-caching
// invariant (Phase B; the invariant itself is ADR 0003).

#ifdef STRATAKV_GTEST_SHIM
#include "gtest_shim.h"
#else
#include <gtest/gtest.h>
#endif

#include "stratakv/common/handles.h"
#include "stratakv/common/model_layout.h"

namespace {

using namespace stratakv::common;

ModelLayout Qwen2Shaped() {
  LayerGroup full;
  full.name = "full_attn";
  full.spec = FullAttnSpec{2, 64};
  full.layer_indices = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                        16, 17, 18, 19, 20, 21, 22, 23};
  full.tokens_per_block = 16;
  return ModelLayout::FromGroups({full});
}

ModelLayout Qwen3NextShaped() {
  LayerGroup full;
  full.name = "full_attn";
  full.spec = FullAttnSpec{4, 128};
  full.layer_indices = {0, 4, 8, 12};
  full.tokens_per_block = 16;
  LayerGroup linear;
  linear.name = "linear_state";
  linear.spec = LinearStateSpec{65536};
  linear.layer_indices = {1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15};
  linear.tokens_per_block = 0;
  return ModelLayout::FromGroups({full, linear});
}

TEST(TaskHandleTest, DefaultInvalidAndEquality) {
  TaskHandle invalid;
  EXPECT_FALSE(invalid.IsValid());
  EXPECT_EQ(invalid.id(), 0u);

  TaskHandle a(42);
  TaskHandle b(42);
  TaskHandle c(43);
  EXPECT_TRUE(a.IsValid());
  EXPECT_TRUE((a == b));
  EXPECT_TRUE((a != c));
}

TEST(TaskStateTest, TerminalStates) {
  EXPECT_FALSE(IsTerminal(TaskState::kStaged));
  EXPECT_FALSE(IsTerminal(TaskState::kLaunched));
  EXPECT_TRUE(IsTerminal(TaskState::kSucceeded));
  EXPECT_TRUE(IsTerminal(TaskState::kFailed));
  EXPECT_TRUE(IsTerminal(TaskState::kCancelled));
}

TEST(CacheEntryTest, HomogeneousModelCompleteEntry) {
  const ModelLayout layout = Qwen2Shaped();
  CacheEntry entry;
  entry.blocks.push_back(BlockRef{/*tier=*/0, /*group=*/0, /*block=*/7, 1});
  entry.blocks.push_back(BlockRef{0, 0, 8, 1});
  entry.is_ready = true;
  EXPECT_TRUE(entry.IsCompleteFor(layout));
}

TEST(CacheEntryTest, NotReadyEntryIsNeverComplete) {
  const ModelLayout layout = Qwen2Shaped();
  CacheEntry entry;
  entry.blocks.push_back(BlockRef{0, 0, 7, 1});
  entry.is_ready = false;  // Put transfer has not landed yet.
  EXPECT_FALSE(entry.IsCompleteFor(layout));
}

TEST(CacheEntryTest, MissingSidecarStateViolatesInvariant) {
  // ADR 0003: for a hybrid model, an entry that carries the full-attn KV
  // but not the linear-attention state snapshots is NOT loadable — loading
  // a partial prefix would silently produce wrong attention.
  const ModelLayout layout = Qwen3NextShaped();
  CacheEntry partial;
  partial.blocks.push_back(BlockRef{0, /*group=*/0, 3, 1});
  partial.is_ready = true;
  EXPECT_FALSE(partial.IsCompleteFor(layout));

  CacheEntry complete = partial;
  complete.states.push_back(StateRef{0, /*group=*/1, /*object=*/0, 1});
  EXPECT_TRUE(complete.IsCompleteFor(layout));
}

TEST(CacheEntryTest, OutOfRangeGroupIndexRejected) {
  const ModelLayout layout = Qwen2Shaped();
  CacheEntry entry;
  entry.blocks.push_back(BlockRef{0, /*group=*/1, 0, 1});  // Only group 0 exists.
  entry.is_ready = true;
  EXPECT_FALSE(entry.IsCompleteFor(layout));
}

TEST(CacheEntryTest, TokenAddressableGroupNeedsBlocks) {
  const ModelLayout layout = Qwen3NextShaped();
  CacheEntry entry;
  entry.states.push_back(StateRef{0, /*group=*/1, 0, 1});  // Sidecar only.
  entry.is_ready = true;
  EXPECT_FALSE(entry.IsCompleteFor(layout));  // Missing full-attn KV.
}

}  // namespace
