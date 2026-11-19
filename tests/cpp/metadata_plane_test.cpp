// Phase C facade tests: MetadataPlane wiring of index + allocators +
// eviction, covering the get/put staging lifecycle the Phase D task
// engine will drive.

#ifdef STRATAKV_GTEST_SHIM
#include "gtest_shim.h"
#else
#include <gtest/gtest.h>
#endif

#include <cstdint>
#include <vector>

#include "stratakv/common/exceptions.h"
#include "stratakv/common/model_layout.h"
#include "stratakv/metadata/index.h"
#include "stratakv/metadata/metadata_plane.h"

namespace {

using stratakv::common::FullAttnSpec;
using stratakv::common::LayerGroup;
using stratakv::common::LinearStateSpec;
using stratakv::common::ModelLayout;
using stratakv::common::TokenId;
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

ModelLayout Hybrid(std::uint32_t tpb) {
  LayerGroup full;
  full.name = "full_attn";
  full.spec = FullAttnSpec{4, 128};
  full.layer_indices = {0, 2};
  full.tokens_per_block = static_cast<int>(tpb);
  LayerGroup linear;
  linear.name = "linear_state";
  linear.spec = LinearStateSpec{4096};
  linear.layer_indices = {1, 3};
  linear.tokens_per_block = 0;
  return ModelLayout::FromGroups({full, linear});
}

MetadataPlaneConfig SmallPools() {
  MetadataPlaneConfig config;
  config.blocks_per_group = 8;
  config.state_objects_per_group = 2;
  config.state_object_bytes = 4096;
  return config;
}

std::vector<TokenId> Tok(std::initializer_list<TokenId> in) {
  return std::vector<TokenId>(in);
}

TEST(MetadataPlaneTest, PutGetLifecycle) {
  const ModelLayout layout = Homogeneous(2);
  MetadataPlane plane(layout, std::make_unique<RadixIndex>(layout),
                      SmallPools());
  const std::vector<TokenId> tokens = Tok({1, 2, 3, 4, 5, 6, 7, 8});

  // Cold: no match, put must store everything.
  const MetadataPlane::PutPlan preview = plane.PlanPut(tokens);
  EXPECT_EQ(preview.matched_blocks, 0u);
  EXPECT_EQ(preview.new_blocks, 4u);
  EXPECT_TRUE(preview.feasible);

  MetadataPlane::PutPlan plan = plane.StagePut(tokens);
  ASSERT_TRUE(plan.entry != nullptr);
  EXPECT_EQ(plan.new_blocks, 4u);
  EXPECT_EQ(plan.blocks_to_store.size(), static_cast<std::size_t>(4));
  EXPECT_TRUE(plan.states_to_snapshot.empty());
  EXPECT_FALSE(plan.entry->is_ready);

  // In-flight: staging match sees the raw prefix, get sees nothing.
  EXPECT_EQ(plane.PlanPut(tokens).new_blocks, 0u);
  EXPECT_EQ(plane.MatchForGet(tokens).matched_blocks, 0u);

  plane.CommitPut(plan.entry);

  const MatchResult match = plane.MatchForGet(tokens);
  EXPECT_EQ(match.matched_blocks, 4u);
  EXPECT_TRUE(match.pin != nullptr);  // MatchForGet pins its prefix.

  plane.Release(match);  // Drop the pin taken by MatchForGet.
}

TEST(MetadataPlaneTest, StagePutReservesAndEvictRecycles) {
  const ModelLayout layout = Homogeneous(2);
  MetadataPlane plane(layout, std::make_unique<RadixIndex>(layout),
                      SmallPools());
  const std::vector<TokenId> tokens = Tok({1, 2, 3, 4, 5, 6, 7, 8});

  MetadataPlane::PutPlan plan = plane.StagePut(tokens);
  plane.CommitPut(plan.entry);

  stratakv::common::CacheEntry* entry = plan.entry;
  const stratakv::metadata::Evicted evicted = plane.EvictBlocks(4);
  EXPECT_EQ(evicted.blocks.size(), static_cast<std::size_t>(4));
  EXPECT_EQ(evicted.entries.size(), static_cast<std::size_t>(1));
  EXPECT_GT(evicted.bytes_reclaimed, 0u);
  // 4 blocks * 4 layers * 2(K+V) * 2 tpb * 2 heads * 64 head_size * 2 B
  // (model-wide block: per-layer slice × layers — FlexKV contract).
  EXPECT_EQ(evicted.bytes_reclaimed, 4ull * 4 * 2 * 2 * 2 * 64 * 2);

  // The entry handle died with the eviction.
  EXPECT_THROW(plane.CommitPut(entry), stratakv::LogicError);
  // Pools are whole again: a fresh put succeeds with recycled ids.
  MetadataPlane::PutPlan again = plane.StagePut(tokens);
  EXPECT_EQ(again.new_blocks, 4u);
  (void)again;
}

TEST(MetadataPlaneTest, ExhaustionLeavesNothingReserved) {
  const ModelLayout layout = Homogeneous(2);
  MetadataPlane plane(layout, std::make_unique<RadixIndex>(layout),
                      SmallPools());  // 8 blocks per group.
  const std::vector<TokenId> first = Tok({1, 2, 3, 4, 5, 6, 7, 8});
  const std::vector<TokenId> second = Tok({9, 10, 11, 12, 13, 14, 15, 16});

  MetadataPlane::PutPlan a = plane.StagePut(first);   // 4 blocks.
  MetadataPlane::PutPlan b = plane.StagePut(second);  // 4 blocks — full.
  const std::vector<TokenId> third = Tok({17, 18, 19, 20, 21, 22, 23, 24});
  EXPECT_FALSE(plane.PlanPut(third).feasible);  // Pools at capacity.

  EXPECT_THROW((void)plane.StagePut(third), stratakv::NotEnoughSpaceError);

  // Commit what staged; evicting one block makes room again.
  plane.CommitPut(a.entry);
  plane.CommitPut(b.entry);
  const stratakv::metadata::Evicted evicted = plane.EvictBlocks(4);
  EXPECT_EQ(evicted.blocks.size(), static_cast<std::size_t>(4));
  EXPECT_EQ(plane.PlanPut(third).feasible, true);
}

TEST(MetadataPlaneHybridTest, StagePutReservesStatesPerGroup) {
  const ModelLayout layout = Hybrid(2);
  MetadataPlane plane(layout, std::make_unique<RadixIndex>(layout),
                      SmallPools());
  const std::vector<TokenId> tokens = Tok({1, 2, 3, 4, 5, 6});

  MetadataPlane::PutPlan plan = plane.StagePut(tokens);
  ASSERT_TRUE(plan.entry != nullptr);
  EXPECT_EQ(plan.blocks_to_store.size(), static_cast<std::size_t>(3));
  EXPECT_EQ(plan.states_to_snapshot.size(), static_cast<std::size_t>(1));
  EXPECT_EQ(plan.states_to_snapshot[0].group, 1u);  // linear_state group.
  EXPECT_FALSE(plan.entry->IsCompleteFor(layout));  // Not ready yet.

  plane.CommitPut(plan.entry);
  EXPECT_TRUE(plan.entry->IsCompleteFor(layout));

  const MatchResult match = plane.MatchForGet(tokens);
  EXPECT_EQ(match.matched_blocks, 3u);
  EXPECT_EQ(match.state_refs.size(), static_cast<std::size_t>(1));
  plane.Release(match);

  // State pool had 2 objects; evicting the single boundary node
  // recycles both blocks and the state object.
  const stratakv::metadata::Evicted evicted = plane.EvictBlocks(1);
  EXPECT_EQ(evicted.blocks.size(), static_cast<std::size_t>(3));
  EXPECT_EQ(evicted.states.size(), static_cast<std::size_t>(1));
}

TEST(MetadataPlaneTest, PinsSurviveEvictionAttempt) {
  const ModelLayout layout = Homogeneous(2);
  MetadataPlane plane(layout, std::make_unique<RadixIndex>(layout),
                      SmallPools());
  const std::vector<TokenId> tokens = Tok({1, 2, 3, 4});
  MetadataPlane::PutPlan plan = plane.StagePut(tokens);
  plane.CommitPut(plan.entry);

  const MatchResult match = plane.MatchForGet(tokens);
  const stratakv::metadata::Evicted evicted = plane.EvictBlocks(4);
  EXPECT_EQ(evicted.blocks.size(), static_cast<std::size_t>(0));

  plane.Release(match);
  const stratakv::metadata::Evicted now = plane.EvictBlocks(4);
  EXPECT_EQ(now.blocks.size(), static_cast<std::size_t>(2));
}

TEST(MetadataPlaneTest, NullIndexRejected) {
  const ModelLayout layout = Homogeneous(2);
  EXPECT_THROW((MetadataPlane{layout, nullptr, SmallPools()}),
               std::invalid_argument);
}

}  // namespace
