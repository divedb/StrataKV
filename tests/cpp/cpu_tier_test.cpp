// Phase E (E2): CpuTier tests — pool geometry per the layout math,
// id→address mapping (the contract the transfers will use), bounds
// checking, and the documented split between byte ownership (tier) and
// id ownership (MetadataPlane allocators).

#ifdef STRATAKV_GTEST_SHIM
#include "gtest_shim.h"
#else
#include <gtest/gtest.h>
#endif

#include <cstdint>

#include "stratakv/common/exceptions.h"
#include "stratakv/common/model_layout.h"
#include "stratakv/data/cpu_tier.h"
#include "stratakv/metadata/metadata_plane.h"

namespace {

using stratakv::common::FullAttnSpec;
using stratakv::common::LayerGroup;
using stratakv::common::LinearStateSpec;
using stratakv::common::ModelLayout;
using stratakv::data::CpuTier;
using stratakv::data::CpuTierConfig;
using stratakv::metadata::MetadataPlane;
using stratakv::metadata::MetadataPlaneConfig;
using stratakv::metadata::RadixIndex;

ModelLayout Hybrid() {
  LayerGroup full;
  full.name = "full_attn";
  full.spec = FullAttnSpec{2, 64};
  full.layer_indices = {0, 2};
  full.tokens_per_block = 2;
  LayerGroup linear;
  linear.name = "linear_state";
  linear.spec = LinearStateSpec{8192};
  linear.layer_indices = {1, 3};
  linear.tokens_per_block = 0;
  return ModelLayout::FromGroups({full, linear});
}

CpuTierConfig TierConfigOf(const MetadataPlaneConfig& plane_config) {
  CpuTierConfig tier_config;
  tier_config.blocks_per_group = plane_config.blocks_per_group;
  tier_config.state_objects_per_group =
      plane_config.state_objects_per_group;
  return tier_config;
}

TEST(CpuTierTest, PoolGeometryMatchesLayoutMath) {
  const ModelLayout layout = Hybrid();
  MetadataPlaneConfig plane_config;
  plane_config.blocks_per_group = 16;
  plane_config.state_objects_per_group = 8;
  CpuTier tier(layout, TierConfigOf(plane_config));

  // full_attn fp16: 2(K+V) * 2 tokens/block * 2 heads * 64 = 1024 B.
  EXPECT_EQ(tier.PoolBytesForGroup(0), 16u * 1024u);
  const stratakv::common::BufferView pool = tier.PoolForGroup(0);
  EXPECT_TRUE(pool.base != nullptr);
  EXPECT_EQ(pool.kind, stratakv::common::DeviceKind::kCpu);
  EXPECT_EQ(pool.size_bytes, 16u * 1024u);
  EXPECT_EQ(tier.tokens_per_block(), 2u);
}

TEST(CpuTierTest, AddressMathAndBounds) {
  const ModelLayout layout = Hybrid();
  MetadataPlaneConfig plane_config;
  plane_config.blocks_per_group = 16;
  plane_config.state_objects_per_group = 8;
  CpuTier tier(layout, TierConfigOf(plane_config));

  char* base = static_cast<char*>(tier.PoolForGroup(0).base);
  EXPECT_EQ(tier.BlockAddress(0, 0), base);
  EXPECT_EQ(static_cast<char*>(tier.BlockAddress(0, 1)), base + 1024);
  EXPECT_EQ(static_cast<char*>(tier.BlockAddress(0, 15)),
            base + 15 * 1024);

  char* states = static_cast<char*>(tier.StateAddress(1, 0));
  EXPECT_TRUE(states != nullptr);
  EXPECT_EQ(static_cast<char*>(tier.StateAddress(1, 7)),
            states + 7 * 8192);

  // Out-of-range and wrong-kind ids fail loudly.
  EXPECT_THROW((void)tier.BlockAddress(0, 16), stratakv::LogicError);
  EXPECT_THROW((void)tier.BlockAddress(1, 0), stratakv::LogicError);
  EXPECT_THROW((void)tier.StateAddress(0, 0), stratakv::LogicError);
  EXPECT_THROW((void)tier.StateAddress(1, 8), stratakv::LogicError);
}

TEST(CpuTierTest, IdsFlowThroughPlaneIntoTierAddresses) {
  // The Phase C/D id space and the Phase E byte space meet: StagePut
  // allocates ids, the tier resolves them to distinct in-pool slots,
  // AbortPut recycles them.
  const ModelLayout layout = Hybrid();
  MetadataPlaneConfig plane_config;
  plane_config.blocks_per_group = 8;
  plane_config.state_objects_per_group = 4;
  plane_config.state_object_bytes = 8192;
  MetadataPlane plane(layout, std::make_unique<RadixIndex>(layout),
                      plane_config);
  CpuTier tier(layout, TierConfigOf(plane_config));

  const MetadataPlane::PutPlan plan =
      plane.StagePut(std::vector<stratakv::common::TokenId>{1, 2, 3, 4, 5, 6});
  ASSERT_EQ(plan.blocks_to_store.size(), static_cast<std::size_t>(3));

  std::vector<void*> addresses;
  for (const auto& ref : plan.blocks_to_store) {
    addresses.push_back(tier.BlockAddress(ref.group, ref.block));
  }
  for (void* address : addresses) {
    EXPECT_TRUE(address != nullptr);
  }
  EXPECT_TRUE(addresses[0] != addresses[1]);
  EXPECT_TRUE(addresses[1] != addresses[2]);

  // State slot for the sidecar snapshot.
  const stratakv::common::StateRef& state = plan.states_to_snapshot[0];
  EXPECT_TRUE(tier.StateAddress(state.group, state.object) != nullptr);

  (void)plane.AbortPut(plan.entry);
  // Ids recycled; the addresses remain valid tier memory (no free of
  // individual slots — pools are wholesale).
  EXPECT_EQ(plane.num_free_blocks(0), plane_config.blocks_per_group);
}

}  // namespace
