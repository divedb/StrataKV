// Phase E (E3): TierTable tests — the id→buffer geometry the CUDA
// transport resolves through. CPU-only: GPU registrations are exercised
// with real descriptors over synthetic device views (the registry
// validates structure, not CUDA residency).

#ifdef STRATAKV_GTEST_SHIM
#include "gtest_shim.h"
#else
#include <gtest/gtest.h>
#endif

#include <cstdint>
#include <vector>

#include "stratakv/common/exceptions.h"
#include "stratakv/common/model_layout.h"
#include "stratakv/common/types.h"
#include "stratakv/data/cpu_tier.h"
#include "stratakv/data/registry.h"
#include "stratakv/data/tier_table.h"
#include "stratakv/metadata/metadata_plane.h"

namespace {

using stratakv::common::BufferView;
using stratakv::common::DeviceKind;
using stratakv::common::FullAttnSpec;
using stratakv::common::LayerGroup;
using stratakv::common::LinearStateSpec;
using stratakv::common::ModelLayout;
using stratakv::data::CpuTier;
using stratakv::data::CpuTierConfig;
using stratakv::data::DeviceMemoryRegistry;
using stratakv::data::GroupDescriptor;
using stratakv::data::GroupLayout;
using stratakv::data::TierTable;
using stratakv::metadata::MetadataPlaneConfig;

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

CpuTierConfig PageableTier(const MetadataPlaneConfig& plane_config) {
  CpuTierConfig tier_config;
  tier_config.blocks_per_group = plane_config.blocks_per_group;
  tier_config.state_objects_per_group =
      plane_config.state_objects_per_group;
  tier_config.backing = stratakv::data::HostBacking::kPageable;
  return tier_config;
}

BufferView FakeGpuView(void* base, std::uint64_t bytes) {
  BufferView view;
  view.base = base;
  view.kind = DeviceKind::kGpu;
  view.device_id = 0;
  view.size_bytes = bytes;
  return view;
}

TEST(TierTableTest, CpuTierGeometryIsBlockwise) {
  const ModelLayout layout = Hybrid();
  MetadataPlaneConfig plane_config;
  plane_config.blocks_per_group = 16;
  plane_config.state_objects_per_group = 8;
  CpuTier tier(layout, PageableTier(plane_config));

  TierTable table;
  table.AddCpuTier(0, layout, tier);

  const stratakv::data::GroupAddressing* full = table.LookUp(0, 0);
  ASSERT_TRUE(full != nullptr);
  EXPECT_EQ(full->bases.size(), static_cast<std::size_t>(1));
  EXPECT_EQ(full->bases[0], tier.BlockAddress(0, 0));
  EXPECT_EQ(full->slice_bytes, 2048u);  // 2 layers x 1024 B per layer.
  EXPECT_EQ(full->block_stride, 2048u);
  EXPECT_EQ(full->kind, DeviceKind::kCpu);
  EXPECT_FALSE(full->is_state);
  EXPECT_FALSE(full->sm_accessible);  // Pageable: copy engine only.

  const stratakv::data::GroupAddressing* linear = table.LookUp(0, 1);
  ASSERT_TRUE(linear != nullptr);
  EXPECT_EQ(linear->slice_bytes, 16384u);  // 2 layers x 8192 B.
  EXPECT_TRUE(linear->is_state);
  EXPECT_FALSE(linear->sm_accessible);

  EXPECT_EQ(table.LookUp(7, 0), nullptr);
}

TEST(TierTableTest, LayerwiseGpuGeometryPerLayerBases) {
  const ModelLayout layout = Hybrid();
  DeviceMemoryRegistry registry;
  GroupDescriptor descriptor;
  descriptor.group_name = "full_attn";
  descriptor.group = 0;
  descriptor.layout = GroupLayout::kLayerwise;
  descriptor.device_id = 0;
  // Two per-layer bases over 8 block slots x 1024 B per-layer slices.
  descriptor.per_layer = {FakeGpuView(reinterpret_cast<void*>(0x1000000),
                                      8 * 1024),
                          FakeGpuView(reinterpret_cast<void*>(0x2000000),
                                      8 * 1024)};
  const auto id = registry.RegisterGroup(std::move(descriptor));

  TierTable table;
  table.AddGpuTier(1, registry, id, layout, /*capacity_slots=*/8);

  const stratakv::data::GroupAddressing* full = table.LookUp(1, 0);
  ASSERT_TRUE(full != nullptr);
  EXPECT_EQ(full->bases.size(), static_cast<std::size_t>(2));
  EXPECT_EQ(full->block_stride, 1024u);
  EXPECT_EQ(full->slice_bytes, 1024u);
  EXPECT_EQ(full->kind, DeviceKind::kGpu);
  EXPECT_TRUE(full->sm_accessible);
  EXPECT_FALSE(full->is_state);
}

TEST(TierTableTest, BlockwiseGpuGeometrySingleBase) {
  const ModelLayout layout = Hybrid();
  DeviceMemoryRegistry registry;
  GroupDescriptor descriptor;
  descriptor.group_name = "full_attn";
  descriptor.group = 0;
  descriptor.layout = GroupLayout::kBlockwise;
  descriptor.device_id = 0;
  // One flat base over 8 block slots x 2048 B model-wide blocks.
  descriptor.per_layer = {FakeGpuView(reinterpret_cast<void*>(0x3000000),
                                      8 * 2048)};
  const auto id = registry.RegisterGroup(std::move(descriptor));

  TierTable table;
  table.AddGpuTier(1, registry, id, layout, 8);
  const stratakv::data::GroupAddressing* full = table.LookUp(1, 0);
  ASSERT_TRUE(full != nullptr);
  EXPECT_EQ(full->bases.size(), static_cast<std::size_t>(1));
  EXPECT_EQ(full->slice_bytes, 2048u);
  EXPECT_EQ(full->block_stride, 2048u);
}

TEST(TierTableTest, DuplicatesAndBadCapacityFailLoudly) {
  const ModelLayout layout = Hybrid();
  MetadataPlaneConfig plane_config;
  plane_config.blocks_per_group = 4;
  CpuTier tier(layout, PageableTier(plane_config));

  TierTable table;
  table.AddCpuTier(0, layout, tier);
  EXPECT_THROW(table.AddCpuTier(0, layout, tier), stratakv::LogicError);

  DeviceMemoryRegistry registry;
  GroupDescriptor descriptor;
  descriptor.group_name = "full_attn";
  descriptor.group = 0;
  descriptor.layout = GroupLayout::kLayerwise;
  descriptor.device_id = 0;
  descriptor.per_layer = {FakeGpuView(reinterpret_cast<void*>(0x1000000),
                                      8 * 1024),
                          FakeGpuView(reinterpret_cast<void*>(0x2000000),
                                      8 * 1024)};
  const auto id = registry.RegisterGroup(std::move(descriptor));

  // Capacity must divide the view size exactly.
  TierTable other;
  EXPECT_THROW(other.AddGpuTier(1, registry, id, layout, 7),
               stratakv::LogicError);
  EXPECT_THROW(other.AddGpuTier(1, registry, id, layout, 0),
               stratakv::LogicError);
  other.AddGpuTier(1, registry, id, layout, 8);
  EXPECT_THROW(other.AddGpuTier(1, registry, id, layout, 8),
               stratakv::LogicError);  // Duplicate (tier, group).

  // Unknown registrations fail via the registry contract.
  EXPECT_THROW(other.AddGpuTier(2, registry, 999, layout, 8),
               stratakv::LogicError);
}

}  // namespace
