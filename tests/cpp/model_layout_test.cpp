// Unit tests for ModelLayout: heterogeneous spec tables, dense layer
// coverage, and the rejection paths (Phase B acceptance criteria).

#ifdef STRATAKV_GTEST_SHIM
#include "gtest_shim.h"
#else
#include <gtest/gtest.h>
#endif

#include <stdexcept>

#include "stratakv/common/model_layout.h"

namespace {

using namespace stratakv::common;

// The single-group homogeneous layout audited on the FlexKV debug host
// (Qwen2-0.5B: 24 full-attention layers, GQA 2 KV heads, head dim 64).
ModelLayout Qwen2Shaped() {
  LayerGroup full;
  full.name = "full_attn";
  full.spec = FullAttnSpec{2, 64};
  full.layer_indices = std::vector<int>(24);
  for (int i = 0; i < 24; ++i) full.layer_indices[static_cast<std::size_t>(i)] = i;
  full.tokens_per_block = 16;
  return ModelLayout::FromGroups({full});
}

// Qwen3-Next-shaped hybrid: 12 full-attention layers + 36 linear-attention
// layers (3:1 ratio), with fixed-size recurrent state sidecars.
ModelLayout Qwen3NextShaped() {
  LayerGroup full;
  full.name = "full_attn";
  full.spec = FullAttnSpec{4, 128};
  full.layer_indices = {0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44};
  full.tokens_per_block = 16;

  LayerGroup linear;
  linear.name = "linear_state";
  linear.spec = LinearStateSpec{2048 * 128 * 2};
  linear.layer_indices.reserve(36);
  for (int i = 0; i < 48; ++i) {
    if (i % 4 != 0) {
      linear.layer_indices.push_back(i);
    }
  }
  linear.tokens_per_block = 0;

  return ModelLayout::FromGroups({full, linear});
}

// DeepSeek-V4-shaped hybrid: full-attention KV + sliding-window KV +
// fixed-size indexer/compress state, three heterogeneous groups.
ModelLayout DeepSeekV4Shaped() {
  LayerGroup full;
  full.name = "full_attn";
  full.spec = FullAttnSpec{8, 128};
  full.layer_indices.resize(60);
  for (int i = 0; i < 60; ++i) {
    full.layer_indices[static_cast<std::size_t>(i)] = i;
  }
  full.tokens_per_block = 16;

  LayerGroup swa;
  swa.name = "swa";
  swa.spec = SlidingWindowSpec{8, 128, 64};
  swa.layer_indices = {60, 61, 62, 63, 64, 65, 66, 67, 68, 69};
  swa.tokens_per_block = 16;

  LayerGroup indexer;
  indexer.name = "indexer_state";
  indexer.spec = LinearStateSpec{1 << 20};
  indexer.layer_indices = {70, 71};
  indexer.tokens_per_block = 0;

  return ModelLayout::FromGroups({full, swa, indexer});
}

TEST(ModelLayoutTest, HomogeneousSingleGroup) {
  const ModelLayout layout = Qwen2Shaped();
  EXPECT_EQ(static_cast<int>(layout.num_groups()), 1);
  EXPECT_EQ(layout.total_layers(), 24);
  EXPECT_TRUE(layout.IsHomogeneousKV());
  ASSERT_TRUE(layout.GroupForLayer(0) != nullptr);
  EXPECT_EQ(layout.GroupForLayer(0)->name, "full_attn");
  EXPECT_EQ(layout.GroupForLayer(23)->name, "full_attn");
  EXPECT_TRUE(layout.GroupForLayer(-1) == nullptr);
  EXPECT_TRUE(layout.GroupForLayer(24) == nullptr);
}

TEST(ModelLayoutTest, Qwen3NextShapedHybrid) {
  const ModelLayout layout = Qwen3NextShaped();
  EXPECT_EQ(static_cast<int>(layout.num_groups()), 2);
  EXPECT_EQ(layout.total_layers(), 48);
  EXPECT_FALSE(layout.IsHomogeneousKV());
  EXPECT_EQ(layout.GroupForLayer(0)->name, "full_attn");
  EXPECT_EQ(layout.GroupForLayer(1)->name, "linear_state");
  EXPECT_EQ(layout.GroupForLayer(47)->name, "linear_state");
  // Full-attn members are exactly the multiples of 4.
  int full_members = 0;
  for (int i = 0; i < 48; ++i) {
    if (layout.GroupForLayer(i)->name == "full_attn") ++full_members;
  }
  EXPECT_EQ(full_members, 12);
}

TEST(ModelLayoutTest, DeepSeekV4ShapedThreeGroups) {
  const ModelLayout layout = DeepSeekV4Shaped();
  EXPECT_EQ(static_cast<int>(layout.num_groups()), 3);
  EXPECT_EQ(layout.total_layers(), 72);
  EXPECT_FALSE(layout.IsHomogeneousKV());
  EXPECT_EQ(layout.GroupForLayer(5)->name, "full_attn");
  EXPECT_EQ(layout.GroupForLayer(65)->name, "swa");
  EXPECT_EQ(layout.GroupForLayer(71)->name, "indexer_state");
}

TEST(ModelLayoutDeathTest, RejectsMalformedInputs) {
  // Empty group set.
  EXPECT_THROW(ModelLayout::FromGroups({}), std::invalid_argument);

  // Duplicate group names.
  LayerGroup a;
  a.name = "full_attn";
  a.spec = FullAttnSpec{2, 64};
  a.layer_indices = {0};
  a.tokens_per_block = 16;
  LayerGroup b = a;
  b.layer_indices = {1};
  EXPECT_THROW(ModelLayout::FromGroups({a, b}), std::invalid_argument);

  // Overlapping layer indices.
  LayerGroup c = a;
  c.name = "other";
  EXPECT_THROW(ModelLayout::FromGroups({a, c}), std::invalid_argument);

  // Non-dense coverage (layers {0, 2} leaves a gap).
  LayerGroup gappy = a;
  gappy.name = "gappy";
  gappy.layer_indices = {0, 2};
  EXPECT_THROW(ModelLayout::FromGroups({gappy}), std::invalid_argument);

  // Linear-state group must not claim a token block size.
  LayerGroup linear_bad;
  linear_bad.name = "linear_state";
  linear_bad.spec = LinearStateSpec{4096};
  linear_bad.layer_indices = {0};
  linear_bad.tokens_per_block = 16;
  EXPECT_THROW(ModelLayout::FromGroups({linear_bad}),
               std::invalid_argument);

  // Token-addressable group must have a positive block size.
  LayerGroup token_bad;
  token_bad.name = "full_attn";
  token_bad.spec = FullAttnSpec{2, 64};
  token_bad.layer_indices = {0};
  token_bad.tokens_per_block = 0;
  EXPECT_THROW(ModelLayout::FromGroups({token_bad}),
               std::invalid_argument);

  // Structurally invalid spec.
  LayerGroup spec_bad;
  spec_bad.name = "full_attn";
  spec_bad.spec = FullAttnSpec{0, 64};
  spec_bad.layer_indices = {0};
  spec_bad.tokens_per_block = 16;
  EXPECT_THROW(ModelLayout::FromGroups({spec_bad}),
               std::invalid_argument);
}

}  // namespace
