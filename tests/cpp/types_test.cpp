// Unit tests for stratakv::common type helpers (Phase B).

#ifdef STRATAKV_GTEST_SHIM
#include "gtest_shim.h"
#else
#include <gtest/gtest.h>
#endif

#include "stratakv/common/types.h"

namespace {

using namespace stratakv::common;

TEST(DeviceKindTest, ToStringCoversAllKinds) {
  EXPECT_EQ(ToString(DeviceKind::kGpu), "gpu");
  EXPECT_EQ(ToString(DeviceKind::kCpuPinned), "cpu_pinned");
  EXPECT_EQ(ToString(DeviceKind::kCpu), "cpu");
  EXPECT_EQ(ToString(DeviceKind::kSsd), "ssd");
}

TEST(BufferViewTest, ValidityChecks) {
  BufferView view;
  EXPECT_FALSE(view.IsValid());  // Null base, zero size.

  int storage[4] = {0, 0, 0, 0};
  view.base = storage;
  view.size_bytes = sizeof(storage);
  view.kind = DeviceKind::kCpu;
  EXPECT_TRUE(view.IsValid());

  // Row-major strides: [outer, inner] must be non-increasing and the
  // outermost must fit in the region.
  view.strides_bytes = {8, 2};
  EXPECT_TRUE(view.IsValid());
  view.strides_bytes = {2, 8};  // Increasing: not row-major.
  EXPECT_FALSE(view.IsValid());
  view.strides_bytes = {sizeof(storage) + 1};
  EXPECT_FALSE(view.IsValid());
}

TEST(LayerSpecTest, SpecNames) {
  EXPECT_EQ(SpecName(LayerSpec(FullAttnSpec{2, 64})), "full_attn");
  EXPECT_EQ(SpecName(LayerSpec(MlaSpec{512, 64})), "mla");
  EXPECT_EQ(SpecName(LayerSpec(SlidingWindowSpec{2, 64, 32})),
            "sliding_window");
  EXPECT_EQ(SpecName(LayerSpec(LinearStateSpec{4096})), "linear_state");
}

TEST(LayerSpecTest, TokenAddressability) {
  EXPECT_TRUE(IsTokenAddressable(LayerSpec(FullAttnSpec{2, 64})));
  EXPECT_TRUE(IsTokenAddressable(LayerSpec(MlaSpec{512, 64})));
  EXPECT_TRUE(IsTokenAddressable(LayerSpec(SlidingWindowSpec{2, 64, 32})));
  EXPECT_FALSE(IsTokenAddressable(LayerSpec(LinearStateSpec{4096})));
}

TEST(LayerSpecTest, BlockBytesMatchesAuditedGeometry) {
  // The audited Qwen2-0.5B GPU geometry: 16 tokens, 2 KV heads, head dim
  // 64, fp16 -> 8 KiB per layer per block (K and V both counted).
  const FullAttnSpec qwen2{2, 64};
  EXPECT_EQ(BlockBytes(LayerSpec(qwen2), /*tokens_per_block=*/16,
                       /*dtype_bytes=*/2),
            8192u);

  // MLA: one latent (rank + rope) per token.
  const MlaSpec mla{512, 64};
  EXPECT_EQ(BlockBytes(LayerSpec(mla), 16, 2),
            16u * (512 + 64) * 2u);

  // Linear states are not block-granular.
  const LinearStateSpec linear{4096};
  EXPECT_EQ(BlockBytes(LayerSpec(linear), 16, 2), 0u);
  EXPECT_EQ(StateBytes(LayerSpec(linear)), 4096u);
  EXPECT_EQ(StateBytes(LayerSpec(qwen2)), 0u);
}

TEST(LayerSpecTest, ValidationRejectsMalformedSpecs) {
  EXPECT_FALSE(IsValid(LayerSpec(FullAttnSpec{0, 64})));
  EXPECT_FALSE(IsValid(LayerSpec(FullAttnSpec{2, 0})));
  EXPECT_FALSE(IsValid(LayerSpec(MlaSpec{0, 64})));
  EXPECT_FALSE(IsValid(LayerSpec(MlaSpec{512, -1})));
  EXPECT_FALSE(IsValid(LayerSpec(SlidingWindowSpec{2, 64, 0})));
  EXPECT_FALSE(IsValid(LayerSpec(LinearStateSpec{0})));
  EXPECT_TRUE(IsValid(LayerSpec(FullAttnSpec{2, 64})));
}

}  // namespace
