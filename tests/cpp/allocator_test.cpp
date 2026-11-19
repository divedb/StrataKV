// Phase C (C3): allocator tests — Mempool free-list semantics port
// (batch allocation, offset-based consumption, loud double-recycle) and
// the LinearState sidecar pool.

#ifdef STRATAKV_GTEST_SHIM
#include "gtest_shim.h"
#else
#include <gtest/gtest.h>
#endif

#include <cstdint>
#include <vector>

#include "stratakv/common/exceptions.h"
#include "stratakv/common/handles.h"
#include "stratakv/metadata/allocator.h"

namespace {

using stratakv::common::BlockId;
using stratakv::common::StateRef;
using stratakv::metadata::BlockAllocator;
using stratakv::metadata::StatePool;

TEST(BlockAllocatorTest, SequentialBatchAllocation) {
  BlockAllocator pool(8);
  EXPECT_EQ(pool.num_free(), 8u);
  EXPECT_EQ(pool.total_blocks(), 8u);

  std::vector<BlockId> first = pool.Allocate(3);
  // Mempool parity: ids come from the materialized free-id array in
  // ascending order until the tail is consumed.
  EXPECT_EQ(first, (std::vector<BlockId>{0, 1, 2}));
  EXPECT_EQ(pool.num_free(), 5u);
  EXPECT_EQ(pool.num_used(), 3u);

  std::vector<BlockId> second = pool.Allocate(2);
  EXPECT_EQ(second, (std::vector<BlockId>{3, 4}));
}

TEST(BlockAllocatorTest, ExhaustionThrows) {
  BlockAllocator pool(4);
  pool.Allocate(4);
  EXPECT_THROW(pool.Allocate(1), stratakv::NotEnoughSpaceError);
  EXPECT_EQ(pool.num_free(), 0u);
}

TEST(BlockAllocatorTest, ReleaseThenReuseAfterMaterialization) {
  BlockAllocator pool(4);
  const std::vector<BlockId> ids = pool.Allocate(4);
  // Free 1 and 3; the next allocate consumes the materialized tail
  // first (empty), re-materializes, and returns 1 — matching Mempool.
  pool.Release(std::vector<BlockId>{ids[1], ids[3]});
  EXPECT_EQ(pool.num_free(), 2u);
  EXPECT_EQ(pool.Allocate(1), (std::vector<BlockId>{ids[1]}));
  EXPECT_EQ(pool.Allocate(1), (std::vector<BlockId>{ids[3]}));
  EXPECT_THROW(pool.Allocate(1), stratakv::NotEnoughSpaceError);
}

TEST(BlockAllocatorTest, DoubleRecycleIsLoud) {
  BlockAllocator pool(4);
  const std::vector<BlockId> ids = pool.Allocate(2);
  pool.Release(std::vector<BlockId>{ids[0]});
  EXPECT_THROW(pool.Release(std::vector<BlockId>{ids[0]}),
               stratakv::LogicError);
  // ids[1] was never released; releasing it now must still succeed.
  pool.Release(std::vector<BlockId>{ids[1]});
  EXPECT_EQ(pool.num_free(), 4u);  // Both distinct ids back in the pool.
}

TEST(BlockAllocatorTest, OutOfRangeReleaseRejected) {
  BlockAllocator pool(4);
  EXPECT_THROW(pool.Release(std::vector<BlockId>{4}), stratakv::LogicError);
  EXPECT_THROW(pool.Release(std::vector<BlockId>{static_cast<BlockId>(-1)}),
               stratakv::LogicError);
}

TEST(BlockAllocatorTest, ZeroRejects) {
  BlockAllocator pool(4);
  EXPECT_THROW(pool.Allocate(0), std::invalid_argument);
  EXPECT_THROW((BlockAllocator{0}), std::invalid_argument);
}

TEST(StatePoolTest, AcquireReleaseRoundTrip) {
  StatePool pool(2, 4096);
  const StateRef a = pool.Acquire(/*tier=*/0, /*group=*/1);
  const StateRef b = pool.Acquire(0, 1);
  EXPECT_EQ(pool.num_free(), 0u);
  EXPECT_EQ(a.group, 1u);
  EXPECT_EQ(b.group, 1u);
  EXPECT_TRUE(a.object != b.object);

  EXPECT_THROW(pool.Acquire(0, 1), stratakv::NotEnoughSpaceError);
  pool.Release(a);
  EXPECT_EQ(pool.num_free(), 1u);
  const StateRef c = pool.Acquire(0, 1);
  EXPECT_EQ(c.object, a.object);  // FIFO free-list reuse.
}

TEST(StatePoolTest, DoubleReleaseIsLoud) {
  StatePool pool(2, 4096);
  const StateRef ref = pool.Acquire(0, 1);
  pool.Release(ref);
  EXPECT_THROW(pool.Release(ref), stratakv::LogicError);
}

TEST(StatePoolTest, AccountingFields) {
  StatePool pool(16, 65536);
  EXPECT_EQ(pool.total_objects(), 16u);
  EXPECT_EQ(pool.object_bytes(), 65536u);
  EXPECT_THROW((StatePool{0, 8}), std::invalid_argument);
}

}  // namespace
