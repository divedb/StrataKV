// Phase C (C1/C2): RadixIndex tests.
//
// The first test replays the exact scenario from FlexKV v1.0.0
// flexkv/cache/radixtree.py __main__ (the recorded golden sequence) and
// asserts identical match lengths and physical-block sets — the C1
// parity requirement. Remaining tests cover the C2 generalizations:
// CacheEntry-bundle payloads, readiness filtering, refcount pins, and
// the ADR 0003 no-partial-caching invariant for hybrid layouts.

#ifdef STRATAKV_GTEST_SHIM
#include "gtest_shim.h"
#else
#include <gtest/gtest.h>
#endif

#include <cstdint>
#include <vector>

#include "stratakv/common/exceptions.h"
#include "stratakv/common/handles.h"
#include "stratakv/common/model_layout.h"
#include "stratakv/common/types.h"
#include "stratakv/metadata/index.h"

namespace {

using stratakv::common::BlockRef;
using stratakv::common::CacheEntry;
using stratakv::common::FullAttnSpec;
using stratakv::common::LayerGroup;
using stratakv::common::LinearStateSpec;
using stratakv::common::ModelLayout;
using stratakv::common::StateRef;
using stratakv::common::TokenId;
using stratakv::metadata::MatchOptions;
using stratakv::metadata::MatchResult;
using stratakv::metadata::RadixIndex;

// Deterministic monotonic clock: every call advances by 1 ns, so LRU
// order is fully reproducible (a real clock's ties would make eviction
// order nondeterministic).
struct TickClock {
  std::uint64_t tick = 0;
  std::uint64_t operator()() { return ++tick; }
};

ModelLayout HomogeneousLayout(std::uint32_t tpb) {
  LayerGroup full;
  full.name = "full_attn";
  full.spec = FullAttnSpec{2, 64};
  full.layer_indices = {0, 1, 2, 3};
  full.tokens_per_block = static_cast<int>(tpb);
  return ModelLayout::FromGroups({full});
}

ModelLayout HybridLayout(std::uint32_t tpb) {
  LayerGroup full;
  full.name = "full_attn";
  full.spec = FullAttnSpec{4, 128};
  full.layer_indices = {0, 4};
  full.tokens_per_block = static_cast<int>(tpb);
  LayerGroup linear;
  linear.name = "linear_state";
  linear.spec = LinearStateSpec{65536};
  linear.layer_indices = {1, 2, 3, 5};
  linear.tokens_per_block = 0;
  return ModelLayout::FromGroups({full, linear});
}

std::vector<TokenId> Tok(std::initializer_list<TokenId> in) {
  return std::vector<TokenId>(in);
}

std::vector<BlockRef> Refs1G(std::initializer_list<std::uint32_t> ids) {
  std::vector<BlockRef> refs;
  for (std::uint32_t id : ids) {
    refs.push_back(BlockRef{/*tier=*/0, /*group=*/0, id, /*refcount=*/0});
  }
  return refs;
}

MatchOptions GetOpts() { return MatchOptions{/*lock_matched=*/false, true}; }
MatchOptions StageOpts() { return MatchOptions{false, false}; }

std::vector<std::uint32_t> BlockIds(const MatchResult& m) {
  std::vector<std::uint32_t> ids;
  for (const BlockRef& ref : m.block_refs) {
    ids.push_back(ref.block);
  }
  return ids;
}

// ---------------------------------------------------------------------------
// FlexKV golden sequence (flexkv/cache/radixtree.py __main__, tpb=2).
// ---------------------------------------------------------------------------

TEST(RadixIndexGoldenTest, FlexKVScenarioMatchLengthsAndEviction) {
  const ModelLayout layout = HomogeneousLayout(/*tpb=*/2);
  TickClock clock;
  RadixIndex index(layout, [&clock] { return clock(); });

  const std::vector<TokenId> seq1 = Tok({1, 2, 3, 4, 5, 6, 7, 8});
  const std::vector<TokenId> seq2 = Tok({1, 2, 3, 4, 15, 16, 17, 18});
  const std::vector<TokenId> seq3 = Tok({1, 2, 3, 4, 0, 0});

  CacheEntry* entry1 = index.Insert(seq1, Refs1G({0, 1, 2, 3}), {});
  EXPECT_TRUE(entry1 != nullptr);
  index.CommitReady(entry1);
  EXPECT_EQ(index.TotalCachedBlocks(), 4u);

  // seq2 shares exactly two blocks with seq1 (FlexKV asserts == 2).
  MatchResult m2 = index.Match(seq2, 2, GetOpts());
  EXPECT_EQ(m2.matched_blocks, 2u);
  EXPECT_EQ(BlockIds(m2), (std::vector<std::uint32_t>{0, 1}));

  CacheEntry* entry2 = index.Insert(seq2, Refs1G({8, 9}), {});
  EXPECT_TRUE(entry2 != nullptr);
  index.CommitReady(entry2);
  EXPECT_EQ(index.TotalCachedBlocks(), 6u);

  // seq3 shares two blocks through the split parent node.
  MatchResult m3 = index.Match(seq3, 2, GetOpts());
  EXPECT_EQ(m3.matched_blocks, 2u);

  // Evict 3: the older seq1 tail leaf (blocks 2, 3) goes first whole;
  // its parent still has the seq2 child so it is NOT a candidate —
  // the seq2 leaf shrinks by one instead (block 9). This is the exact
  // FlexKV eviction trace (only childless parents are promoted).
  stratakv::metadata::Evicted evicted = index.Evict(3);
  std::vector<std::uint32_t> evicted_ids;
  for (const BlockRef& ref : evicted.blocks) {
    evicted_ids.push_back(ref.block);
  }
  EXPECT_EQ(evicted_ids, (std::vector<std::uint32_t>{2, 3, 9}));
  EXPECT_EQ(index.TotalCachedBlocks(), 3u);

  // seq1 still matches the intact shared prefix (2 blocks); seq2
  // matches 3 (shared prefix + the shrunken seq2 leaf's first block).
  EXPECT_EQ(index.Match(seq1, 2, GetOpts()).matched_blocks, 2u);
  EXPECT_EQ(index.Match(seq2, 2, GetOpts()).matched_blocks, 3u);

  evicted = index.Evict(10);
  EXPECT_EQ(evicted.blocks.size(), static_cast<std::size_t>(3));
  EXPECT_EQ(index.TotalCachedBlocks(), 0u);

  EXPECT_EQ(index.Match(seq1, 2, GetOpts()).matched_blocks, 0u);
  EXPECT_EQ(index.Match(seq2, 2, GetOpts()).matched_blocks, 0u);
}

// ---------------------------------------------------------------------------
// Core semantics.
// ---------------------------------------------------------------------------

TEST(RadixIndexTest, ColdMissIsEmpty) {
  const ModelLayout layout = HomogeneousLayout(2);
  RadixIndex index(layout);
  const MatchResult m = index.Match(Tok({5, 6, 7, 8}), 2, GetOpts());
  EXPECT_EQ(m.matched_blocks, 0u);
  EXPECT_EQ(m.block_refs.size(), static_cast<std::size_t>(0));
  EXPECT_EQ(m.token_mask, (std::vector<std::uint8_t>{0, 0, 0, 0}));
}

TEST(RadixIndexTest, MidNodePartialMatchBinarySearch) {
  const ModelLayout layout = HomogeneousLayout(2);
  RadixIndex index(layout);
  CacheEntry* entry = index.Insert(Tok({1, 2, 3, 4, 5, 6, 7, 8}),
                                   Refs1G({0, 1, 2, 3}), {});
  index.CommitReady(entry);

  // Shares three of four blocks; the fourth diverges mid-node.
  const MatchResult m = index.Match(Tok({1, 2, 3, 4, 5, 6, 99, 100}), 2,
                                    GetOpts());
  EXPECT_EQ(m.matched_blocks, 3u);
  EXPECT_EQ(BlockIds(m), (std::vector<std::uint32_t>{0, 1, 2}));
  EXPECT_EQ(m.token_mask, (std::vector<std::uint8_t>{1, 1, 1, 1, 1, 1, 0, 0}));
}

TEST(RadixIndexTest, ReadinessFiltersUntilCommit) {
  const ModelLayout layout = HomogeneousLayout(2);
  RadixIndex index(layout);
  CacheEntry* entry =
      index.Insert(Tok({1, 2, 3, 4}), Refs1G({7, 8}), {});
  EXPECT_TRUE(entry != nullptr);
  EXPECT_FALSE(entry->is_ready);

  const MatchResult staged = index.Match(Tok({1, 2, 3, 4}), 2, GetOpts());
  EXPECT_EQ(staged.matched_blocks, 0u);
  EXPECT_EQ(staged.matched_blocks_reserved, 2u);

  const MatchResult raw = index.Match(Tok({1, 2, 3, 4}), 2, StageOpts());
  EXPECT_EQ(raw.matched_blocks, 2u);  // Staging mode ignores readiness.

  index.CommitReady(entry);
  EXPECT_TRUE(entry->is_ready);
  const MatchResult ready = index.Match(Tok({1, 2, 3, 4}), 2, GetOpts());
  EXPECT_EQ(ready.matched_blocks, 2u);
  EXPECT_EQ(ready.matched_blocks_reserved, 0u);
  EXPECT_EQ(index.TotalReadyBlocks(), 2u);
}

TEST(RadixIndexTest, ConservativeCommitDoesNotOverMarkAncestors) {
  // Extending put lands while the base put is still in flight: the base
  // node must NOT become ready (FlexKV's ready_length ancestor marking
  // would over-mark it — see index.h).
  const ModelLayout layout = HomogeneousLayout(2);
  RadixIndex index(layout);
  CacheEntry* base = index.Insert(Tok({1, 2, 3, 4}), Refs1G({0, 1}), {});
  CacheEntry* ext =
      index.Insert(Tok({1, 2, 3, 4, 5, 6}), Refs1G({2}), {});
  index.CommitReady(ext);

  const MatchResult m = index.Match(Tok({1, 2, 3, 4, 5, 6}), 2, GetOpts());
  EXPECT_EQ(m.matched_blocks, 0u);  // Base blocks still unready.
  EXPECT_EQ(m.matched_blocks_reserved, 3u);

  index.CommitReady(base);
  const MatchResult done = index.Match(Tok({1, 2, 3, 4, 5, 6}), 2, GetOpts());
  EXPECT_EQ(done.matched_blocks, 3u);
}

TEST(RadixIndexTest, CommitReadyUnknownEntryThrows) {
  const ModelLayout layout = HomogeneousLayout(2);
  RadixIndex index(layout);
  stratakv::common::CacheEntry bogus;
  EXPECT_THROW(index.CommitReady(&bogus), stratakv::LogicError);
}

TEST(RadixIndexTest, InsertFullyPresentReturnsNull) {
  const ModelLayout layout = HomogeneousLayout(2);
  RadixIndex index(layout);
  CacheEntry* entry = index.Insert(Tok({1, 2, 3, 4}), Refs1G({0, 1}), {});
  EXPECT_TRUE(index.Insert(Tok({1, 2, 3, 4}), Refs1G({}), {}) == nullptr);
  EXPECT_TRUE(entry != nullptr);
}

TEST(RadixIndexTest, InsertValidatesRefShape) {
  const ModelLayout layout = HomogeneousLayout(2);
  RadixIndex index(layout);
  EXPECT_THROW(index.Insert(Tok({1, 2, 3, 4}), Refs1G({0}), {}),
               std::invalid_argument);
  EXPECT_THROW(index.Insert(Tok({1, 2, 3, 4}),
                            {BlockRef{0, /*group=*/5, 0, 0},
                             BlockRef{0, /*group=*/5, 1, 0}},
                            {}),
               std::invalid_argument);
}

TEST(RadixIndexTest, PinsBlockEvictionUntilRelease) {
  const ModelLayout layout = HomogeneousLayout(2);
  TickClock clock;
  RadixIndex index(layout, [&clock] { return clock(); });
  CacheEntry* entry =
      index.Insert(Tok({1, 2, 3, 4}), Refs1G({0, 1}), {});
  index.CommitReady(entry);

  MatchOptions lock_opts{/*lock_matched=*/true, /*require_ready=*/true};
  MatchResult pinned = index.Match(Tok({1, 2, 3, 4}), 2, lock_opts);
  EXPECT_EQ(pinned.matched_blocks, 2u);
  EXPECT_EQ(pinned.block_refs[0].refcount, 1u);  // Pin visible in copies.

  stratakv::metadata::Evicted none = index.Evict(10);
  EXPECT_EQ(none.blocks.size(), static_cast<std::size_t>(0));

  index.Release(pinned);
  index.Release(pinned);  // Idempotent.

  stratakv::metadata::Evicted all = index.Evict(10);
  EXPECT_EQ(all.blocks.size(), static_cast<std::size_t>(2));
  EXPECT_TRUE(all.entries.size() == static_cast<std::size_t>(1));
  EXPECT_EQ(index.TotalCachedBlocks(), 0u);
}

TEST(RadixIndexTest, SplitPreservesAllBranches) {
  const ModelLayout layout = HomogeneousLayout(2);
  RadixIndex index(layout);
  const std::vector<TokenId> a = Tok({1, 2, 3, 4, 5, 6, 7, 8});
  const std::vector<TokenId> b = Tok({1, 2, 3, 4, 5, 6, 7, 99});
  const std::vector<TokenId> c = Tok({1, 2, 30, 40});

  CacheEntry* ea = index.Insert(a, Refs1G({0, 1, 2, 3}), {});
  index.CommitReady(ea);
  // b diverges at the last block: splits a's node 3/1.
  CacheEntry* eb = index.Insert(b, Refs1G({4}), {});
  index.CommitReady(eb);

  // Before c is cached it matches only the shared first block.
  EXPECT_EQ(index.Match(c, 2, GetOpts()).matched_blocks, 1u);

  // c diverges one block earlier: splits the 3-block prefix node 1/2.
  CacheEntry* ec = index.Insert(c, Refs1G({5}), {});
  index.CommitReady(ec);

  EXPECT_EQ(index.Match(a, 2, GetOpts()).matched_blocks, 4u);
  EXPECT_EQ(index.Match(b, 2, GetOpts()).matched_blocks, 4u);
  EXPECT_EQ(index.Match(c, 2, GetOpts()).matched_blocks, 2u);
  EXPECT_EQ(index.TotalCachedBlocks(), 6u);
}

// ---------------------------------------------------------------------------
// Hybrid (ADR 0003): bundle truncation and state-boundary eviction.
// ---------------------------------------------------------------------------

TEST(RadixIndexHybridTest, MatchTruncatesToCompleteBundleBoundary) {
  const ModelLayout layout = HybridLayout(2);
  RadixIndex index(layout);
  const std::vector<TokenId> a = Tok({1, 2, 3, 4, 5, 6, 7, 8});   // 4 blocks
  const std::vector<TokenId> b = Tok({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});

  CacheEntry* ea =
      index.Insert(a, Refs1G({0, 1, 2, 3}),
                   {StateRef{0, /*group=*/1, /*object=*/0, 0}});
  index.CommitReady(ea);
  CacheEntry* eb =
      index.Insert(b, Refs1G({4, 5}), {StateRef{0, 1, 1, 0}});
  index.CommitReady(eb);

  // Full match lands on the 6-block boundary with its state bundle.
  const MatchResult full = index.Match(b, 2, GetOpts());
  EXPECT_EQ(full.matched_blocks, 6u);
  EXPECT_EQ(full.state_refs.size(), static_cast<std::size_t>(1));
  EXPECT_EQ(full.state_refs[0].object, 1u);

  // Sharing 5 blocks must truncate to the LAST COMPLETE boundary (4):
  // serving block 4 would resume mid-state-interval with stale states —
  // the exact wrong-attention failure ADR 0003 forbids.
  const std::vector<TokenId> query5 = Tok({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  const MatchResult partial = index.Match(query5, 2, GetOpts());
  EXPECT_EQ(partial.matched_blocks, 4u);
  EXPECT_EQ(partial.matched_blocks_reserved, 1u);
  EXPECT_EQ(partial.state_refs[0].object, 0u);

  // Staging mode ignores the truncation (puts extend raw prefixes).
  const MatchResult raw = index.Match(query5, 2, StageOpts());
  EXPECT_EQ(raw.matched_blocks, 5u);
}

TEST(RadixIndexHybridTest, UnreadyBundleIsNotServable) {
  const ModelLayout layout = HybridLayout(2);
  RadixIndex index(layout);
  CacheEntry* entry = index.Insert(
      Tok({1, 2, 3, 4}), Refs1G({0, 1}), {StateRef{0, 1, 0, 0}});
  EXPECT_FALSE(entry->is_ready);

  const MatchResult m = index.Match(Tok({1, 2, 3, 4}), 2, GetOpts());
  EXPECT_EQ(m.matched_blocks, 0u);
  EXPECT_EQ(m.matched_blocks_reserved, 2u);

  index.CommitReady(entry);
  EXPECT_TRUE(entry->IsCompleteFor(layout));
  const MatchResult ready = index.Match(Tok({1, 2, 3, 4}), 2, GetOpts());
  EXPECT_EQ(ready.matched_blocks, 2u);
}

TEST(RadixIndexHybridTest, HybridInsertRequiresStates) {
  const ModelLayout layout = HybridLayout(2);
  RadixIndex index(layout);
  // Homogeneous-style insert (no states) must be rejected outright.
  EXPECT_THROW(index.Insert(Tok({1, 2, 3, 4}), Refs1G({0, 1}), {}),
               std::invalid_argument);
}

TEST(RadixIndexHybridTest, StateNodesEvictWholeNeverShrink) {
  const ModelLayout layout = HybridLayout(2);
  TickClock clock;
  RadixIndex index(layout, [&clock] { return clock(); });
  CacheEntry* entry = index.Insert(
      Tok({1, 2, 3, 4, 5, 6, 7, 8}), Refs1G({0, 1, 2, 3}),
      {StateRef{0, 1, /*object=*/7, 0}});
  index.CommitReady(entry);

  // Ask for 1 block; the leaf carries boundary states, so it is evicted
  // whole (4 blocks + 1 state) rather than partially shrunk.
  stratakv::metadata::Evicted evicted = index.Evict(1);
  EXPECT_EQ(evicted.blocks.size(), static_cast<std::size_t>(4));
  EXPECT_EQ(evicted.states.size(), static_cast<std::size_t>(1));
  EXPECT_EQ(evicted.states[0].object, 7u);
  EXPECT_EQ(evicted.entries.size(), static_cast<std::size_t>(1));
  EXPECT_EQ(index.TotalCachedBlocks(), 0u);
}

TEST(RadixIndexTest, GraceTimeProtectsYoungLeaves) {
  const ModelLayout layout = HomogeneousLayout(2);
  TickClock clock;
  RadixIndex index(layout, [&clock] { return clock(); },
                   /*eviction_grace=*/std::chrono::milliseconds(1000));
  CacheEntry* entry = index.Insert(Tok({1, 2, 3, 4}), Refs1G({0, 1}), {});
  index.CommitReady(entry);

  EXPECT_EQ(index.Evict(10).blocks.size(), static_cast<std::size_t>(0));

  clock.tick += 2'000'000'000ULL;  // Age the leaf past the grace window.
  EXPECT_EQ(index.Evict(10).blocks.size(), static_cast<std::size_t>(2));
}

TEST(RadixIndexTest, ResetClearsEverything) {
  const ModelLayout layout = HomogeneousLayout(2);
  RadixIndex index(layout);
  CacheEntry* entry = index.Insert(Tok({1, 2, 3, 4}), Refs1G({0, 1}), {});
  index.CommitReady(entry);
  index.Reset();
  EXPECT_EQ(index.TotalCachedBlocks(), 0u);
  EXPECT_EQ(index.TotalNodes(), static_cast<std::size_t>(0));
  EXPECT_EQ(index.Match(Tok({1, 2, 3, 4}), 2, GetOpts()).matched_blocks, 0u);
}

TEST(RadixIndexTest, LayoutInvariantsEnforced) {
  EXPECT_THROW(RadixIndex(HybridLayout(0)), std::invalid_argument);
  // Non-uniform tokens_per_block across token-addressable groups.
  LayerGroup g0;
  g0.name = "a";
  g0.spec = FullAttnSpec{2, 64};
  g0.layer_indices = {0};
  g0.tokens_per_block = 16;
  LayerGroup g1;
  g1.name = "b";
  g1.spec = FullAttnSpec{2, 64};
  g1.layer_indices = {1};
  g1.tokens_per_block = 32;
  const ModelLayout mixed = ModelLayout::FromGroups({g0, g1});
  EXPECT_THROW((RadixIndex{mixed}), std::invalid_argument);
  // Pure LinearState layouts cannot use a token radix index.
  LayerGroup lin;
  lin.name = "lin";
  lin.spec = LinearStateSpec{1024};
  lin.layer_indices = {0};
  lin.tokens_per_block = 0;
  const ModelLayout pure_lin = ModelLayout::FromGroups({lin});
  EXPECT_THROW((RadixIndex{pure_lin}), std::invalid_argument);
}

}  // namespace
