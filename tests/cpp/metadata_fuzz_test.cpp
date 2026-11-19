// Phase C (C5): randomized interleavings of insert/commit/match/release/
// evict against an independent oracle model.
//
// Mode A (no eviction) checks EXACT match semantics: the oracle tracks
// every staged entry (token span, readiness, hybrid bundle boundary) and
// predicts matched/servable/reserved for every query.
// Mode B adds eviction; exact prediction would require replicating the
// LRU leaf walk and partial-shrink positions, so it checks structural
// invariants instead: allocator conservation (allocators throw loudly on
// double-recycle; end-of-run pools must be whole again), servable <=
// raw, token-mask prefix consistency, and monotonicity. Memory safety
// itself is covered by the ASan/UBSan CI job.
//
// The roadmap's Python oracle lands with the Phase D bindings (a Python
// reference implementation can then drive the identical op stream); the
// C++ oracle here is deliberately independent of the radix code paths:
// flat vectors and linear scans, no trees, no hashes.
//
// Iterations default to 400 per mode (bounded CI time); set
// STRATAKV_FUZZ_ITERS to run longer locally, e.g. 100000.

#ifdef STRATAKV_GTEST_SHIM
#include "gtest_shim.h"
#else
#include <gtest/gtest.h>
#endif

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <utility>
#include <vector>

#include "stratakv/common/exceptions.h"
#include "stratakv/common/handles.h"
#include "stratakv/common/model_layout.h"
#include "stratakv/metadata/index.h"
#include "stratakv/metadata/metadata_plane.h"

namespace {

using stratakv::common::CacheEntry;
using stratakv::common::FullAttnSpec;
using stratakv::common::LayerGroup;
using stratakv::common::LinearStateSpec;
using stratakv::common::ModelLayout;
using stratakv::common::TokenId;
using stratakv::metadata::Evicted;
using stratakv::metadata::MatchOptions;
using stratakv::metadata::MatchResult;
using stratakv::metadata::MetadataPlane;
using stratakv::metadata::MetadataPlaneConfig;
using stratakv::metadata::RadixIndex;

// --- Independent oracle ---------------------------------------------------

struct OracleEntry {
  std::vector<TokenId> tokens;   // Full prefix at insert time.
  std::uint32_t matched_blocks;  // Blocks already present then.
  bool ready = false;
  bool dead = false;             // Evicted (Mode B).
};

std::uint32_t CommonBlocks(const std::vector<TokenId>& stored,
                           const std::vector<TokenId>& query,
                           std::uint32_t tpb) {
  const std::uint32_t max_blocks = static_cast<std::uint32_t>(
      std::min(stored.size(), query.size()) / tpb);
  std::uint32_t common = 0;
  while (common < max_blocks) {
    bool equal = true;
    for (std::uint32_t t = 0; t < tpb; ++t) {
      if (stored[common * tpb + t] != query[common * tpb + t]) {
        equal = false;
        break;
      }
    }
    if (!equal) break;
    ++common;
  }
  return common;
}

struct Oracle {
  std::uint32_t tpb;
  bool hybrid;
  std::vector<OracleEntry> entries;

  std::uint32_t RawMatch(const std::vector<TokenId>& q) const {
    std::uint32_t best = 0;
    for (const OracleEntry& e : entries) {
      if (e.dead) continue;
      best = std::max(best, CommonBlocks(e.tokens, q, tpb));
    }
    return best;
  }

  // Contiguous ready prefix, QUERY-RELATIVE: the owner of block r on
  // q's path is the unique entry whose staged span covers r and whose
  // tokens agree with q through r (ownership is a tree, not a linear
  // partition — at most one entry can satisfy both for a given r).
  std::uint32_t ReadyPrefix(const std::vector<TokenId>& q,
                            std::uint32_t raw) const {
    std::uint32_t r = 0;
    while (r < raw) {
      const OracleEntry* owner = nullptr;
      for (const OracleEntry& e : entries) {
        if (e.dead) continue;
        const std::uint32_t len = static_cast<std::uint32_t>(
            e.tokens.size() / tpb);
        if (e.matched_blocks <= r && r < len &&
            CommonBlocks(e.tokens, q, tpb) > r) {
          owner = &e;
          break;
        }
      }
      if (owner == nullptr || !owner->ready) break;
      ++r;
    }
    return r;
  }

  // Hybrid: latest ready entry end at or below `cap` whose end lies ON
  // q's path (q agrees with the entry through its whole length; entry
  // ends from other branches are not boundaries on this path).
  std::uint32_t BundleBoundary(const std::vector<TokenId>& q,
                               std::uint32_t cap) const {
    if (!hybrid) return cap;
    std::uint32_t boundary = 0;
    for (const OracleEntry& e : entries) {
      if (e.dead || !e.ready) continue;
      const std::uint32_t len = static_cast<std::uint32_t>(
          e.tokens.size() / tpb);
      if (len <= cap && CommonBlocks(e.tokens, q, tpb) >= len) {
        boundary = std::max(boundary, len);
      }
    }
    return boundary;
  }
};

// --- Shared fuzz driver ---------------------------------------------------

struct TickClock {
  std::uint64_t tick = 0;
  std::uint64_t operator()() { return ++tick; }
};

ModelLayout MakeLayout(bool hybrid, std::uint32_t tpb) {
  LayerGroup full;
  full.name = "full_attn";
  full.spec = FullAttnSpec{2, 64};
  full.layer_indices = hybrid ? std::vector<int>{0, 3}
                              : std::vector<int>{0, 1, 2, 3};
  full.tokens_per_block = static_cast<int>(tpb);
  std::vector<LayerGroup> groups{full};
  if (hybrid) {
    LayerGroup linear;
    linear.name = "linear_state";
    linear.spec = LinearStateSpec{8192};
    linear.layer_indices = {1, 2};
    linear.tokens_per_block = 0;
    groups.push_back(linear);
  }
  return ModelLayout::FromGroups(groups);
}

std::uint32_t QueryIters() {
  const char* env = std::getenv("STRATAKV_FUZZ_ITERS");
  if (env == nullptr) return 400;
  const long parsed = std::strtol(env, nullptr, 10);
  return parsed > 0 ? static_cast<std::uint32_t>(parsed) : 400;
}

std::vector<TokenId> RandomTokens(std::mt19937_64* rng, std::uint32_t tpb,
                                   bool extend_existing,
                                   const Oracle& oracle) {
  std::uniform_int_distribution<int> len_dist(1, 8);
  std::uniform_int_distribution<int> tok_dist(1, 5);  // Small alphabet.
  std::vector<TokenId> tokens;
  if (extend_existing) {
    std::vector<const OracleEntry*> live;
    for (const OracleEntry& e : oracle.entries) {
      if (!e.dead) live.push_back(&e);
    }
    if (!live.empty()) {
      std::uniform_int_distribution<std::size_t> pick(0, live.size() - 1);
      const std::vector<TokenId>& base = live[pick(*rng)]->tokens;
      std::uniform_int_distribution<std::size_t> cut(1, base.size());
      tokens.assign(base.begin(),
                    base.begin() + static_cast<std::ptrdiff_t>(cut(*rng)));
    }
  }
  const std::uint32_t target_blocks =
      static_cast<std::uint32_t>(len_dist(*rng));
  while (tokens.size() / tpb < target_blocks) {
    tokens.push_back(tok_dist(*rng));
  }
  return tokens;
}

void RunFuzz(bool hybrid, bool with_evict) {
  constexpr std::uint32_t kTpb = 2;
  const ModelLayout layout = MakeLayout(hybrid, kTpb);
  MetadataPlaneConfig config;
  config.blocks_per_group = 64;
  config.state_objects_per_group = 16;
  config.state_object_bytes = 4096;
  TickClock clock;
  MetadataPlane plane(layout, std::make_unique<RadixIndex>(
                                  layout, [&clock] { return clock(); }),
                      config);
  auto* index = dynamic_cast<RadixIndex*>(&plane.index());
  ASSERT_TRUE(index != nullptr);

  Oracle oracle{kTpb, hybrid, {}};
  // Live entry handle -> oracle index (staged and committed both).
  std::vector<std::pair<const CacheEntry*, std::size_t>> live;
  std::vector<std::size_t> pending;  // Oracle indices awaiting commit.
  std::mt19937_64 rng(0x5354524154524131ULL);  // Fixed: reproducible.

  std::uint32_t matches_checked = 0;
  std::uint32_t staged = 0;

  auto verify_exact = [&](const std::vector<TokenId>& q) {
    const MatchResult m = plane.MatchForGet(q);
    const std::uint32_t raw = oracle.RawMatch(q);
    const std::uint32_t expect = std::min(
        oracle.ReadyPrefix(q, raw), oracle.BundleBoundary(q, raw));
#ifdef STRATAKV_FUZZ_DEBUG
    if (m.matched_blocks != expect) {
      std::fprintf(stderr, "MISMATCH plane=%u oracle=%u raw=%u q=[",
                   m.matched_blocks, expect, raw);
      for (TokenId t : q) std::fprintf(stderr, "%lld,", static_cast<long long>(t));
      std::fprintf(stderr, "]\n");
      for (std::size_t i = 0; i < oracle.entries.size(); ++i) {
        const OracleEntry& e = oracle.entries[i];
        std::fprintf(stderr, "  e%zu m=%u ready=%d toks=[", i,
                     e.matched_blocks, e.ready ? 1 : 0);
        for (TokenId t : e.tokens) std::fprintf(stderr, "%lld,", static_cast<long long>(t));
        std::fprintf(stderr, "]\n");
      }
    }
#endif
    ASSERT_EQ(m.matched_blocks, expect);
    ASSERT_EQ(m.matched_blocks + m.matched_blocks_reserved, raw);
    ASSERT_EQ(m.block_refs.size(), static_cast<std::size_t>(expect));
    ASSERT_EQ(m.token_mask.size(), q.size());
    for (std::size_t t = 0; t < q.size(); ++t) {
      ASSERT_EQ(m.token_mask[t],
                t < static_cast<std::size_t>(expect) * kTpb ? 1 : 0);
    }
    plane.Release(m);
    ++matches_checked;
  };

  auto verify_bounded = [&](const std::vector<TokenId>& q) {
    const MatchResult m = plane.MatchForGet(q);
    const MatchOptions stage{/*lock_matched=*/false,
                             /*require_ready=*/false};
    const MatchResult raw_m = index->Match(q, kTpb, stage);
    // Internal consistency: servable is a prefix of raw, and the
    // remainder is reported as reserved. (No oracle bound on raw:
    // eviction can leave a dead entry's blocks alive inside split
    // parents while dropping the entry, and partial shrinks remove
    // block tails — the oracle cannot model either positionally.)
    ASSERT_TRUE(m.matched_blocks <= raw_m.matched_blocks);
    ASSERT_EQ(m.matched_blocks + m.matched_blocks_reserved,
              raw_m.matched_blocks);
    ASSERT_EQ(m.block_refs.size(),
              static_cast<std::size_t>(m.matched_blocks));
    ASSERT_EQ(m.token_mask.size(), q.size());
    for (std::size_t t = 0; t < q.size(); ++t) {
      ASSERT_EQ(m.token_mask[t],
                t < static_cast<std::size_t>(m.matched_blocks) * kTpb
                    ? 1 : 0);
    }
    plane.Release(m);  // Must never throw (underflow guard).
    ++matches_checked;
  };

  auto drop_dead = [&](const CacheEntry* dead) {
    for (auto it = live.begin(); it != live.end();) {
      if (it->first == dead) {
        oracle.entries[it->second].dead = true;
        it = live.erase(it);
      } else {
        ++it;
      }
    }
    for (auto it = pending.begin(); it != pending.end();) {
      if (oracle.entries[*it].dead) {
        it = pending.erase(it);
      } else {
        ++it;
      }
    }
  };

  auto live_handle = [&](std::size_t idx) -> const CacheEntry* {
    for (const auto& l : live) {
      if (l.second == idx) return l.first;
    }
    return nullptr;
  };

  const std::uint32_t iters = QueryIters();
  for (std::uint32_t op = 0; op < iters; ++op) {
    const int action = std::uniform_int_distribution<int>(0, 99)(rng);
    if (action < 40) {
      const std::vector<TokenId> tokens =
          RandomTokens(&rng, kTpb, /*extend_existing=*/op % 3 == 0, oracle);
      try {
        const MetadataPlane::PutPlan plan = plane.StagePut(tokens);
        if (plan.entry != nullptr) {
          oracle.entries.push_back(
              OracleEntry{tokens, plan.matched_blocks, false, false});
          live.emplace_back(plan.entry, oracle.entries.size() - 1);
          pending.push_back(oracle.entries.size() - 1);
          ++staged;
        }
      } catch (const stratakv::NotEnoughSpaceError&) {
        // Exhaustion is acceptable; StagePut reserves nothing on throw.
      }
    } else if (action < 60) {
      if (!pending.empty()) {
        const std::size_t idx = pending.front();
        pending.erase(pending.begin());
        const CacheEntry* entry = live_handle(idx);
        if (!oracle.entries[idx].dead && entry != nullptr) {
          plane.CommitPut(const_cast<CacheEntry*>(entry));
          oracle.entries[idx].ready = true;
        }
      }
    } else if (action < 90) {
      const std::vector<TokenId> q =
          RandomTokens(&rng, kTpb, true, oracle);
      if (with_evict) {
        verify_bounded(q);
      } else {
        verify_exact(q);
      }
    } else if (with_evict) {
      std::uniform_int_distribution<std::uint32_t> k(1, 6);
      const Evicted evicted = plane.EvictBlocks(k(rng));
      for (const CacheEntry* dead : evicted.entries) {
        drop_dead(dead);
      }
      verify_bounded(RandomTokens(&rng, kTpb, true, oracle));
    } else {
      verify_exact(RandomTokens(&rng, kTpb, false, oracle));
    }
  }

  // Conservation: commit whatever is left, evict everything evictable,
  // then every pool must be whole again (no leaked ids; double recycles
  // would have thrown inside EvictBlocks already).
  for (std::size_t idx : pending) {
    const CacheEntry* entry = live_handle(idx);
    if (oracle.entries[idx].dead || entry == nullptr) continue;
    try {
      plane.CommitPut(const_cast<CacheEntry*>(entry));
    } catch (const stratakv::LogicError&) {
    }
  }
  for (int i = 0; i < 10000; ++i) {
    const Evicted evicted = plane.EvictBlocks(1u << 30);
    if (evicted.blocks.empty() && evicted.states.empty()) break;
    for (const CacheEntry* dead : evicted.entries) {
      drop_dead(dead);
    }
  }
  EXPECT_EQ(index->TotalCachedBlocks(), 0u);
  EXPECT_EQ(plane.num_free_blocks(0), config.blocks_per_group);
  if (hybrid) {
    EXPECT_EQ(plane.num_free_states(1), config.state_objects_per_group);
  }
  EXPECT_TRUE(staged > 0);
  EXPECT_TRUE(matches_checked > 0);
}

}  // namespace

TEST(MetadataFuzzTest, ExactOracleNoEvict_Homogeneous) {
  RunFuzz(/*hybrid=*/false, /*with_evict=*/false);
}

TEST(MetadataFuzzTest, ExactOracleNoEvict_Hybrid) {
  RunFuzz(/*hybrid=*/true, /*with_evict=*/false);
}

TEST(MetadataFuzzTest, InvariantsWithEvict_Homogeneous) {
  RunFuzz(/*hybrid=*/false, /*with_evict=*/true);
}

TEST(MetadataFuzzTest, InvariantsWithEvict_Hybrid) {
  RunFuzz(/*hybrid=*/true, /*with_evict=*/true);
}
