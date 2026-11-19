// Phase C (C4): index match-latency benchmark.
//
// Scenarios (Qwen3-Next-shaped hybrid layout: full_attn + linear_state,
// tokens_per_block = 16):
//   deep-tree warm hit   — 1024-block query, interleaved-insert tree
//   deep-tree miss       — 1024-block query sharing nothing
//   single-node warm hit — 1024 blocks in one node (fresh-prefix shape)
//   partial hit          — 512 of 1024 blocks
// Reports p50/p95/p99 and checks the roadmap budget (p50 < 100 µs at
// 1k blocks) on the warm-hit paths; prints PASS/FAIL for the budget —
// deliberately NOT a hard assertion (CI runners are too noisy for
// latency gates; regressions are read from this table, per the
// benchmark policy in roadmap.md).

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <random>
#include <vector>

#include "stratakv/common/model_layout.h"
#include "stratakv/metadata/index.h"
#include "stratakv/metadata/metadata_plane.h"

namespace {

using stratakv::common::FullAttnSpec;
using stratakv::common::LayerGroup;
using stratakv::common::LinearStateSpec;
using stratakv::common::ModelLayout;
using stratakv::common::TokenId;
using stratakv::metadata::MatchOptions;
using stratakv::metadata::MatchResult;
using stratakv::metadata::MetadataPlane;
using stratakv::metadata::MetadataPlaneConfig;
using stratakv::metadata::RadixIndex;

constexpr std::uint32_t kTpb = 16;
constexpr std::uint32_t kBlocks = 1024;

ModelLayout Qwen3NextShaped() {
  LayerGroup full;
  full.name = "full_attn";
  full.spec = FullAttnSpec{4, 128};
  full.layer_indices = {0, 4, 8, 12};
  full.tokens_per_block = static_cast<int>(kTpb);
  LayerGroup linear;
  linear.name = "linear_state";
  linear.spec = LinearStateSpec{65536};
  linear.layer_indices = {1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15};
  linear.tokens_per_block = 0;
  return ModelLayout::FromGroups({full, linear});
}

std::vector<TokenId> Sequence(std::uint32_t blocks, std::uint64_t salt) {
  std::mt19937_64 rng(salt);
  std::vector<TokenId> tokens;
  tokens.reserve(static_cast<std::size_t>(blocks) * kTpb);
  for (std::uint32_t b = 0; b < blocks; ++b) {
    for (std::uint32_t t = 0; t < kTpb; ++t) {
      tokens.push_back(static_cast<TokenId>(rng() % 100000));
    }
  }
  return tokens;
}

double Percentile(std::vector<double> samples, double p) {
  std::sort(samples.begin(), samples.end());
  const std::size_t idx = static_cast<std::size_t>(
      p * static_cast<double>(samples.size() - 1));
  return samples[idx];
}

struct Sample {
  const char* name;
  double p50_us;
  double p95_us;
  double p99_us;
  std::uint32_t matched;
};

Sample BenchMatch(MetadataPlane* plane, const char* name,
                  const std::vector<TokenId>& query, int iterations) {
  std::vector<double> micros;
  micros.reserve(static_cast<std::size_t>(iterations));
  std::uint32_t matched = 0;
  for (int i = 0; i < iterations; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    MatchResult m = plane->MatchForGet(query);
    const auto t1 = std::chrono::steady_clock::now();
    matched = m.matched_blocks;
    plane->Release(m);
    micros.push_back(
        std::chrono::duration<double, std::micro>(t1 - t0).count());
  }
  return Sample{name, Percentile(micros, 0.50), Percentile(micros, 0.95),
                Percentile(micros, 0.99), matched};
}

}  // namespace

int main() {
  const ModelLayout layout = Qwen3NextShaped();
  MetadataPlaneConfig config;
  config.blocks_per_group = 64 * kBlocks;
  config.state_objects_per_group = 4 * kBlocks;
  MetadataPlane plane(layout, std::make_unique<RadixIndex>(layout), config);

  // Deep tree: 256 interleaved requests sharing growing prefixes, each
  // ending on its own bundle boundary (the realistic hybrid shape).
  for (int r = 0; r < 256; ++r) {
    const std::uint32_t blocks =
        4 + static_cast<std::uint32_t>(r) % (kBlocks / 8);
    const std::vector<TokenId> tokens = Sequence(blocks, 1000 + r);
    auto plan = plane.StagePut(tokens);
    if (plan.entry != nullptr) {
      plane.CommitPut(plan.entry);
    }
  }
  // One full-depth entry to pin the 1k-block prefix, and a 512-block
  // prefix entry so a state bundle boundary exists at 512 (without it a
  // partial hybrid match would correctly truncate to 0 — ADR 0003).
  {
    const std::vector<TokenId> half = Sequence(kBlocks / 2, 7);
    auto plan = plane.StagePut(half);
    if (plan.entry != nullptr) {
      plane.CommitPut(plan.entry);
    }
    const std::vector<TokenId> deep = Sequence(kBlocks, 7);
    auto deep_plan = plane.StagePut(deep);
    if (deep_plan.entry != nullptr) {
      plane.CommitPut(deep_plan.entry);
    }
  }

  const int kIterations = 2000;
  std::vector<Sample> samples;
  samples.push_back(BenchMatch(&plane, "deep_warm_hit_1k",
                               Sequence(kBlocks, 7), kIterations));
  samples.push_back(BenchMatch(&plane, "deep_miss_1k",
                               Sequence(kBlocks, 999999), kIterations));
  std::vector<TokenId> partial = Sequence(512, 7);
  {
    const std::vector<TokenId> tail = Sequence(512, 424242);
    partial.insert(partial.end(), tail.begin(), tail.end());
  }
  samples.push_back(
      BenchMatch(&plane, "deep_partial_512_of_1k", partial, kIterations));

  // Single-node shape: fresh index, one 1k-block prefix.
  {
    MetadataPlane single(layout, std::make_unique<RadixIndex>(layout),
                         config);
    const std::vector<TokenId> tokens = Sequence(kBlocks, 42);
    auto plan = single.StagePut(tokens);
    if (plan.entry != nullptr) {
      single.CommitPut(plan.entry);
    }
    samples.push_back(BenchMatch(&single, "single_node_warm_hit_1k",
                                 tokens, kIterations));
  }

  std::printf("%-24s %10s %10s %10s %8s\n", "scenario", "p50_us", "p95_us",
              "p99_us", "matched");
  bool budget_pass = true;
  for (const Sample& s : samples) {
    std::printf("%-24s %10.1f %10.1f %10.1f %8u\n", s.name, s.p50_us,
                s.p95_us, s.p99_us, s.matched);
    if (std::string(s.name).find("warm_hit") != std::string::npos &&
        s.p50_us >= 100.0) {
      budget_pass = false;
    }
  }
  std::printf("budget p50 < 100us on warm-hit paths: %s\n",
              budget_pass ? "PASS" : "FAIL");
  return 0;
}
