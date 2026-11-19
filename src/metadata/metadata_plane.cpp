#include "stratakv/metadata/metadata_plane.h"

#include <algorithm>
#include <map>
#include <stdexcept>
#include <utility>

#include "stratakv/common/exceptions.h"

namespace stratakv::metadata {

namespace {
constexpr common::TierId kCpuTier = 0;
}

MetadataPlane::MetadataPlane(const common::ModelLayout& layout,
                             std::unique_ptr<IIndex> index,
                             MetadataPlaneConfig config)
    : layout_(layout),
      index_(std::move(index)),
      config_(std::move(config)) {
  if (index_ == nullptr) {
    throw std::invalid_argument("MetadataPlane: index must not be null");
  }
  for (std::size_t g = 0; g < layout_.num_groups(); ++g) {
    const common::LayerGroup& group =
        layout_.Group(static_cast<int>(g));
    if (IsTokenAddressable(group.spec)) {
      token_groups_.push_back(static_cast<std::uint32_t>(g));
      block_allocators_.push_back(std::make_unique<BlockAllocator>(
          config_.blocks_per_group));
    } else {
      state_groups_.push_back(static_cast<std::uint32_t>(g));
      state_pools_.push_back(std::make_unique<StatePool>(
          config_.state_objects_per_group, config_.state_object_bytes));
    }
  }
  tokens_per_block_ = token_groups_.empty()
      ? 0
      : static_cast<std::uint32_t>(
            layout_.Group(static_cast<int>(token_groups_.front()))
                .tokens_per_block);
}

MetadataPlane::~MetadataPlane() = default;

std::uint32_t MetadataPlane::num_free_blocks(std::uint32_t group) const {
  const auto it = std::find(token_groups_.begin(), token_groups_.end(),
                            group);
  if (it == token_groups_.end()) {
    throw std::invalid_argument("MetadataPlane: group " +
                                std::to_string(group) +
                                " is not token-addressable");
  }
  return block_allocators_[static_cast<std::size_t>(
                               std::distance(token_groups_.begin(), it))]
      ->num_free();
}

std::uint32_t MetadataPlane::num_free_states(std::uint32_t group) const {
  const auto it = std::find(state_groups_.begin(), state_groups_.end(),
                            group);
  if (it == state_groups_.end()) {
    throw std::invalid_argument("MetadataPlane: group " +
                                std::to_string(group) +
                                " is not a LinearState group");
  }
  return state_pools_[static_cast<std::size_t>(
                          std::distance(state_groups_.begin(), it))]
      ->num_free();
}

MatchResult MetadataPlane::MatchForGet(
    std::span<const common::TokenId> tokens) {
  common::ScopedLatency timing(match_latency_);
  MatchOptions opts;
  opts.lock_matched = true;
  opts.require_ready = true;  // ADR 0003: complete, landed bundles only.
  return index_->Match(tokens, tokens_per_block_, opts);
}

MetadataPlane::PutPlan MetadataPlane::PlanPut(
    std::span<const common::TokenId> tokens) {
  return PlanLocked(tokens, /*stage=*/false);
}

MetadataPlane::PutPlan MetadataPlane::StagePut(
    std::span<const common::TokenId> tokens) {
  return PlanLocked(tokens, /*stage=*/true);
}

MetadataPlane::PutPlan MetadataPlane::PlanLocked(
    std::span<const common::TokenId> tokens, bool stage) {
  MatchOptions opts;
  opts.lock_matched = false;
  opts.require_ready = false;  // Staging extends in-flight prefixes.
  const MatchResult match = index_->Match(tokens, tokens_per_block_, opts);

  PutPlan plan;
  plan.matched_blocks = match.matched_blocks;
  const std::uint32_t total_blocks = static_cast<std::uint32_t>(
      tokens.size() / tokens_per_block_);
  plan.new_blocks =
      total_blocks > plan.matched_blocks ? total_blocks - plan.matched_blocks
                                         : 0;

  // Capacity precheck across every group before reserving anything.
  bool feasible = true;
  for (std::uint32_t g = 0; g < token_groups_.size(); ++g) {
    if (block_allocators_[g]->num_free() < plan.new_blocks) {
      feasible = false;
    }
  }
  for (const auto& pool : state_pools_) {
    if (pool->num_free() == 0) {
      feasible = false;
    }
  }
  plan.feasible = feasible;
  if (!stage) {
    return plan;  // Read-only preview.
  }
  if (!feasible) {
    throw NotEnoughSpaceError(
        "MetadataPlane::StagePut: insufficient pool capacity for " +
        std::to_string(plan.new_blocks) + " new blocks");
  }
  if (plan.new_blocks == 0) {
    return plan;  // Fully present; nothing to stage (FlexKV parity).
  }

  const std::uint32_t num_token_groups =
      static_cast<std::uint32_t>(token_groups_.size());
  plan.blocks_to_store.reserve(
      static_cast<std::size_t>(plan.new_blocks) * num_token_groups);
  for (std::uint32_t g = 0; g < num_token_groups; ++g) {
    const std::vector<common::BlockId> ids =
        block_allocators_[g]->Allocate(plan.new_blocks);
    for (common::BlockId id : ids) {
      plan.blocks_to_store.push_back(
          common::BlockRef{kCpuTier, token_groups_[g], id, /*refcount=*/0});
    }
  }
  // Reorder position-major / group-minor (allocate returned group-major).
  std::vector<common::BlockRef> ordered;
  ordered.reserve(plan.blocks_to_store.size());
  for (std::uint32_t i = 0; i < plan.new_blocks; ++i) {
    for (std::uint32_t g = 0; g < num_token_groups; ++g) {
      ordered.push_back(
          plan.blocks_to_store[static_cast<std::size_t>(i) +
                               static_cast<std::size_t>(g) * plan.new_blocks]);
    }
  }
  plan.blocks_to_store = std::move(ordered);

  plan.states_to_snapshot.reserve(state_groups_.size());
  for (std::uint32_t s = 0; s < state_groups_.size(); ++s) {
    plan.states_to_snapshot.push_back(
        state_pools_[s]->Acquire(kCpuTier, state_groups_[s]));
  }

  plan.entry = index_->Insert(tokens, plan.blocks_to_store,
                              plan.states_to_snapshot);
  if (plan.entry == nullptr) {
    // Raced with an identical insert (single-scheduler protocol makes
    // this unreachable); still return the reserved ids so the caller can
    // release them.
    plan.new_blocks = 0;
    plan.blocks_to_store.clear();
    plan.states_to_snapshot.clear();
  }
  return plan;
}

void MetadataPlane::CommitPut(common::CacheEntry* entry) {
  index_->CommitReady(entry);
}

void MetadataPlane::Release(const MatchResult& match) {
  index_->Release(match);
}

Evicted MetadataPlane::EvictBlocks(std::uint32_t max_blocks) {
  auto* radix = dynamic_cast<RadixIndex*>(index_.get());
  if (radix == nullptr) {
    throw LogicError(
        "MetadataPlane::EvictBlocks: eviction requires RadixIndex "
        "(Phase C)");
  }
  Evicted evicted = radix->Evict(max_blocks);
  evicted.bytes_reclaimed = RecycleEvicted(evicted);
  return evicted;
}

Evicted MetadataPlane::AbortPut(common::CacheEntry* entry) {
  auto* radix = dynamic_cast<RadixIndex*>(index_.get());
  if (radix == nullptr) {
    throw LogicError(
        "MetadataPlane::AbortPut: requires RadixIndex (Phase D)");
  }
  Evicted reclaimed = radix->RemoveEntry(entry);
  reclaimed.bytes_reclaimed = RecycleEvicted(reclaimed);
  return reclaimed;
}

std::uint64_t MetadataPlane::RecycleEvicted(const Evicted& evicted) {
  std::map<std::uint32_t, std::vector<common::BlockId>> blocks_by_group;
  std::uint64_t bytes = 0;
  for (const common::BlockRef& ref : evicted.blocks) {
    blocks_by_group[ref.group].push_back(ref.block);
    bytes += BlockBytesForGroup(ref.group);
  }
  for (const auto& [group, ids] : blocks_by_group) {
    const auto it = std::find(token_groups_.begin(), token_groups_.end(),
                              group);
    if (it == token_groups_.end()) {
      throw LogicError("MetadataPlane: evicted unknown group " +
                       std::to_string(group));
    }
    block_allocators_[static_cast<std::size_t>(
                          std::distance(token_groups_.begin(), it))]
        ->Release(ids);
  }
  for (const common::StateRef& ref : evicted.states) {
    const auto it = std::find(state_groups_.begin(), state_groups_.end(),
                              ref.group);
    if (it == state_groups_.end()) {
      throw LogicError("MetadataPlane: evicted unknown state group " +
                       std::to_string(ref.group));
    }
    state_pools_[static_cast<std::size_t>(
                     std::distance(state_groups_.begin(), it))]
        ->Release(ref);
  }
  return bytes;
}

}  // namespace stratakv::metadata
