#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "stratakv/common/handles.h"
#include "stratakv/common/model_layout.h"
#include "stratakv/metadata/allocator.h"
#include "stratakv/metadata/index.h"

namespace stratakv::metadata {

/// \brief Configuration for the CPU tier's metadata pools (Phase C is
/// metadata-only; bytes land in Phase E).
struct MetadataPlaneConfig {
  /// \brief Free-list capacity per token-addressable group.
  std::uint32_t blocks_per_group = 1u << 20;
  /// \brief State-object pool size per LinearState group.
  std::uint32_t state_objects_per_group = 1024;
  /// \brief Bytes of one state object (accounting only here).
  std::uint64_t state_object_bytes = 1u << 16;
  /// \brief Element size for byte accounting (default fp16).
  int dtype_bytes = 2;
  /// \brief Leaves younger than this are never evicted (0 = FlexKV
  /// parity).
  std::chrono::milliseconds eviction_grace{0};
};

/// \brief Facade wiring index, allocators, and eviction (design.md
/// Phase C facade).
///
/// Owns one BlockAllocator per token-addressable group and one StatePool
/// per LinearState group for the CPU tier (TierId 0 in Phase C).
/// Thread safety: all public methods safe from any engine thread (the
/// index serializes internally; allocators are individually locked;
/// multi-structure operations are sequenced by protocol, matching the
/// single-scheduler control plane of Phase D).
class MetadataPlane {
 public:
  /// \brief Offload plan returned by PlanPut/StagePut (design.md PutPlan
  /// plus Phase C staging fields).
  struct PutPlan {
    /// \brief Reserved CPU blocks for the unmatched suffix,
    /// position-major / group-minor (empty for a PlanPut preview).
    std::vector<common::BlockRef> blocks_to_store;
    /// \brief Reserved state objects for the new boundary (empty for a
    /// PlanPut preview).
    std::vector<common::StateRef> states_to_snapshot;
    /// \brief Staged boundary entry (StagePut only); not ready until
    /// CommitPut. Owned by the index; invalidated by eviction.
    common::CacheEntry* entry = nullptr;
    /// \brief Blocks of the prefix already present (raw match, ignoring
    /// readiness — puts extend in-flight prefixes).
    std::uint32_t matched_blocks = 0;
    /// \brief Blocks the put must store (query blocks - matched).
    std::uint32_t new_blocks = 0;
    /// \brief PlanPut only: false when a group lacks capacity for
    /// new_blocks.
    bool feasible = true;
  };

  /// \param layout Model description; must outlive the plane.
  /// \param index Prefix index (e.g. RadixIndex); owned by the plane.
  /// \param config Pool sizing.
  ///
  /// \throws std::invalid_argument via RadixIndex when the layout has no
  /// token-addressable group or non-uniform tokens_per_block.
  MetadataPlane(const common::ModelLayout& layout,
                std::unique_ptr<IIndex> index,
                MetadataPlaneConfig config = {});

  ~MetadataPlane();

  MetadataPlane(const MetadataPlane&) = delete;
  MetadataPlane& operator=(const MetadataPlane&) = delete;

  /// \brief Readiness-filtered match for the scheduler-critical path
  /// (locks/pins the matched prefix; budget p50 < 100 µs at 1k blocks —
  /// see benchmarks/index_bench).
  MatchResult MatchForGet(std::span<const common::TokenId> tokens);

  /// \brief Read-only decision: how many trailing blocks/states a put of
  /// `tokens` would store, and whether the pools have capacity.
  PutPlan PlanPut(std::span<const common::TokenId> tokens);

  /// \brief Reserve storage, create the (not-ready) entry, return the
  /// plan. The caller transfers blocks_to_store/states_to_snapshot data
  /// (Phase E) and finishes with CommitPut.
  ///
  /// \throws NotEnoughSpaceError when any group's allocator or a state
  /// pool is exhausted (nothing is reserved in that case).
  [[nodiscard]] PutPlan StagePut(std::span<const common::TokenId> tokens);

  /// \brief Called by the control plane when a put task succeeds:
  /// flips the staged entry (and its node) to ready.
  ///
  /// The design.md TaskHandle-keyed overload arrives with the Phase D
  /// task engine (which owns the task→entry map); Phase C stages via
  /// PutPlan::entry.
  ///
  /// \throws LogicError on an unknown (or evicted) entry.
  void CommitPut(common::CacheEntry* entry);

  /// \brief Called on get-completion or request abort; drops pins
  /// acquired by MatchForGet. Idempotent per MatchResult.
  void Release(const MatchResult& match);

  /// \brief Abort a staged put (or force-drop a committed one):
  /// removes its entry/node from the index and recycles the reserved
  /// block/state ids into the pools.
  ///
  /// \throws LogicError (via the index) if the entry was extended or
  /// pinned, or is unknown.
  /// \return The Evicted record describing what was reclaimed.
  Evicted AbortPut(common::CacheEntry* entry);

  /// \brief Evict up to max_blocks LRU-first, recycling reclaimed ids
  /// into the owning pools.
  ///
  /// \return The Evicted record (blocks/states reclaimed, entries now
  /// dangling, bytes accounted per the layout geometry).
  Evicted EvictBlocks(std::uint32_t max_blocks);

  /// \brief The wrapped index (tests/introspection).
  IIndex& index() { return *index_; }

  /// \brief Free block ids for a token-addressable group (by group
  /// index, not pool index).
  std::uint32_t num_free_blocks(std::uint32_t group) const;

  /// \brief Free state objects for a LinearState group.
  std::uint32_t num_free_states(std::uint32_t group) const;

  /// \brief Bytes of one MODEL-WIDE token block in `group` per the
  /// layout geometry (per-layer BlockBytes × layers in group — the
  /// FlexKV physical-block contract; 0 for LinearState groups).
  std::uint64_t BlockBytesForGroup(std::uint32_t group) const {
    const common::LayerGroup& group_spec =
        layout_.Group(static_cast<int>(group));
    return common::BlockBytes(group_spec.spec,
                              static_cast<int>(tokens_per_block_),
                              config_.dtype_bytes) *
           group_spec.layer_indices.size();
  }

 private:
  PutPlan PlanLocked(std::span<const common::TokenId> tokens,
                     bool stage);

  /// \brief Recycle an Evicted record's ids into the owning pools;
  /// returns reclaimed bytes (shared by EvictBlocks and AbortPut).
  std::uint64_t RecycleEvicted(const Evicted& evicted);

  const common::ModelLayout& layout_;
  std::unique_ptr<IIndex> index_;
  MetadataPlaneConfig config_;
  std::uint32_t tokens_per_block_ = 0;
  std::vector<std::uint32_t> token_groups_;
  std::vector<std::uint32_t> state_groups_;
  std::vector<std::unique_ptr<BlockAllocator>> block_allocators_;
  std::vector<std::unique_ptr<StatePool>> state_pools_;
};

}  // namespace stratakv::metadata
