#pragma once

#include <chrono>
#include <cstdint>
#include <atomic>
#include <functional>
#include <memory>
#include <span>
#include <vector>

#include "stratakv/common/handles.h"
#include "stratakv/common/model_layout.h"
#include "stratakv/metadata/hasher.h"

namespace stratakv::metadata {

/// \brief Options controlling readiness filtering and locking.
struct MatchOptions {
  /// \brief Increment refcounts along the matched (servable) prefix,
  /// pinning it against eviction until Release().
  bool lock_matched = true;

  /// \brief ADR 0003: only complete, landed bundles are servable. When
  /// false (staging path), matching ignores readiness and the hybrid
  /// bundle-boundary truncation — it reports raw hash-matched blocks so
  /// puts can extend not-yet-ready prefixes.
  bool require_ready = true;
};

/// \brief Opaque pin bookkeeping backing Release() (split-safe: Release
/// re-walks the query hashes instead of holding node pointers).
struct MatchPin {
  std::vector<BlockHash> hashes;      ///< Query block hashes (re-walk key).
  std::uint32_t tokens_per_block = 0;
  std::uint32_t matched_blocks = 0;   ///< Pinned prefix length in blocks.
  bool pins_states = false;           ///< Boundary state refs also pinned.
  std::atomic<bool> applied{false};   ///< Guards double Release().
};

/// \brief Result of a prefix match against the index.
struct MatchResult {
  std::uint32_t matched_blocks = 0;           ///< Longest servable prefix.
  /// \brief Present but not servable: hash-matched blocks that are not
  /// ready (transfer in flight) or lie beyond the last complete hybrid
  /// bundle boundary.
  std::uint32_t matched_blocks_reserved = 0;
  /// \brief Refs for matched_blocks, position-major / group-minor (one
  /// BlockRef per token-addressable ModelLayout group per position).
  std::vector<common::BlockRef> block_refs;
  /// \brief Boundary states for matched_blocks (hybrid layouts only).
  std::vector<common::StateRef> state_refs;
  /// \brief Token-level mask (true = externally servable), sized to the
  /// query token count.
  std::vector<std::uint8_t> token_mask;
  /// \brief Set iff MatchOptions::lock_matched; consumed by Release().
  std::shared_ptr<MatchPin> pin;
};

/// \brief Blocks/states reclaimed by Evict().
struct Evicted {
  /// \brief Evicted block refs, position-major / group-minor (tier-local
  /// ids; the caller recycles them into the owning allocators).
  std::vector<common::BlockRef> blocks;
  /// \brief Evicted boundary states (recycle into the state pools).
  std::vector<common::StateRef> states;
  /// \brief Entries whose owning nodes were evicted — these pointers are
  /// DANGLING from the caller's perspective and must be dropped.
  std::vector<const common::CacheEntry*> entries;
  /// \brief Bytes reclaimed (filled by MetadataPlane::EvictBlocks; 0
  /// when Evict is called on a bare index).
  std::uint64_t bytes_reclaimed = 0;
};

/// \brief Wall-clock source (nanoseconds since an arbitrary epoch);
/// injectable for deterministic grace-time tests.
using TimeSource = std::function<std::uint64_t()>;

/// \brief Interface over the prefix index (port of the FlexKV radix
/// tree).
///
/// Thread safety: implementations must serialize or shard internally;
/// all methods are safe to call from any engine thread.
class IIndex {
 public:
  virtual ~IIndex() = default;

  /// \brief Longest-prefix match with readiness filtering. Never blocks
  /// on I/O; pure metadata lookup.
  virtual MatchResult Match(std::span<const common::TokenId> tokens,
                            std::uint32_t tokens_per_block,
                            const MatchOptions& opts) = 0;

  /// \brief Insert (or extend) an entry for the block-aligned prefix.
  ///
  /// \param tokens Full token sequence; trailing partial-block tokens
  /// are ignored.
  /// \param blocks Refs for the blocks NOT already present, in prefix
  /// order, one per token-addressable group per position
  /// (blocks.size() == new_blocks * num_token_groups). Refcounts are
  /// zeroed on insertion.
  /// \param states One ref per LinearState group (the sidecar snapshot
  /// at this boundary); empty for homogeneous layouts.
  /// \return The boundary CacheEntry (owned by the index; valid until
  /// evicted), with is_ready = false until CommitReady(), or nullptr
  /// when the full block-aligned prefix is already present (nothing to
  /// insert; FlexKV parity).
  virtual common::CacheEntry* Insert(
      std::span<const common::TokenId> tokens,
      std::vector<common::BlockRef> blocks,
      std::vector<common::StateRef> states) = 0;

  /// \brief Flip an entry (and its node) to ready after its transfer
  /// completed.
  ///
  /// Span-bounded ancestor marking (FlexKV's set_ready ready_length
  /// walk, restricted to the blocks THIS put wrote): ancestors created
  /// by splits of this entry's own node become ready with it; ancestors
  /// belonging to earlier, still-in-flight puts are NOT marked (no
  /// over-marking — see RadixIndexTest.ConservativeCommitDoesNotOverMark
  /// Ancestors).
  ///
  /// \throws LogicError if the entry is unknown (or was evicted).
  virtual void CommitReady(common::CacheEntry* entry) = 0;

  /// \brief Drop refcounts acquired by a Match(lock_matched=true).
  /// Idempotent per MatchResult.
  virtual void Release(const MatchResult& match) = 0;
};

/// \brief Chained-hash radix index (port of FlexKV `CRadixTreeIndex`).
///
/// Hash contract: hash(i) covers all tokens up to and including block i
/// (see GenBlockHashes). Node payloads are generalized to per-group
/// BlockRef bundles plus boundary StateRefs (ADR 0003); split/shrink/
/// evict semantics follow FlexKV. Layout invariants enforced at
/// construction: at least one token-addressable group exists, and all
/// token-addressable groups share one tokens_per_block (per-group block
/// sizing is deferred until a model needs it).
///
/// Deliberate divergences from the FlexKV port (each covered by unit
/// tests): (1) the child-hash lookup is skipped when the node would
/// fully consume the query — FlexKV reads one element past the hash
/// tensor there; (2) ready accounting is a CONTIGUOUS prefix (FlexKV
/// sums ready segments even across unready gaps, so it could serve
/// through unready data); (3) CommitReady marks its own node plus the
/// split ancestors inside this put's span only (FlexKV's ancestor walk
/// can over-mark blocks of still-in-flight earlier puts).
class RadixIndex final : public IIndex {
 public:
  /// \param layout Model description; must outlive the index.
  /// \param now Wall-clock source (defaults to steady_clock).
  /// \param eviction_grace Leaves younger than this are never selected
  /// for eviction (FlexKV has no grace; keep 0 for FlexKV parity).
  ///
  /// \throws std::invalid_argument on layout invariant violations.
  RadixIndex(const common::ModelLayout& layout, TimeSource now = nullptr,
             std::chrono::milliseconds eviction_grace = {});

  ~RadixIndex() override;

  RadixIndex(const RadixIndex&) = delete;
  RadixIndex& operator=(const RadixIndex&) = delete;

  MatchResult Match(std::span<const common::TokenId> tokens,
                    std::uint32_t tokens_per_block,
                    const MatchOptions& opts) override;

  common::CacheEntry* Insert(std::span<const common::TokenId> tokens,
                             std::vector<common::BlockRef> blocks,
                             std::vector<common::StateRef> states) override;

  void CommitReady(common::CacheEntry* entry) override;

  void Release(const MatchResult& match) override;

  /// \brief Evict up to max_blocks LRU-first from evictable leaves
  /// (leaf, ready, unpinned, grace-expired). Leaf nodes carrying
  /// boundary states are only evicted whole (never partially shrunk), so
  /// the returned count may exceed max_blocks on hybrid layouts;
  /// homogeneous leaves shrink exactly like FlexKV's tree.
  Evicted Evict(std::uint32_t max_blocks);

  /// \brief Drop everything and reinitialize (test/admin helper).
  void Reset();

  /// \brief Blocks stored across all nodes.
  std::uint32_t TotalCachedBlocks() const;
  /// \brief Blocks in ready nodes.
  std::uint32_t TotalReadyBlocks() const;
  /// \brief Node count excluding the root.
  std::size_t TotalNodes() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace stratakv::metadata
