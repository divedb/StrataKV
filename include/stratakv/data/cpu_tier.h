#pragma once

#include <cstdint>
#include <vector>

#include "stratakv/common/model_layout.h"
#include "stratakv/common/types.h"
#include "stratakv/metadata/metadata_plane.h"

namespace stratakv::data {

/// \brief Host-memory backing of a CpuTier's pools (Phase E2).
enum class HostBacking : std::uint8_t {
  kPageable,   ///< posix_memalign heap (CPU-only builds; copy-engine only).
  kPinned,     ///< cudaMallocHost — SM-accessible page-locked memory; the
               ///< ONLY kernel-safe host backing on WSL2 (Phase 0, F4).
               ///< Requires STRATAKV_GPU.
  kRegistered  ///< Heap + cudaHostRegister. Copy-engine accessible on all
               ///< platforms; SM access is platform-dependent (faults on
               ///< WSL2), so the transport routes it through memcpys.
               ///< Requires STRATAKV_GPU.
};

/// \brief Configuration of one CPU tier instance.
struct CpuTierConfig {
  /// \brief Pool capacity per token-addressable group; MUST match the
  /// MetadataPlaneConfig handed to the owning plane (ids are shared).
  std::uint32_t blocks_per_group = 1u << 20;
  /// \brief State-object pool size per LinearState group.
  std::uint32_t state_objects_per_group = 1024;
  /// \brief Alignment of each pool base (2 MiB = default hugepage-ish;
  /// page-locked pools are >=256 B aligned by construction).
  std::size_t alignment = 2u << 20;
  /// \brief Page-locking policy (GPU builds only; see HostBacking).
  HostBacking backing = HostBacking::kPageable;
};

/// \brief One CPU tier: the byte-backed pools behind the metadata
/// plane's id spaces (roadmap E2).
///
/// Division of labor (deliberate, Phase C heritage): the
/// MetadataPlane owns the BlockAllocator/StatePool ID spaces —
/// StagePut/EvictBlocks drive them — and this tier owns the BYTES each
/// id names. BlockAddress/StateAddress map ids to pointers with the
/// layout's geometry; nothing else moves.
///
/// Block granularity (the FlexKV physical-block contract, roadmap
/// Appendix B): one block id names the MODEL-WIDE block of its group —
/// every layer's slice for those tokens, laid out contiguously
/// [layer][slice]. Slot bytes = common::BlockBytes(spec, tpb, dtype) ×
/// layers-in-group. State objects follow the same rule:
/// StateBytes(spec) × layers-in-group per object id. The CPU pool is
/// one flat BLOCKWISE reservation per group (LAYERWISE decompositions
/// live on the GPU side only).
///
/// Memory backing: kPageable on any build; kPinned (cudaMallocHost)
/// and kRegistered (cudaHostRegister) require the STRATAKV_GPU build
/// and fail loudly elsewhere. The backing decides the transport's copy
/// path (SM kernels vs copy engine) via memory_kind().
///
/// Host-registration chunking policy (E2 documentation requirement):
/// one contiguous reservation per group, sized for the full pool; tiers
/// beyond ~capacity of one reservation must be split into multiple
/// CpuTier instances (per-group) rather than growing a pool — keeps
/// NUMA placement explicit and avoids mid-pool reallocation entirely.
///
/// Thread safety: address math only — safe from any thread once
/// constructed; construction/destruction is the owner's to serialize.
class CpuTier {
 public:
  /// \param layout Model description (must outlive the tier).
  /// \param config Pool sizing (must match the owning plane's config).
  ///
  /// \throws std::invalid_argument on layout geometry violations
  /// (zero-sized blocks for token-addressable groups etc.),
  /// stratakv::LogicError on a page-locked backing in a CPU-only
  /// build, and std::bad_alloc/std::runtime_error when the
  /// reservations cannot be made.
  CpuTier(const common::ModelLayout& layout, CpuTierConfig config);

  ~CpuTier();

  CpuTier(const CpuTier&) = delete;
  CpuTier& operator=(const CpuTier&) = delete;

  /// \brief The tier's single flat pool for one token-addressable
  /// group (BLOCKWISE; kind per the backing, tier-local device id -1).
  common::BufferView PoolForGroup(std::uint32_t group) const;

  /// \brief Address of one block slot (id from the plane's allocator).
  /// \throws LogicError for out-of-range ids or non-token groups.
  void* BlockAddress(std::uint32_t group, common::BlockId block) const;

  /// \brief Address of one state-object slot.
  /// \throws LogicError for out-of-range ids or non-state groups.
  void* StateAddress(std::uint32_t group, std::uint32_t object) const;

  /// \brief Bytes actually reserved for `group`'s block pool.
  std::uint64_t PoolBytesForGroup(std::uint32_t group) const;

  /// \brief Bytes of one model-wide block slot in `group`
  /// (per-layer slice × layers in group).
  std::uint64_t BlockSlotBytes(std::uint32_t group) const;

  /// \brief Bytes of one model-wide state object in `group`
  /// (per-layer state bytes × layers in group).
  std::uint64_t StateSlotBytes(std::uint32_t group) const;

  /// \brief Memory kind of every pool (per the backing).
  common::DeviceKind memory_kind() const;

  HostBacking backing() const { return backing_; }
  std::uint32_t tokens_per_block() const { return tokens_per_block_; }

 private:
  struct Pool {
    void* base = nullptr;
    std::uint64_t bytes = 0;
    std::uint32_t group = 0;
    std::uint64_t slot_bytes = 0;  ///< Per block/state id.
  };

  void* Allocate(std::uint64_t bytes) const;
  void Deallocate(void* base) const;

  const common::ModelLayout& layout_;
  CpuTierConfig config_;
  HostBacking backing_ = HostBacking::kPageable;
  std::uint32_t tokens_per_block_ = 0;
  std::vector<Pool> block_pools_;   ///< Per token-addressable group.
  std::vector<Pool> state_pools_;   ///< Per LinearState group.
  std::vector<std::uint32_t> token_groups_;
  std::vector<std::uint32_t> state_groups_;
};

}  // namespace stratakv::data
