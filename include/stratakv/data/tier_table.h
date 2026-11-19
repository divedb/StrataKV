#pragma once

#include <cstdint>
#include <map>
#include <utility>
#include <vector>

#include "stratakv/common/model_layout.h"
#include "stratakv/common/types.h"
#include "stratakv/data/cpu_tier.h"
#include "stratakv/data/registry.h"

namespace stratakv::data {

/// \brief Addressing geometry of one (tier, group) storage pool, as the
/// transfer engine needs it (Phase E3).
///
/// One block/state id names the MODEL-WIDE slot of its group. `bases`
/// holds one pointer per layer segment of the group: LAYERWISE GPU
/// registrations contribute one base per layer (block id b addresses
/// base + b * block_stride in EACH base, slice_bytes per base); a flat
/// BLOCKWISE pool contributes a single base whose slice covers the
/// whole model-wide slot. Token blocks and state objects use identical
/// math (is_state distinguishes them for validation/errs).
///
/// Thread safety: populated at wiring time, then read-only; safe to
/// share across transport threads. References engine/tier-owned memory
/// that must outlive the transport.
struct GroupAddressing {
  std::vector<void*> bases;        ///< One per layer segment (>= 1).
  std::uint64_t block_stride = 0;  ///< Bytes between consecutive ids, per base.
  std::uint64_t slice_bytes = 0;   ///< Bytes moved per (id, base) pair.
  common::DeviceKind kind{};       ///< Memory kind of every base.
  bool sm_accessible = false;      ///< May SM kernels touch these bases?
                                    ///< cudaMallocHost and device memory:
                                    ///< yes. cudaHostRegister/pageable:
                                    ///< platform-dependent (faults on
                                    ///< WSL2 — benchmark F4) — conservative
                                    ///< false; copy engine only.
  bool is_state = false;           ///< State-object (not token) addressing.
};

/// \brief Tier-local id → buffer geometry for the transport (E3).
///
/// Owns nothing: entries reference CpuTier pools and DeviceMemoryRegistry
/// registrations (engine-owned GPU memory, ADR 0005) that must outlive
/// both the table and any transport using it.
///
/// Thread safety: not thread-safe for population; read-only LookUp is
/// safe from any thread afterwards.
class TierTable {
 public:
  /// \brief Add one CPU tier's groups (BLOCKWISE pools; geometry from
  /// the tier's own slot math). The tier must outlive the table.
  ///
  /// \throws stratakv::LogicError on a duplicate (tier, group) entry.
  void AddCpuTier(common::TierId tier, const common::ModelLayout& layout,
                  const CpuTier& cpu_tier);

  /// \brief Add one GPU registration's groups (LAYERWISE: one base per
  /// layer; BLOCKWISE: one flat base). `capacity_slots` is the number of
  /// block ids (token groups) or state-object ids (LinearState groups)
  /// the registered buffers hold; the per-base stride is derived as
  /// view size / capacity and must divide exactly.
  ///
  /// \throws stratakv::LogicError on unknown registration, duplicate
  /// (tier, group), mismatched view counts, or non-dividing capacity.
  void AddGpuTier(common::TierId tier, const DeviceMemoryRegistry& registry,
                  DeviceMemoryRegistry::RegistrationId registration,
                  const common::ModelLayout& layout,
                  std::uint32_t capacity_slots);

  /// \brief Geometry for (tier, group); nullptr when unknown.
  const GroupAddressing* LookUp(common::TierId tier,
                                std::uint32_t group) const;

 private:
  void Insert(common::TierId tier, std::uint32_t group,
              GroupAddressing addressing);

  std::map<std::pair<common::TierId, std::uint32_t>, GroupAddressing>
      entries_;
};

}  // namespace stratakv::data
