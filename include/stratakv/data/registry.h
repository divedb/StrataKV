#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "stratakv/common/model_layout.h"
#include "stratakv/common/tier.h"
#include "stratakv/common/types.h"

namespace stratakv::data {

/// \brief Opaque CUDA IPC handle for one registered allocation (ADR
/// 0005: 64-byte payload, no torch ABI below the adapter).
struct IpcHandle {
  std::array<std::uint8_t, 64> opaque{};
};

/// \brief Layout of a group's per-layer buffers (roadmap E1; FlexKV
/// Appendix B heritage: GPU tensors arrive per-layer (LAYERWISE) or as
/// one flat pool (BLOCKWISE)).
enum class GroupLayout : std::uint8_t {
  kLayerwise,  ///< One BufferView per layer of the group.
  kBlockwise,  ///< One flat BufferView covering the whole group pool.
};

/// \brief Descriptor of one GPU group registration (ADR 0005: explicit
/// descriptors, never torch pointers).
///
/// Ownership: buffers remain engine property; StrataKV never frees
/// registered memory. Registered buffers must remain valid until
/// Unregister (adapter contract, mirrors vLLM register_kv_caches).
struct GroupDescriptor {
  std::string group_name;          ///< ModelLayout group name.
  std::uint32_t group = 0;         ///< ModelLayout group index.
  GroupLayout layout = GroupLayout::kLayerwise;
  /// \brief Per-layer views (kLayerwise: exactly the group's layer
  /// count; kBlockwise: exactly one view).
  std::vector<common::BufferView> per_layer;
  int device_id = 0;
};

/// \brief Source of real CUDA IPC handles (cudaIpcGetMemHandle).
///
/// The STRATAKV_GPU build installs the CUDA implementation; without
/// one, ExportIpcHandles fails loudly (no silent stubs). Tests may
/// install their own source to exercise the registry contract.
class IIpcHandleSource {
 public:
  virtual ~IIpcHandleSource() = default;

  /// \brief Export one registered allocation. Called under the
  /// registry's registration; implementations must not call back into
  /// the registry.
  virtual IpcHandle Export(const common::BufferView& view) = 0;
};

/// \brief Registry of engine-owned GPU memory StrataKV may access
/// (ADR 0005: the sole gateway).
///
/// Thread safety: all methods safe from any engine thread.
/// Ownership: registration NEVER transfers buffer ownership; Unregister
/// invalidates all exported handles immediately (generation bump) and
/// is the engine's responsibility at shutdown/realloc.
class DeviceMemoryRegistry {
 public:
  using RegistrationId = std::uint64_t;

  /// \brief Register one group's buffers.
  ///
  /// \throws std::invalid_argument on malformed descriptors (unknown
  /// group index, wrong view count for the layout, invalid views,
  /// non-GPU views, device id < 0).
  /// \return Stable registration id (non-zero).
  RegistrationId RegisterGroup(GroupDescriptor descriptor);

  /// \brief Install the CUDA IPC source (STRATAKV_GPU builds).
  void SetIpcHandleSource(std::shared_ptr<IIpcHandleSource> source);

  /// \brief Export CUDA IPC handles for one registration's views.
  ///
  /// \throws LogicError when no IPC source is installed (CPU-only
  /// build — in-process mode needs no IPC, ADR 0002) or the
  /// registration is unknown/unregistered.
  std::vector<IpcHandle> ExportIpcHandles(RegistrationId id) const;

  /// \brief In-process access (ADR 0002: same registry contract in
  /// both isolation modes): views of one registration.
  ///
  /// \throws LogicError when unknown/unregistered.
  std::vector<common::BufferView> ViewsOf(RegistrationId id) const;

  /// \brief The descriptor of one registration (wiring/introspection —
  /// e.g. TierTable reads group index and layout kind).
  ///
  /// \throws LogicError when unknown/unregistered.
  const GroupDescriptor& DescriptorOf(RegistrationId id) const;

  /// \brief The registration owning `group`, or 0 when unregistered
  /// (single registration per group at a time).
  RegistrationId RegistrationForGroup(std::uint32_t group) const;

  /// \brief True iff this handle was exported by the current
  /// generation of its registration (ADR 0005 immediate invalidation).
  bool IsHandleValid(const IpcHandle& handle) const;

  /// \brief Invalidate the registration and every handle it exported.
  ///
  /// \throws LogicError when unknown. Idempotent per id? No — unknown
  /// ids throw (double unregister is an adapter bug).
  void Unregister(RegistrationId id);

  std::size_t num_registrations() const;

 private:
  struct Registration {
    GroupDescriptor descriptor;
    /// \brief Handles exported by the live generation; cleared on
    /// Unregister (ADR 0005 immediate invalidation). The payloads stay
    /// pristine — validity is registry-side bookkeeping, never bytes
    /// embedded into the opaque CUDA payload.
    mutable std::vector<IpcHandle> exported;
    bool live = true;
  };

  mutable std::mutex mu_;
  std::shared_ptr<IIpcHandleSource> ipc_source_;
  std::unordered_map<RegistrationId, Registration> registrations_;
  std::unordered_map<std::uint32_t, RegistrationId> group_owner_;
  RegistrationId next_id_ = 1;
};

}  // namespace stratakv::data
