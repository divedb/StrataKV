#include "stratakv/data/cpu_tier.h"

#include "stratakv/common/exceptions.h"
#include "host_memory.h"

namespace stratakv::data {

namespace {
std::size_t LayersIn(const common::LayerGroup& group) {
  return group.layer_indices.size();
}
}  // namespace

void* CpuTier::Allocate(std::uint64_t bytes) const {
  switch (backing_) {
    case HostBacking::kPageable:
      return host_memory::AllocatePageable(config_.alignment, bytes);
#if defined(STRATAKV_HAVE_CUDA)
    case HostBacking::kPinned:
      return host_memory::AllocatePinned(config_.alignment, bytes);
    case HostBacking::kRegistered:
      return host_memory::AllocateRegistered(config_.alignment, bytes);
#else
    case HostBacking::kPinned:
    case HostBacking::kRegistered:
      throw LogicError(
          "CpuTier: page-locked backing requires a STRATAKV_GPU build");
#endif
  }
  return nullptr;  // Unreachable.
}

void CpuTier::Deallocate(void* base) const {
  switch (backing_) {
    case HostBacking::kPageable:
      host_memory::FreePageable(base);
      return;
#if defined(STRATAKV_HAVE_CUDA)
    case HostBacking::kPinned:
      host_memory::FreePinned(base);
      return;
    case HostBacking::kRegistered:
      host_memory::FreeRegistered(base);
      return;
#else
    case HostBacking::kPinned:
    case HostBacking::kRegistered:
      return;  // Never allocated in this build (Allocate throws).
#endif
  }
}

CpuTier::CpuTier(const common::ModelLayout& layout, CpuTierConfig config)
    : layout_(layout), config_(config), backing_(config.backing) {
  // Fail loudly BEFORE any reservation when the build cannot honor the
  // requested backing (no silent pageable fallback).
#if !defined(STRATAKV_HAVE_CUDA)
  if (backing_ != HostBacking::kPageable) {
    throw LogicError(
        "CpuTier: page-locked backing requires a STRATAKV_GPU build");
  }
#endif

  for (std::size_t g = 0; g < layout_.num_groups(); ++g) {
    const common::LayerGroup& group =
        layout_.Group(static_cast<int>(g));
    if (stratakv::common::IsTokenAddressable(group.spec)) {
      token_groups_.push_back(static_cast<std::uint32_t>(g));
    } else {
      state_groups_.push_back(static_cast<std::uint32_t>(g));
    }
  }
  tokens_per_block_ = token_groups_.empty()
      ? 0
      : static_cast<std::uint32_t>(
            layout_.Group(static_cast<int>(token_groups_.front()))
                .tokens_per_block);

  for (std::uint32_t group : token_groups_) {
    const common::LayerSpec spec =
        layout_.Group(static_cast<int>(group)).spec;
    // Dtype: 2 bytes (fp16) — the Phase C/D default geometry.
    // Model-wide block: per-layer slice × layers (FlexKV physical-block
    // contract; the group's pool is one flat BLOCKWISE reservation).
    const std::uint64_t slot_bytes =
        stratakv::common::BlockBytes(spec,
                                     static_cast<int>(tokens_per_block_),
                                     /*dtype_bytes=*/2) *
        LayersIn(layout_.Group(static_cast<int>(group)));
    if (slot_bytes == 0) {
      throw std::invalid_argument(
          "CpuTier: token-addressable group with zero block bytes");
    }
    Pool pool;
    pool.group = group;
    pool.slot_bytes = slot_bytes;
    pool.bytes = slot_bytes * config_.blocks_per_group;
    pool.base = Allocate(pool.bytes);
    block_pools_.push_back(pool);
  }
  for (std::uint32_t group : state_groups_) {
    const common::LayerSpec spec =
        layout_.Group(static_cast<int>(group)).spec;
    // Model-wide state object: per-layer state bytes × layers.
    const std::uint64_t slot_bytes =
        stratakv::common::StateBytes(spec) *
        LayersIn(layout_.Group(static_cast<int>(group)));
    if (slot_bytes == 0) {
      throw std::invalid_argument(
          "CpuTier: LinearState group with zero state bytes");
    }
    Pool pool;
    pool.group = group;
    pool.slot_bytes = slot_bytes;
    pool.bytes = slot_bytes * config_.state_objects_per_group;
    pool.base = Allocate(pool.bytes);
    state_pools_.push_back(pool);
  }
}

CpuTier::~CpuTier() {
  for (Pool& pool : block_pools_) {
    Deallocate(pool.base);
  }
  for (Pool& pool : state_pools_) {
    Deallocate(pool.base);
  }
}

common::BufferView CpuTier::PoolForGroup(std::uint32_t group) const {
  for (const Pool& pool : block_pools_) {
    if (pool.group != group) continue;
    common::BufferView view;
    view.base = pool.base;
    view.kind = memory_kind();
    view.device_id = -1;
    view.size_bytes = pool.bytes;
    return view;
  }
  throw LogicError("CpuTier: group " + std::to_string(group) +
                   " is not a token-addressable group of this tier");
}

void* CpuTier::BlockAddress(std::uint32_t group,
                            common::BlockId block) const {
  for (const Pool& pool : block_pools_) {
    if (pool.group != group) continue;
    if (static_cast<std::uint64_t>(block) >= config_.blocks_per_group) {
      throw LogicError("CpuTier: block id " + std::to_string(block) +
                       " out of range");
    }
    return static_cast<char*>(pool.base) +
           static_cast<std::ptrdiff_t>(
               static_cast<std::uint64_t>(block) * pool.slot_bytes);
  }
  throw LogicError("CpuTier: group " + std::to_string(group) +
                   " is not a token-addressable group of this tier");
}

void* CpuTier::StateAddress(std::uint32_t group,
                            std::uint32_t object) const {
  for (const Pool& pool : state_pools_) {
    if (pool.group != group) continue;
    if (object >= config_.state_objects_per_group) {
      throw LogicError("CpuTier: state object " + std::to_string(object) +
                       " out of range");
    }
    return static_cast<char*>(pool.base) +
           static_cast<std::ptrdiff_t>(
               static_cast<std::uint64_t>(object) * pool.slot_bytes);
  }
  throw LogicError("CpuTier: group " + std::to_string(group) +
                   " is not a LinearState group of this tier");
}

std::uint64_t CpuTier::PoolBytesForGroup(std::uint32_t group) const {
  for (const Pool& pool : block_pools_) {
    if (pool.group == group) return pool.bytes;
  }
  throw LogicError("CpuTier: group " + std::to_string(group) +
                   " is not a token-addressable group of this tier");
}

std::uint64_t CpuTier::BlockSlotBytes(std::uint32_t group) const {
  for (const Pool& pool : block_pools_) {
    if (pool.group == group) return pool.slot_bytes;
  }
  throw LogicError("CpuTier: group " + std::to_string(group) +
                   " is not a token-addressable group of this tier");
}

std::uint64_t CpuTier::StateSlotBytes(std::uint32_t group) const {
  for (const Pool& pool : state_pools_) {
    if (pool.group == group) return pool.slot_bytes;
  }
  throw LogicError("CpuTier: group " + std::to_string(group) +
                   " is not a LinearState group of this tier");
}

common::DeviceKind CpuTier::memory_kind() const {
  switch (backing_) {
    case HostBacking::kPageable:
      return common::DeviceKind::kCpu;
    case HostBacking::kPinned:
    case HostBacking::kRegistered:
      return common::DeviceKind::kCpuPinned;
  }
  return common::DeviceKind::kCpu;
}

}  // namespace stratakv::data
