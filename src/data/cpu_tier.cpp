#include "stratakv/data/cpu_tier.h"

#include <cstdlib>

#include "stratakv/common/exceptions.h"
namespace stratakv::data {

namespace {

void* AlignedAlloc(std::size_t alignment, std::uint64_t bytes) {
  // posix_memalign (no size-multiple constraint, unlike
  // std::aligned_alloc); the GPU build swaps this for page-locked
  // reservations (cudaHostRegister).
  if (bytes == 0) {
    return nullptr;
  }
  void* base = nullptr;
  if (::posix_memalign(&base, alignment,
                       static_cast<std::size_t>(bytes)) != 0 ||
      base == nullptr) {
    throw std::runtime_error("CpuTier: pool reservation failed");
  }
  return base;
}

}  // namespace

CpuTier::CpuTier(const common::ModelLayout& layout, CpuTierConfig config)
    : layout_(layout), config_(config) {
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
    const std::uint64_t slot_bytes =
        stratakv::common::BlockBytes(spec,
                                     static_cast<int>(tokens_per_block_),
                                     /*dtype_bytes=*/2);
    if (slot_bytes == 0) {
      throw std::invalid_argument(
          "CpuTier: token-addressable group with zero block bytes");
    }
    Pool pool;
    pool.group = group;
    pool.slot_bytes = slot_bytes;
    pool.bytes = slot_bytes * config_.blocks_per_group;
    pool.base = AlignedAlloc(config_.alignment, pool.bytes);
    block_pools_.push_back(pool);
  }
  for (std::uint32_t group : state_groups_) {
    const common::LayerSpec spec =
        layout_.Group(static_cast<int>(group)).spec;
    const std::uint64_t slot_bytes =
        stratakv::common::StateBytes(spec);
    if (slot_bytes == 0) {
      throw std::invalid_argument(
          "CpuTier: LinearState group with zero state bytes");
    }
    Pool pool;
    pool.group = group;
    pool.slot_bytes = slot_bytes;
    pool.bytes = slot_bytes * config_.state_objects_per_group;
    pool.base = AlignedAlloc(config_.alignment, pool.bytes);
    state_pools_.push_back(pool);
  }
}

CpuTier::~CpuTier() {
  for (Pool& pool : block_pools_) {
    std::free(pool.base);
  }
  for (Pool& pool : state_pools_) {
    std::free(pool.base);
  }
}

common::BufferView CpuTier::PoolForGroup(std::uint32_t group) const {
  for (const Pool& pool : block_pools_) {
    if (pool.group != group) continue;
    common::BufferView view;
    view.base = pool.base;
    view.kind = common::DeviceKind::kCpu;
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

}  // namespace stratakv::data
