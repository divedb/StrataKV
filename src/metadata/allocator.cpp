#include "stratakv/metadata/allocator.h"

#include <algorithm>
#include <stdexcept>

#include "stratakv/common/exceptions.h"

namespace stratakv::metadata {

BlockAllocator::BlockAllocator(std::uint32_t total_blocks)
    : total_blocks_(total_blocks),
      free_mask_(total_blocks, 1),
      num_free_(total_blocks) {
  if (total_blocks == 0) {
    throw std::invalid_argument("BlockAllocator: total_blocks must be > 0");
  }
  MaterializeFreeIds();
}

void BlockAllocator::MaterializeFreeIds() {
  // Mempool parity: a materialized array of free ids consumed via a
  // moving offset; rebuilt only when the tail is exhausted.
  free_ids_.clear();
  free_ids_.reserve(num_free_);
  for (std::uint32_t id = 0; id < total_blocks_; ++id) {
    if (free_mask_[id] != 0) {
      free_ids_.push_back(id);
    }
  }
  free_ids_offset_ = 0;
}

std::vector<common::BlockId> BlockAllocator::Allocate(std::uint32_t n) {
  std::lock_guard<std::mutex> lock(mu_);
  if (n == 0) {
    throw std::invalid_argument("BlockAllocator: Allocate(0) is invalid");
  }
  if (n > num_free_) {
    throw NotEnoughSpaceError("BlockAllocator: not enough free blocks (need " +
                              std::to_string(n) + ", free " +
                              std::to_string(num_free_) + ")");
  }
  if (static_cast<std::size_t>(n) >
      free_ids_.size() - free_ids_offset_) {
    MaterializeFreeIds();
  }
  std::vector<common::BlockId> ids(
      free_ids_.begin() + static_cast<std::ptrdiff_t>(free_ids_offset_),
      free_ids_.begin() + static_cast<std::ptrdiff_t>(free_ids_offset_ + n));
  free_ids_offset_ += n;
  for (common::BlockId id : ids) {
    free_mask_[id] = 0;
  }
  num_free_ -= n;
  return ids;
}

void BlockAllocator::Release(std::span<const common::BlockId> ids) {
  std::lock_guard<std::mutex> lock(mu_);
  for (common::BlockId id : ids) {
    if (id >= total_blocks_) {
      throw LogicError("BlockAllocator: id " + std::to_string(id) +
                       " out of range [0, " + std::to_string(total_blocks_) +
                       ")");
    }
    if (free_mask_[id] != 0) {
      throw LogicError("BlockAllocator: cannot recycle free block " +
                       std::to_string(id) + " repeatedly");
    }
  }
  for (common::BlockId id : ids) {
    free_mask_[id] = 1;
  }
  num_free_ += static_cast<std::uint32_t>(ids.size());
}

std::uint32_t BlockAllocator::num_free() const {
  std::lock_guard<std::mutex> lock(mu_);
  return num_free_;
}

StatePool::StatePool(std::uint32_t total_objects,
                     std::uint64_t object_bytes)
    : total_objects_(total_objects),
      object_bytes_(object_bytes),
      free_mask_(total_objects, 1),
      num_free_(total_objects) {
  if (total_objects == 0) {
    throw std::invalid_argument("StatePool: total_objects must be > 0");
  }
  for (std::uint32_t id = 0; id < total_objects_; ++id) {
    free_ids_.push_back(id);
  }
  free_ids_offset_ = 0;
}

common::StateRef StatePool::Acquire(common::TierId tier,
                                    std::uint32_t group) {
  std::lock_guard<std::mutex> lock(mu_);
  if (num_free_ == 0) {
    throw NotEnoughSpaceError("StatePool: no free state objects");
  }
  if (free_ids_offset_ >= free_ids_.size()) {
    free_ids_.clear();
    for (std::uint32_t id = 0; id < total_objects_; ++id) {
      if (free_mask_[id] != 0) {
        free_ids_.push_back(id);
      }
    }
    free_ids_offset_ = 0;
  }
  const std::uint32_t object = free_ids_[free_ids_offset_++];
  free_mask_[object] = 0;
  --num_free_;
  return common::StateRef{tier, group, object, /*refcount=*/0};
}

void StatePool::Release(const common::StateRef& ref) {
  std::lock_guard<std::mutex> lock(mu_);
  if (ref.object >= total_objects_) {
    throw LogicError("StatePool: object id " + std::to_string(ref.object) +
                     " out of range");
  }
  if (free_mask_[ref.object] != 0) {
    throw LogicError("StatePool: cannot release free object " +
                     std::to_string(ref.object) + " repeatedly");
  }
  free_mask_[ref.object] = 1;
  ++num_free_;
}

std::uint32_t StatePool::num_free() const {
  std::lock_guard<std::mutex> lock(mu_);
  return num_free_;
}

}  // namespace stratakv::metadata
