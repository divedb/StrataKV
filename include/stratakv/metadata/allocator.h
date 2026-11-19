#pragma once

#include <cstdint>
#include <mutex>
#include <span>
#include <vector>

#include "stratakv/common/handles.h"

namespace stratakv::metadata {

/// \brief Block-id free-list for one tier (C++ port of FlexKV `Mempool`).
///
/// Harvested semantics: batch allocation walks a materialized free-id
/// array with a moving offset (re-materialized only when exhausted),
/// making Allocate amortized O(1) per id; Release marks ids free and
/// rejects double-recycles loudly. Ids are tier-local; the allocator
/// never touches memory, only the id space (the data plane owns bytes,
/// Phase E). Thread safety: internally locked; safe from any engine
/// thread.
class BlockAllocator {
 public:
  /// \param total_blocks Size of the id space [0, total_blocks).
  ///
  /// \throws std::invalid_argument if total_blocks == 0.
  explicit BlockAllocator(std::uint32_t total_blocks);

  /// \brief Allocate n free ids.
  ///
  /// \throws NotEnoughSpaceError if n > num_free().
  /// \throws std::invalid_argument if n == 0.
  std::vector<common::BlockId> Allocate(std::uint32_t n);

  /// \brief Return ids to the pool.
  ///
  /// \throws LogicError if any id is out of range or already free
  /// (double-recycle is an error, never a silent drop).
  void Release(std::span<const common::BlockId> ids);

  /// \brief Free ids remaining in the pool.
  std::uint32_t num_free() const;

  /// \brief Ids currently handed out.
  std::uint32_t num_used() const {
    return total_blocks_ - num_free();
  }

  /// \brief Capacity handed in at construction.
  std::uint32_t total_blocks() const { return total_blocks_; }

 private:
  void MaterializeFreeIds();

  std::uint32_t total_blocks_;
  std::vector<std::uint8_t> free_mask_;  ///< 1 = free (Mempool numpy mask).
  std::uint32_t num_free_ = 0;
  std::vector<common::BlockId> free_ids_;
  std::size_t free_ids_offset_ = 0;
  mutable std::mutex mu_;
};

/// \brief Fixed-size state-object pool for LinearState sidecars (ADR
/// 0003).
///
/// Metadata-only pool of object ids (bytes belong to the tier's data
/// plane, Phase E). Objects are acquired per LinearState group and
/// referenced by StateRef; refcount pinning is driven by the index, not
/// the pool. Thread safety: internally locked.
class StatePool {
 public:
  /// \param total_objects Size of the object id space [0, total_objects).
  /// \param object_bytes Byte size of one state object (accounting only;
  /// no memory is touched at this phase).
  ///
  /// \throws std::invalid_argument if total_objects == 0.
  StatePool(std::uint32_t total_objects, std::uint64_t object_bytes);

  /// \brief Acquire one free object for a group.
  ///
  /// \param tier Tier the object belongs to.
  /// \param group ModelLayout index of the LinearState group.
  /// \throws NotEnoughSpaceError if the pool is exhausted.
  common::StateRef Acquire(common::TierId tier, std::uint32_t group);

  /// \brief Return an object to the pool.
  ///
  /// \throws LogicError if the ref does not identify a currently-held
  /// object of this pool (wrong group, out of range, or double-release).
  void Release(const common::StateRef& ref);

  std::uint32_t num_free() const;
  std::uint32_t total_objects() const { return total_objects_; }
  std::uint64_t object_bytes() const { return object_bytes_; }

 private:
  std::uint32_t total_objects_;
  std::uint64_t object_bytes_;
  std::vector<std::uint8_t> free_mask_;
  std::vector<common::BlockId> free_ids_;  ///< Object ids (same id space).
  std::size_t free_ids_offset_ = 0;
  std::uint32_t num_free_ = 0;
  mutable std::mutex mu_;
};

}  // namespace stratakv::metadata
