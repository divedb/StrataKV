#pragma once

#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "stratakv/common/handles.h"

namespace stratakv::control {

/// \brief One data movement composing a task (design.md Phase D).
///
/// Ops address storage exclusively through tier-local ids (metadata
/// plane refs); no pointers, no torch types — the data plane resolves
/// ids to buffers via the registry (Phase E).
struct TransferOp {
  enum class Kind : std::uint8_t {
    kGpuToCpu,
    kCpuToGpu,
    kStateSnapshot,  ///< GPU state object -> tier state slot (put side).
    kStateRestore,   ///< Tier state slot -> GPU state object (get side).
    kSsdToCpu,
    kCpuToSsd,
  };

  Kind kind{};
  common::TierId src_tier = 0;
  common::TierId dst_tier = 0;
  std::uint32_t src_group = 0;  ///< ModelLayout group index.
  std::uint32_t dst_group = 0;
  /// \brief Source tier-local block ids (token-addressable ops).
  std::vector<common::BlockId> src_blocks;
  /// \brief Destination tier-local block ids, parallel to src_blocks.
  std::vector<common::BlockId> dst_blocks;
  std::uint32_t src_object = 0;  ///< State-op source object id.
  std::uint32_t dst_object = 0;  ///< State-op destination object id.
  /// \brief Engine-assigned completion id (opaque to builders; set by
  /// TaskEngine before execution and echoed by CompletionSink).
  std::uint64_t op_id = 0;
};

/// \brief DAG of transfer ops composing one task.
///
/// Thread safety: not thread-safe; owned by one builder then handed to
/// the engine (harvest of FlexKV's TransferOpGraph shape: a flat op list
/// plus explicit dependency edges; ops without incoming edges start
/// immediately).
class TransferGraph {
 public:
  /// \brief Append an op; returns its index.
  std::size_t AddOp(TransferOp op) {
    ops_.push_back(std::move(op));
    return ops_.size() - 1;
  }

  /// \brief Edge: op_index depends on depends_on (both must exist;
  /// dependencies may be declared before ops are appended via index
  /// returned by AddOp — indices are stable).
  ///
  /// \throws std::invalid_argument on self-dependency.
  void AddDependency(std::size_t op_index, std::size_t depends_on) {
    if (op_index == depends_on) {
      throw std::invalid_argument("TransferGraph: self-dependency");
    }
    dependencies_.emplace_back(op_index, depends_on);
  }

  std::span<const TransferOp> ops() const { return ops_; }
  /// \brief Dependency edges as (op_index, depends_on) pairs.
  std::span<const std::pair<std::size_t, std::size_t>> dependencies()
      const {
    return dependencies_;
  }

 private:
  std::vector<TransferOp> ops_;
  std::vector<std::pair<std::size_t, std::size_t>> dependencies_;
};

/// \brief Aggregate result of a Wait/TryWait batch.
struct WaitResult {
  struct Item {
    common::TaskHandle handle;
    common::TaskState state;
    std::string error;  ///< Populated on kFailed.
  };
  std::vector<Item> items;
};

}  // namespace stratakv::control
