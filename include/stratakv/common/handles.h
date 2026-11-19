#pragma once

#include <cstdint>
#include <vector>

#include "stratakv/common/model_layout.h"
#include "stratakv/common/types.h"

namespace stratakv::common {

/// \brief Task lifecycle states (control plane state machine, Phase D).
enum class TaskState {
  kStaged,     ///< Created, waiting for launch.
  kLaunched,   ///< Submitted to the data plane.
  kSucceeded,  ///< Terminal: transfer completed.
  kFailed,     ///< Terminal: transfer errored.
  kCancelled,  ///< Terminal: cancelled before or during execution.
};

/// \brief True for terminal states (Succeeded/Failed/Cancelled).
bool IsTerminal(TaskState state);

/// \brief Opaque, copyable handle to a submitted task.
///
/// Thread safety: value type; safe to copy across threads. Validity is
/// lazily checked against the owning TaskEngine (Phase D); a default
/// constructed handle is invalid.
class TaskHandle {
 public:
  /// \brief Default construction yields an invalid handle (id 0).
  TaskHandle() = default;

  /// \brief Construct from an engine-assigned id (non-zero).
  explicit TaskHandle(std::uint64_t id) : id_(id) {}

  /// \brief Engine-assigned identifier; 0 when invalid.
  std::uint64_t id() const { return id_; }

  /// \brief False only for default-constructed handles.
  bool IsValid() const { return id_ != 0; }

  friend bool operator==(const TaskHandle& a, const TaskHandle& b) {
    return a.id_ == b.id_;
  }
  friend bool operator!=(const TaskHandle& a, const TaskHandle& b) {
    return !(a == b);
  }

 private:
  std::uint64_t id_ = 0;
};

/// \brief Reference to one token-block slot inside a tier's pool.
///
/// `group` indexes the ModelLayout group the block stores data for. The
/// refcount pins the block against eviction while > 0.
struct BlockRef {
  TierId tier = 0;         ///< Owning tier.
  std::uint32_t group = 0; ///< ModelLayout group index.
  BlockId block = 0;       ///< Tier-local block id.
  std::uint32_t refcount = 0;
};

/// \brief Reference to one fixed-size state object (sidecar) in a tier.
struct StateRef {
  TierId tier = 0;
  std::uint32_t group = 0;  ///< LinearState group index.
  std::uint32_t object = 0; ///< Tier-local state-object id.
  std::uint32_t refcount = 0;
};

/// \brief Everything needed to resume a prefix at one boundary (ADR 0003).
///
/// A CacheEntry is loadable only when is_ready is set AND every group of
/// the model is covered (blocks for token-addressable groups, states for
/// LinearState groups) — the no-partial-caching invariant. Checking
/// per-group block-count consistency against a prefix length is the
/// metadata plane's responsibility (Phase C); here we validate coverage.
struct CacheEntry {
  std::vector<BlockRef> blocks;  ///< Token blocks, prefix order.
  std::vector<StateRef> states;  ///< Sidecar states at this boundary.
  bool is_ready = false;         ///< Data landed in its tier.

  /// \brief Validate the no-partial-caching invariant against a layout.
  ///
  /// \param layout Model description whose groups must each be covered.
  /// \return True iff is_ready, all ref group indices are in range, every
  /// token-addressable group has >= 1 block, and every LinearState group
  /// has >= 1 state.
  bool IsCompleteFor(const ModelLayout& layout) const;
};

}  // namespace stratakv::common
