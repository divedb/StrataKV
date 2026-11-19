#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "stratakv/common/types.h"

namespace stratakv::common {

/// \brief A group of model layers sharing one LayerSpec and one pool.
///
/// Layer indices are strictly increasing and owned by exactly one group;
/// a ModelLayout validates that the union over groups covers [0, N).
/// tokens_per_block must be > 0 for token-addressable groups and exactly 0
/// for LinearStateSpec groups.
struct LayerGroup {
  std::string name;  ///< Group name, unique within a ModelLayout.
  LayerSpec spec{};
  std::vector<int> layer_indices;  ///< Member layer indices, sorted.
  int tokens_per_block = 16;       ///< 0 for LinearStateSpec groups.

  /// \brief Structural validation per the invariants above.
  bool IsValid() const;
};

/// \brief Heterogeneous model description; the root of all layout math.
///
/// Immutable after construction. Replaces the single-tuple coupling of
/// FlexKV v1.0.0's ModelConfig (num_layers/num_kv_heads/head_size) with a
/// per-group table, so hybrid models are expressible from day one (ADR
/// 0003). Thread safety: safe to share across threads after construction.
class ModelLayout {
 public:
  /// \brief Build and validate a layout from layer groups.
  ///
  /// \param groups Non-empty set of valid groups whose layer indices
  /// densely cover [0, total_layers) with no gaps or overlaps, and whose
  /// names are unique.
  /// \throws std::invalid_argument describing the first violated invariant.
  static ModelLayout FromGroups(std::vector<LayerGroup> groups);

  /// \brief Access group by index (0..num_groups()-1).
  const LayerGroup& Group(std::size_t index) const;

  /// \brief Number of groups.
  std::size_t num_groups() const { return groups_.size(); }

  /// \brief Total layer count (dense union of all groups).
  int total_layers() const { return total_layers_; }

  /// \brief Group owning the given layer index, or nullptr if out of range.
  const LayerGroup* GroupForLayer(int layer_index) const;

  /// \brief True iff every group is FullAttn or Mla (classic cacheable
  /// model with no sidecar or window semantics).
  bool IsHomogeneousKV() const;

 private:
  ModelLayout() = default;
  std::vector<LayerGroup> groups_;
  std::vector<int> group_of_layer_;  ///< Group index per layer.
  int total_layers_ = 0;
};

}  // namespace stratakv::common
