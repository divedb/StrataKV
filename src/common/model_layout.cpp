#include "stratakv/common/model_layout.h"

#include <algorithm>
#include <set>
#include <stdexcept>

namespace stratakv::common {

bool LayerGroup::IsValid() const {
  if (name.empty()) return false;
  if (!::stratakv::common::IsValid(spec)) return false;
  if (layer_indices.empty()) return false;
  for (int index : layer_indices) {
    if (index < 0) return false;
  }
  // Strictly increasing implies sorted and unique.
  if (!std::is_sorted(layer_indices.begin(), layer_indices.end())) {
    return false;
  }
  if (std::adjacent_find(layer_indices.begin(), layer_indices.end()) !=
      layer_indices.end()) {
    return false;
  }
  if (IsTokenAddressable(spec)) {
    return tokens_per_block > 0;
  }
  return tokens_per_block == 0;  // LinearState groups are block-free.
}

ModelLayout ModelLayout::FromGroups(std::vector<LayerGroup> groups) {
  if (groups.empty()) {
    throw std::invalid_argument("ModelLayout: at least one group required");
  }
  std::set<std::string> names;
  std::set<int> layers;
  int max_layer = -1;
  for (const auto& group : groups) {
    if (!group.IsValid()) {
      throw std::invalid_argument("ModelLayout: group '" + group.name +
                                  "' is structurally invalid");
    }
    if (!names.insert(group.name).second) {
      throw std::invalid_argument("ModelLayout: duplicate group name '" +
                                  group.name + "'");
    }
    for (int index : group.layer_indices) {
      if (!layers.insert(index).second) {
        throw std::invalid_argument(
            "ModelLayout: layer " + std::to_string(index) +
            " claimed by more than one group");
      }
      max_layer = std::max(max_layer, index);
    }
  }
  // Dense coverage of [0, total_layers).
  const int total = max_layer + 1;
  if (static_cast<int>(layers.size()) != total) {
    throw std::invalid_argument(
        "ModelLayout: layer indices must densely cover [0, N); found " +
        std::to_string(layers.size()) + " of " + std::to_string(total));
  }

  ModelLayout layout;
  layout.groups_ = std::move(groups);
  layout.total_layers_ = total;
  layout.group_of_layer_.assign(static_cast<std::size_t>(total), -1);
  for (std::size_t g = 0; g < layout.groups_.size(); ++g) {
    for (int index : layout.groups_[g].layer_indices) {
      layout.group_of_layer_[static_cast<std::size_t>(index)] =
          static_cast<int>(g);
    }
  }
  return layout;
}

const LayerGroup& ModelLayout::Group(std::size_t index) const {
  if (index >= groups_.size()) {
    throw std::out_of_range("ModelLayout: group index out of range");
  }
  return groups_[index];
}

const LayerGroup* ModelLayout::GroupForLayer(int layer_index) const {
  if (layer_index < 0 ||
      layer_index >= static_cast<int>(group_of_layer_.size())) {
    return nullptr;
  }
  const int group = group_of_layer_[static_cast<std::size_t>(layer_index)];
  return group < 0 ? nullptr : &groups_[static_cast<std::size_t>(group)];
}

bool ModelLayout::IsHomogeneousKV() const {
  return std::all_of(groups_.begin(), groups_.end(), [](const LayerGroup& g) {
    return std::holds_alternative<FullAttnSpec>(g.spec) ||
           std::holds_alternative<MlaSpec>(g.spec);
  });
}

}  // namespace stratakv::common
