#include "stratakv/data/tier_table.h"

#include "stratakv/common/exceptions.h"

namespace stratakv::data {

void TierTable::Insert(common::TierId tier, std::uint32_t group,
                       GroupAddressing addressing) {
  const auto key = std::make_pair(tier, group);
  if (entries_.count(key) != 0) {
    throw LogicError("TierTable: duplicate entry for tier " +
                     std::to_string(tier) + " group " +
                     std::to_string(group));
  }
  entries_.emplace(key, std::move(addressing));
}

void TierTable::AddCpuTier(common::TierId tier,
                           const common::ModelLayout& layout,
                           const CpuTier& cpu_tier) {
  const bool sm_accessible =
      cpu_tier.backing() == HostBacking::kPinned;
  for (std::uint32_t group = 0; group < layout.num_groups(); ++group) {
    const common::LayerGroup& group_spec =
        layout.Group(static_cast<int>(group));
    GroupAddressing addressing;
    addressing.kind = cpu_tier.memory_kind();
    addressing.sm_accessible = sm_accessible;
    addressing.is_state =
        !stratakv::common::IsTokenAddressable(group_spec.spec);
    if (addressing.is_state) {
      addressing.bases = {cpu_tier.StateAddress(group, 0)};
      addressing.slice_bytes = cpu_tier.StateSlotBytes(group);
    } else {
      addressing.bases = {cpu_tier.BlockAddress(group, 0)};
      addressing.slice_bytes = cpu_tier.BlockSlotBytes(group);
    }
    addressing.block_stride = addressing.slice_bytes;  // Flat BLOCKWISE.
    Insert(tier, group, std::move(addressing));
  }
}

void TierTable::AddGpuTier(common::TierId tier,
                           const DeviceMemoryRegistry& registry,
                           DeviceMemoryRegistry::RegistrationId registration,
                           const common::ModelLayout& layout,
                           std::uint32_t capacity_slots) {
  if (capacity_slots == 0) {
    throw LogicError("TierTable: zero capacity is not addressable");
  }
  const std::vector<common::BufferView> views =
      registry.ViewsOf(registration);  // Throws LogicError when unknown.
  const GroupDescriptor descriptor =
      registry.DescriptorOf(registration);
  const common::LayerGroup& group_spec =
      layout.Group(static_cast<int>(descriptor.group));

  const std::size_t expected_views =
      descriptor.layout == GroupLayout::kLayerwise
          ? group_spec.layer_indices.size()
          : 1;
  if (views.size() != expected_views) {
    throw LogicError("TierTable: registration view count " +
                     std::to_string(views.size()) + " != layout's " +
                     std::to_string(expected_views) + " for group " +
                     std::to_string(descriptor.group));
  }
  const std::uint64_t view_bytes = views.front().size_bytes;
  for (const common::BufferView& view : views) {
    if (view.size_bytes != view_bytes) {
      throw LogicError("TierTable: registration views must be uniform");
    }
  }
  if (view_bytes % capacity_slots != 0) {
    throw LogicError(
        "TierTable: view size " + std::to_string(view_bytes) +
        " not divisible by capacity " + std::to_string(capacity_slots));
  }

  GroupAddressing addressing;
  addressing.bases.reserve(views.size());
  for (const common::BufferView& view : views) {
    addressing.bases.push_back(view.base);
  }
  addressing.block_stride = view_bytes / capacity_slots;
  addressing.slice_bytes = addressing.block_stride;
  addressing.kind = common::DeviceKind::kGpu;
  addressing.sm_accessible = true;
  addressing.is_state =
      !stratakv::common::IsTokenAddressable(group_spec.spec);
  Insert(tier, descriptor.group, std::move(addressing));
}

const GroupAddressing* TierTable::LookUp(common::TierId tier,
                                         std::uint32_t group) const {
  const auto it = entries_.find(std::make_pair(tier, group));
  return it == entries_.end() ? nullptr : &it->second;
}

}  // namespace stratakv::data
