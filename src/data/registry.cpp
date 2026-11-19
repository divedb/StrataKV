#include "stratakv/data/registry.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

#include "stratakv/common/exceptions.h"

namespace stratakv::data {

DeviceMemoryRegistry::RegistrationId
DeviceMemoryRegistry::RegisterGroup(GroupDescriptor descriptor) {
  std::lock_guard<std::mutex> lock(mu_);
  if (descriptor.group_name.empty()) {
    throw std::invalid_argument("RegisterGroup: name must be non-empty");
  }
  if (descriptor.device_id < 0) {
    throw std::invalid_argument("RegisterGroup: device_id must be >= 0");
  }
  if (descriptor.per_layer.empty()) {
    throw std::invalid_argument("RegisterGroup: no views given");
  }
  if (descriptor.layout == GroupLayout::kBlockwise &&
      descriptor.per_layer.size() != 1) {
    throw std::invalid_argument(
        "RegisterGroup: BLOCKWISE groups take exactly one view");
  }
  for (const common::BufferView& view : descriptor.per_layer) {
    if (!view.IsValid()) {
      throw std::invalid_argument(
          "RegisterGroup: invalid BufferView (null/zero/strides)");
    }
    if (view.kind != common::DeviceKind::kGpu) {
      throw std::invalid_argument(
          "RegisterGroup: views must point at GPU memory");
    }
    if (view.device_id != descriptor.device_id) {
      throw std::invalid_argument(
          "RegisterGroup: view device mismatch with descriptor");
    }
  }

  const RegistrationId id = next_id_++;
  Registration registration;
  registration.descriptor = std::move(descriptor);
  registration.live = true;

  auto existing = group_owner_.find(registration.descriptor.group);
  if (existing != group_owner_.end()) {
    const auto old = registrations_.find(existing->second);
    if (old != registrations_.end() && old->second.live) {
      throw std::invalid_argument(
          "RegisterGroup: group already registered (Unregister first)");
    }
  }
  group_owner_[registration.descriptor.group] = id;
  registrations_.emplace(id, std::move(registration));
  return id;
}

void DeviceMemoryRegistry::SetIpcHandleSource(
    std::shared_ptr<IIpcHandleSource> source) {
  std::lock_guard<std::mutex> lock(mu_);
  ipc_source_ = std::move(source);
}

std::vector<IpcHandle> DeviceMemoryRegistry::ExportIpcHandles(
    RegistrationId id) const {
  std::lock_guard<std::mutex> lock(mu_);
  auto it = registrations_.find(id);
  if (it == registrations_.end() || !it->second.live) {
    throw LogicError("ExportIpcHandles: unknown or unregistered id " +
                     std::to_string(id));
  }
  if (ipc_source_ == nullptr) {
    throw LogicError(
        "ExportIpcHandles: no IPC source installed — CPU-only build "
        "(in-process mode needs no IPC per ADR 0002); STRATAKV_GPU "
        "builds install the CUDA source");
  }
  std::vector<IpcHandle> handles;
  handles.reserve(it->second.descriptor.per_layer.size());
  for (const common::BufferView& view : it->second.descriptor.per_layer) {
    // The IPC payload stays pristine: validity is tracked registry-side
    // (Unregister clears the exported set — ADR 0005 immediate
    // invalidation), so real cudaIpcGetMemHandle bytes survive intact.
    IpcHandle handle = ipc_source_->Export(view);
    it->second.exported.push_back(handle);
    handles.push_back(handle);
  }
  return handles;
}

std::vector<common::BufferView> DeviceMemoryRegistry::ViewsOf(
    RegistrationId id) const {
  std::lock_guard<std::mutex> lock(mu_);
  auto it = registrations_.find(id);
  if (it == registrations_.end() || !it->second.live) {
    throw LogicError("ViewsOf: unknown or unregistered id " +
                     std::to_string(id));
  }
  return it->second.descriptor.per_layer;
}

DeviceMemoryRegistry::RegistrationId
DeviceMemoryRegistry::RegistrationForGroup(std::uint32_t group) const {
  std::lock_guard<std::mutex> lock(mu_);
  auto owner = group_owner_.find(group);
  if (owner == group_owner_.end()) {
    return 0;
  }
  auto it = registrations_.find(owner->second);
  if (it == registrations_.end() || !it->second.live) {
    return 0;
  }
  return owner->second;
}

bool DeviceMemoryRegistry::IsHandleValid(const IpcHandle& handle) const {
  std::lock_guard<std::mutex> lock(mu_);
  for (const auto& [id, registration] : registrations_) {
    if (!registration.live) continue;
    for (const IpcHandle& exported : registration.exported) {
      if (exported.opaque == handle.opaque) {
        return true;
      }
    }
  }
  return false;
}

void DeviceMemoryRegistry::Unregister(RegistrationId id) {
  std::lock_guard<std::mutex> lock(mu_);
  auto it = registrations_.find(id);
  if (it == registrations_.end() || !it->second.live) {
    throw LogicError("Unregister: unknown or unregistered id " +
                     std::to_string(id));
  }
  it->second.live = false;
  it->second.exported.clear();  // Every handle dies now (ADR 0005).
  auto owner = group_owner_.find(it->second.descriptor.group);
  if (owner != group_owner_.end() && owner->second == id) {
    group_owner_.erase(owner);
  }
}

std::size_t DeviceMemoryRegistry::num_registrations() const {
  std::lock_guard<std::mutex> lock(mu_);
  std::size_t live = 0;
  for (const auto& [id, registration] : registrations_) {
    (void)id;
    if (registration.live) ++live;
  }
  return live;
}

}  // namespace stratakv::data
