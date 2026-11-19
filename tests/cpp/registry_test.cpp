// Phase E (E1): DeviceMemoryRegistry tests — descriptor validation,
// the ADR 0005 ownership/lifetime contract (Unregister invalidates
// every exported handle immediately), in-process view access, and the
// fail-loud CPU-only behavior of IPC export (a CUDA IIpcHandleSource
// arrives with the STRATAKV_GPU build; the contract is exercised here
// against a test source).

#ifdef STRATAKV_GTEST_SHIM
#include "gtest_shim.h"
#else
#include <gtest/gtest.h>
#endif

#include <cstdint>
#include <memory>

#include "stratakv/common/exceptions.h"
#include "stratakv/common/model_layout.h"
#include "stratakv/data/registry.h"

namespace {

using stratakv::common::BufferView;
using stratakv::common::DeviceKind;
using stratakv::common::FullAttnSpec;
using stratakv::common::LayerGroup;
using stratakv::common::LinearStateSpec;
using stratakv::common::ModelLayout;
using stratakv::data::DeviceMemoryRegistry;
using stratakv::data::GroupDescriptor;
using stratakv::data::GroupLayout;
using stratakv::data::IIpcHandleSource;
using stratakv::data::IpcHandle;

ModelLayout Layout() {
  LayerGroup full;
  full.name = "full_attn";
  full.spec = FullAttnSpec{2, 64};
  full.layer_indices = {0, 1};
  full.tokens_per_block = 16;
  LayerGroup linear;
  linear.name = "linear_state";
  linear.spec = LinearStateSpec{4096};
  linear.layer_indices = {2, 3};
  linear.tokens_per_block = 0;
  return ModelLayout::FromGroups({full, linear});
}

BufferView GpuView(void* base, std::uint64_t bytes) {
  BufferView view;
  view.base = base;
  view.kind = DeviceKind::kGpu;
  view.device_id = 0;
  view.size_bytes = bytes;
  return view;
}

GroupDescriptor LayerwiseDescriptor() {
  GroupDescriptor descriptor;
  descriptor.group_name = "full_attn";
  descriptor.group = 0;
  descriptor.layout = GroupLayout::kLayerwise;
  descriptor.device_id = 0;
  descriptor.per_layer = {GpuView(reinterpret_cast<void*>(0x1000), 4096),
                          GpuView(reinterpret_cast<void*>(0x2000), 4096)};
  return descriptor;
}

/// \brief Test IPC source: payload counter (proves bytes stay pristine).
class CountingIpcSource final : public IIpcHandleSource {
 public:
  IpcHandle Export(const BufferView&) override {
    IpcHandle handle;
    handle.opaque[0] = static_cast<std::uint8_t>(++count_);
    handle.opaque[63] = 0xAB;
    return handle;
  }

  int count_ = 0;
};

TEST(DeviceMemoryRegistryTest, RegisterAndAccessViews) {
  DeviceMemoryRegistry registry;
  const DeviceMemoryRegistry::RegistrationId id =
      registry.RegisterGroup(LayerwiseDescriptor());
  EXPECT_NE(id, 0u);
  EXPECT_EQ(registry.num_registrations(), static_cast<std::size_t>(1));
  EXPECT_EQ(registry.RegistrationForGroup(0), id);

  const std::vector<BufferView> views = registry.ViewsOf(id);
  EXPECT_EQ(views.size(), static_cast<std::size_t>(2));
  EXPECT_EQ(views[0].size_bytes, 4096u);
}

TEST(DeviceMemoryRegistryTest, DescriptorValidation) {
  DeviceMemoryRegistry registry;
  GroupDescriptor bad = LayerwiseDescriptor();
  bad.group_name.clear();
  EXPECT_THROW((void)registry.RegisterGroup(bad), std::invalid_argument);

  bad = LayerwiseDescriptor();
  bad.layout = GroupLayout::kBlockwise;  // Two views for a flat pool.
  EXPECT_THROW((void)registry.RegisterGroup(bad), std::invalid_argument);

  bad = LayerwiseDescriptor();
  bad.device_id = -1;
  EXPECT_THROW((void)registry.RegisterGroup(bad), std::invalid_argument);

  bad = LayerwiseDescriptor();
  bad.per_layer[1].kind = DeviceKind::kCpu;  // Not GPU memory.
  EXPECT_THROW((void)registry.RegisterGroup(bad), std::invalid_argument);

  bad = LayerwiseDescriptor();
  bad.per_layer[1].base = nullptr;  // Invalid view.
  EXPECT_THROW((void)registry.RegisterGroup(bad), std::invalid_argument);

  EXPECT_EQ(registry.num_registrations(), static_cast<std::size_t>(0));
}

TEST(DeviceMemoryRegistryTest, GroupIsSingleRegistered) {
  DeviceMemoryRegistry registry;
  (void)registry.RegisterGroup(LayerwiseDescriptor());
  EXPECT_THROW((void)registry.RegisterGroup(LayerwiseDescriptor()),
               std::invalid_argument);
}

TEST(DeviceMemoryRegistryTest, IpcExportFailsLoudWithoutSource) {
  DeviceMemoryRegistry registry;
  const auto id = registry.RegisterGroup(LayerwiseDescriptor());
  // CPU-only build: no CUDA source installed — must throw, never
  // return fabricated handles (no silent stubs).
  EXPECT_THROW((void)registry.ExportIpcHandles(id), stratakv::LogicError);
}

TEST(DeviceMemoryRegistryTest, IpcContractAndImmediateInvalidation) {
  DeviceMemoryRegistry registry;
  auto source = std::make_shared<CountingIpcSource>();
  registry.SetIpcHandleSource(source);
  const auto id = registry.RegisterGroup(LayerwiseDescriptor());

  const std::vector<IpcHandle> handles = registry.ExportIpcHandles(id);
  ASSERT_EQ(handles.size(), static_cast<std::size_t>(2));
  EXPECT_EQ(source->count_, 2);
  // Payloads stay pristine (registry tracks validity by copy, not by
  // embedding headers).
  EXPECT_EQ(handles[0].opaque[0], 1);
  EXPECT_EQ(handles[0].opaque[63], 0xAB);
  EXPECT_TRUE(registry.IsHandleValid(handles[0]));
  EXPECT_TRUE(registry.IsHandleValid(handles[1]));

  // ADR 0005: Unregister invalidates every exported handle
  // IMMEDIATELY and is the engine's responsibility.
  registry.Unregister(id);
  EXPECT_FALSE(registry.IsHandleValid(handles[0]));
  EXPECT_FALSE(registry.IsHandleValid(handles[1]));
  EXPECT_EQ(registry.RegistrationForGroup(0), 0u);

  // Post-unregister: access throws; double unregister is an error.
  EXPECT_THROW((void)registry.ViewsOf(id), stratakv::LogicError);
  EXPECT_THROW((void)registry.ExportIpcHandles(id), stratakv::LogicError);
  EXPECT_THROW((void)registry.Unregister(id), stratakv::LogicError);

  // Re-registration of the same group is allowed after unregister.
  const auto again = registry.RegisterGroup(LayerwiseDescriptor());
  EXPECT_NE(again, id);
  EXPECT_EQ(registry.RegistrationForGroup(0), again);
  const std::vector<IpcHandle> fresh = registry.ExportIpcHandles(again);
  EXPECT_TRUE(registry.IsHandleValid(fresh[0]));
  EXPECT_FALSE(registry.IsHandleValid(handles[0]));  // Old still dead.
}

TEST(DeviceMemoryRegistryTest, RepeatedRegistrationInvalidatesOldHandles) {
  DeviceMemoryRegistry registry;
  auto source = std::make_shared<CountingIpcSource>();
  registry.SetIpcHandleSource(source);
  for (int iteration = 0; iteration < 100; ++iteration) {
    auto id = registry.RegisterGroup(LayerwiseDescriptor());
    auto handles = registry.ExportIpcHandles(id);
    EXPECT_TRUE(registry.IsHandleValid(handles[0]));
    registry.Unregister(id);
    EXPECT_FALSE(registry.IsHandleValid(handles[0]));
    EXPECT_EQ(registry.num_registrations(), 0u);
  }
}

}  // namespace
