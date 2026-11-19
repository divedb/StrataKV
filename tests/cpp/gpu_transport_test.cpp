// Phase E (GPU portion): CudaStreamTransport + tier backings + IPC
// source tests — real CUDA round trips through the full stack
// (TaskEngine -> TransportExecutor -> TransferScheduler ->
// CudaStreamTransport), byte-verified with XXH3 like the Phase 0
// benchmark. Built only under STRATAKV_GPU; each test skips at runtime
// when no usable device exists.

#ifdef STRATAKV_GTEST_SHIM
#include "gtest_shim.h"
#else
#include <gtest/gtest.h>
#endif

#include <cuda_runtime.h>

#include <signal.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "xxhash.h"

#include "stratakv/common/exceptions.h"
#include "stratakv/common/model_layout.h"
#include "stratakv/common/types.h"
#include "stratakv/control/task_engine.h"
#include "stratakv/data/cpu_tier.h"
#include "stratakv/data/ipc_transport.h"
#include "stratakv/data/registry.h"
#include "stratakv/data/tier_table.h"
#include "stratakv/data/transport.h"
#include "stratakv/data/transport_cuda.h"
#include "stratakv/metadata/metadata_plane.h"

namespace {

using stratakv::common::BlockId;
using stratakv::common::BufferView;
using stratakv::common::DeviceKind;
using stratakv::common::FullAttnSpec;
using stratakv::common::LayerGroup;
using stratakv::common::LinearStateSpec;
using stratakv::common::ModelLayout;
using stratakv::common::TaskHandle;
using stratakv::common::TaskState;
using stratakv::common::TokenId;
using stratakv::control::TaskEngine;
using stratakv::data::CpuTier;
using stratakv::data::CpuTierConfig;
using stratakv::data::CudaIpcHandleSource;
using stratakv::data::CudaStreamTransport;
using stratakv::data::DeviceMemoryRegistry;
using stratakv::data::GroupDescriptor;
using stratakv::data::GroupLayout;
using stratakv::data::HostBacking;
using stratakv::data::IpcTransportClient;
using stratakv::data::IsolationMode;
using stratakv::data::TierTable;
using stratakv::data::TransferScheduler;
using stratakv::data::TransportExecutor;
using stratakv::metadata::MetadataPlane;
using stratakv::metadata::MetadataPlaneConfig;
using stratakv::metadata::RadixIndex;

#define CUDA_TEST_CHECK(expr)                                            \
  do {                                                                   \
    cudaError_t err_ = (expr);                                           \
    ASSERT_EQ(err_, cudaSuccess) << #expr << ": "                        \
                                 << cudaGetErrorString(err_);            \
  } while (0)

bool GpuAvailable() {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices <= 0) {
    return false;
  }
  return true;
}

std::uint64_t SplitMix64(std::uint64_t& state) {
  state += 0x9E3779B97F4A7C15ULL;
  std::uint64_t z = state;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

/// \brief Qwen2-0.5B-shaped single-group layout: 24 layers x 8 KiB
/// per-layer block slice = 192 KiB model-wide block (the Phase 0 / E0
/// geometry).
ModelLayout Qwen2Shaped() {
  LayerGroup full;
  full.name = "full_attn";
  full.spec = FullAttnSpec{2, 64};
  full.layer_indices = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                        12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};
  full.tokens_per_block = 16;
  return ModelLayout::FromGroups({full});
}

/// \brief Hybrid layout (ADR 0003 shape): full attention + linear
/// state sidecars, 2 layers per group.
ModelLayout HybridShaped() {
  LayerGroup full;
  full.name = "full_attn";
  full.spec = FullAttnSpec{2, 64};
  full.layer_indices = {0, 2};
  full.tokens_per_block = 2;
  LayerGroup linear;
  linear.name = "linear_state";
  linear.spec = LinearStateSpec{8192};
  linear.layer_indices = {1, 3};
  linear.tokens_per_block = 0;
  return ModelLayout::FromGroups({full, linear});
}

/// \brief Engine-owned GPU memory stand-in: one device buffer per
/// layer of the group, LAYERWISE-registered (Appendix B heritage).
class GpuGroupBuffers {
 public:
  GpuGroupBuffers(std::size_t layers, std::uint64_t bytes_per_layer) {
    buffers_.resize(layers, nullptr);
    for (void*& buffer : buffers_) {
      const cudaError_t err = cudaMalloc(&buffer, bytes_per_layer);
      if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMalloc: ") +
                                 cudaGetErrorString(err));
      }
    }
  }
  ~GpuGroupBuffers() {
    for (void* buffer : buffers_) {
      (void)cudaFree(buffer);
    }
  }
  GpuGroupBuffers(const GpuGroupBuffers&) = delete;
  GpuGroupBuffers& operator=(const GpuGroupBuffers&) = delete;

  void* layer(std::size_t index) const { return buffers_.at(index); }

 private:
  std::vector<void*> buffers_;
};

constexpr std::uint64_t kPerLayer = 8192;  // BlockBytes / 24 layers.
constexpr std::uint64_t kSlotBytes = 24 * kPerLayer;

std::uint64_t HashBytes(const void* data, std::uint64_t bytes) {
  return XXH3_64bits(data, static_cast<std::size_t>(bytes));
}

TEST(GpuCpuTierTest, BackingsConstructAndReportKinds) {
  if (!GpuAvailable()) GTEST_SKIP() << "no CUDA device";
  const ModelLayout layout = HybridShaped();
  MetadataPlaneConfig plane_config;
  plane_config.blocks_per_group = 8;
  plane_config.state_objects_per_group = 4;

  CpuTierConfig pageable;
  pageable.blocks_per_group = 8;
  pageable.state_objects_per_group = 4;
  pageable.backing = HostBacking::kPageable;
  CpuTier pageable_tier(layout, pageable);
  EXPECT_EQ(pageable_tier.memory_kind(), DeviceKind::kCpu);
  EXPECT_FALSE(
      pageable_tier.PoolForGroup(0).kind == DeviceKind::kCpuPinned);

  CpuTierConfig pinned = pageable;
  pinned.backing = HostBacking::kPinned;
  CpuTier pinned_tier(layout, pinned);
  EXPECT_EQ(pinned_tier.memory_kind(), DeviceKind::kCpuPinned);
  EXPECT_EQ(pinned_tier.PoolForGroup(0).kind, DeviceKind::kCpuPinned);

  CpuTierConfig registered = pageable;
  registered.backing = HostBacking::kRegistered;
  try {
    CpuTier registered_tier(layout, registered);
    EXPECT_EQ(registered_tier.memory_kind(), DeviceKind::kCpuPinned);
  } catch (const std::runtime_error& e) {
    // Platform refused page registration (copy-engine transfers still
    // usable with the pageable tier); record and move on.
    SUCCEED() << "kRegistered backing unavailable here: " << e.what();
  }
}

/// \brief End-to-end block round trip (put D2H + get H2D) with the
/// pinned tier (kernel path) or pageable tier (copy-engine fallback).
void BlockRoundTrip(HostBacking backing) {
  if (!GpuAvailable()) GTEST_SKIP() << "no CUDA device";
  const ModelLayout layout = Qwen2Shaped();
  constexpr std::uint32_t kCapacity = 64;
  constexpr std::uint32_t kTpb = 16;

  MetadataPlaneConfig plane_config;
  plane_config.blocks_per_group = kCapacity;
  MetadataPlane plane(layout, std::make_unique<RadixIndex>(layout),
                      plane_config);

  CpuTierConfig tier_config;
  tier_config.blocks_per_group = kCapacity;
  tier_config.state_objects_per_group = 8;
  tier_config.backing = backing;
  CpuTier cpu_tier(layout, tier_config);

  GpuGroupBuffers gpu(24, kCapacity * kPerLayer);
  DeviceMemoryRegistry registry;
  GroupDescriptor descriptor;
  descriptor.group_name = "full_attn";
  descriptor.group = 0;
  descriptor.layout = GroupLayout::kLayerwise;
  descriptor.device_id = 0;
  for (std::size_t l = 0; l < 24; ++l) {
    BufferView view;
    view.base = gpu.layer(l);
    view.kind = DeviceKind::kGpu;
    view.device_id = 0;
    view.size_bytes = kCapacity * kPerLayer;
    descriptor.per_layer.push_back(view);
  }
  const auto registration = registry.RegisterGroup(std::move(descriptor));

  TierTable table;
  table.AddCpuTier(0, layout, cpu_tier);
  table.AddGpuTier(1, registry, registration, layout, kCapacity);

  CudaStreamTransport::Config transport_config;
  transport_config.streams = 2;
  auto transport = std::make_unique<CudaStreamTransport>(
      std::move(table), transport_config);
  CudaStreamTransport* transport_raw = transport.get();
  TransferScheduler::Config scheduler_config;
  scheduler_config.max_batch_ops = 64;
  scheduler_config.max_pending_tasks = 8;
  auto scheduler = std::make_shared<TransferScheduler>(
      std::move(transport), IsolationMode::kInProcess, scheduler_config);
  TaskEngine engine(&plane, std::make_shared<TransportExecutor>(scheduler));

  // Seed GPU slots 0..3 with a deterministic golden pattern (per-layer
  // uploads from one flat model-wide staging block).
  std::vector<std::uint64_t> staging(kSlotBytes / 8);
  std::vector<std::uint64_t> expected_hashes;
  for (std::uint32_t b = 0; b < 4; ++b) {
    std::uint64_t state = 0xC0FFEE + b;
    for (auto& word : staging) {
      word = SplitMix64(state);
    }
    expected_hashes.push_back(
        HashBytes(staging.data(), kSlotBytes));
    for (std::size_t l = 0; l < 24; ++l) {
      CUDA_TEST_CHECK(cudaMemcpy(
          static_cast<char*>(gpu.layer(l)) + b * kPerLayer,
          reinterpret_cast<const char*>(staging.data()) + l * kPerLayer,
          kPerLayer, cudaMemcpyHostToDevice));
    }
  }

  // Put: offload GPU blocks 0..3 into fresh CPU slots.
  std::vector<TokenId> tokens;
  for (std::uint32_t t = 0; t < 4 * kTpb; ++t) {
    tokens.push_back(static_cast<TokenId>(1000 + t));
  }
  const MetadataPlane::PutPlan plan = plane.StagePut(tokens);
  ASSERT_EQ(plan.blocks_to_store.size(), static_cast<std::size_t>(4));
  std::vector<BlockId> gpu_blocks = {0, 1, 2, 3};
  const TaskHandle put = engine.SubmitPut(plan, gpu_blocks);
  scheduler->Flush();
  ASSERT_EQ(engine.Wait(std::span<const TaskHandle>(&put, 1),
                        std::chrono::milliseconds(10000))
                .items.at(0)
                .state,
            TaskState::kSucceeded);
  for (std::size_t i = 0; i < 4; ++i) {
    EXPECT_EQ(HashBytes(cpu_tier.BlockAddress(0, plan.blocks_to_store[i]
                                                        .block),
                        kSlotBytes),
              expected_hashes[i]);
  }

  // Get: restore into GPU slots 20..23 from the CPU pool.
  const auto match = plane.MatchForGet(tokens);
  ASSERT_EQ(match.matched_blocks, 4u);
  std::vector<BlockId> gpu_slots = {20, 21, 22, 23};
  const TaskHandle get = engine.SubmitGet(match, gpu_slots);
  scheduler->Flush();
  ASSERT_EQ(engine.Wait(std::span<const TaskHandle>(&get, 1),
                        std::chrono::milliseconds(10000))
                .items.at(0)
                .state,
            TaskState::kSucceeded);
  for (std::uint32_t b = 0; b < 4; ++b) {
    std::vector<std::uint64_t> gather(kSlotBytes / 8);
    for (std::size_t l = 0; l < 24; ++l) {
      CUDA_TEST_CHECK(cudaMemcpy(
          reinterpret_cast<char*>(gather.data()) + l * kPerLayer,
          static_cast<const char*>(gpu.layer(l)) +
              static_cast<std::uint64_t>(20 + b) * kPerLayer,
          kPerLayer, cudaMemcpyDeviceToHost));
    }
    EXPECT_EQ(HashBytes(gather.data(), kSlotBytes), expected_hashes[b]);
  }

  engine.Shutdown(std::chrono::milliseconds(2000));
  scheduler->Shutdown(std::chrono::milliseconds(2000));
  EXPECT_EQ(transport_raw->num_in_flight(), static_cast<std::size_t>(0));
}

TEST(GpuTransportTest, BlockRoundTripPinnedKernelPath) {
  BlockRoundTrip(HostBacking::kPinned);
}

TEST(GpuTransportTest, BlockRoundTripPageableCopyEnginePath) {
  BlockRoundTrip(HostBacking::kPageable);
}

TEST(GpuTransportTest, HybridStateRoundTrip) {
  if (!GpuAvailable()) GTEST_SKIP() << "no CUDA device";
  const ModelLayout layout = HybridShaped();
  constexpr std::uint32_t kBlocks = 16;
  constexpr std::uint32_t kStates = 8;
  constexpr std::uint64_t kStateBytes = 2 * 8192;  // 2 layers x 8 KiB.

  MetadataPlaneConfig plane_config;
  plane_config.blocks_per_group = kBlocks;
  plane_config.state_objects_per_group = kStates;
  plane_config.state_object_bytes = kStateBytes;
  MetadataPlane plane(layout, std::make_unique<RadixIndex>(layout),
                      plane_config);

  CpuTierConfig tier_config;
  tier_config.blocks_per_group = kBlocks;
  tier_config.state_objects_per_group = kStates;
  tier_config.backing = HostBacking::kPinned;
  CpuTier cpu_tier(layout, tier_config);

  // GPU groups: full_attn (2 layer buffers) + linear_state (2 buffers).
  GpuGroupBuffers full_gpu(2, kBlocks * 1024);     // 1024 B/layer/block.
  GpuGroupBuffers state_gpu(2, kStates * kStateBytes / 2);
  DeviceMemoryRegistry registry;
  auto add_group = [&](std::uint32_t group, const char* name,
                       GpuGroupBuffers& buffers, std::uint32_t slots,
                       std::uint64_t slot_bytes) {
    GroupDescriptor descriptor;
    descriptor.group_name = name;
    descriptor.group = group;
    descriptor.layout = GroupLayout::kLayerwise;
    descriptor.device_id = 0;
    for (std::size_t l = 0; l < 2; ++l) {
      BufferView view;
      view.base = buffers.layer(l);
      view.kind = DeviceKind::kGpu;
      view.device_id = 0;
      view.size_bytes = slots * slot_bytes / 2;
      descriptor.per_layer.push_back(view);
    }
    return registry.RegisterGroup(std::move(descriptor));
  };
  const auto full_reg = add_group(0, "full_attn", full_gpu, kBlocks, 2048);
  const auto state_reg =
      add_group(1, "linear_state", state_gpu, kStates, kStateBytes);

  TierTable table;
  table.AddCpuTier(0, layout, cpu_tier);
  table.AddGpuTier(1, registry, full_reg, layout, kBlocks);
  table.AddGpuTier(1, registry, state_reg, layout, kStates);

  auto transport = std::make_unique<CudaStreamTransport>(std::move(table));
  auto scheduler = std::make_shared<TransferScheduler>(
      std::move(transport), IsolationMode::kInProcess,
      TransferScheduler::Config());
  TaskEngine engine(&plane, std::make_shared<TransportExecutor>(scheduler));

  // Golden state pattern in GPU state slots 0 (2 layers x 8 KiB).
  std::vector<std::uint64_t> golden(kStateBytes / 8);
  std::uint64_t rng = 42;
  for (auto& word : golden) word = SplitMix64(rng);
  const std::uint64_t golden_hash = HashBytes(golden.data(), kStateBytes);
  for (std::size_t l = 0; l < 2; ++l) {
    CUDA_TEST_CHECK(cudaMemcpy(
        state_gpu.layer(l),
        reinterpret_cast<const char*>(golden.data()) + l * (kStateBytes / 2),
        kStateBytes / 2, cudaMemcpyHostToDevice));
  }

  // Put with a state snapshot at the boundary: GPU state slot 0 ->
  // CPU state object (engine allocates via plan).
  const std::vector<TokenId> tokens = {1, 2, 3, 4};
  const MetadataPlane::PutPlan plan = plane.StagePut(tokens);
  ASSERT_FALSE(plan.states_to_snapshot.empty());
  const TaskHandle put = engine.SubmitPut(plan, std::vector<BlockId>{0, 1},
                                          std::vector<std::uint32_t>{0});
  scheduler->Flush();
  ASSERT_EQ(engine.Wait(std::span<const TaskHandle>(&put, 1),
                        std::chrono::milliseconds(10000))
                .items.at(0)
                .state,
            TaskState::kSucceeded);
  const std::uint32_t stored_object =
      plan.states_to_snapshot.front().object;
  EXPECT_EQ(HashBytes(cpu_tier.StateAddress(1, stored_object),
                      kStateBytes),
            golden_hash);

  // Get: restore the boundary state into GPU slot 5 + blocks 8..9.
  const auto match = plane.MatchForGet(tokens);
  ASSERT_EQ(match.matched_blocks, 2u);
  ASSERT_EQ(match.state_refs.size(), static_cast<std::size_t>(1));
  const TaskHandle get = engine.SubmitGet(match,
                                          std::vector<BlockId>{8, 9},
                                          std::vector<std::uint32_t>{5});
  scheduler->Flush();
  ASSERT_EQ(engine.Wait(std::span<const TaskHandle>(&get, 1),
                        std::chrono::milliseconds(10000))
                .items.at(0)
                .state,
            TaskState::kSucceeded);
  std::vector<std::uint64_t> gather(kStateBytes / 8);
  for (std::size_t l = 0; l < 2; ++l) {
    CUDA_TEST_CHECK(cudaMemcpy(
        reinterpret_cast<char*>(gather.data()) + l * (kStateBytes / 2),
        static_cast<const char*>(state_gpu.layer(l)) +
            5ull * (kStateBytes / 2),
        kStateBytes / 2, cudaMemcpyDeviceToHost));
  }
  EXPECT_EQ(HashBytes(gather.data(), kStateBytes), golden_hash);

  engine.Shutdown(std::chrono::milliseconds(2000));
  scheduler->Shutdown(std::chrono::milliseconds(2000));
}

TEST(GpuIpcTest, ExportSucceedsOrFailsLoudly) {
  if (!GpuAvailable()) GTEST_SKIP() << "no CUDA device";
  void* device_buffer = nullptr;
  CUDA_TEST_CHECK(cudaMalloc(&device_buffer, 1u << 20));
  DeviceMemoryRegistry registry;
  GroupDescriptor descriptor;
  descriptor.group_name = "full_attn";
  descriptor.group = 0;
  descriptor.layout = GroupLayout::kBlockwise;
  descriptor.device_id = 0;
  BufferView view;
  view.base = device_buffer;
  view.kind = DeviceKind::kGpu;
  view.device_id = 0;
  view.size_bytes = 1u << 20;
  descriptor.per_layer = {view};
  const auto registration = registry.RegisterGroup(std::move(descriptor));
  registry.SetIpcHandleSource(std::make_shared<CudaIpcHandleSource>());

  try {
    const std::vector<stratakv::data::IpcHandle> handles =
        registry.ExportIpcHandles(registration);
    ASSERT_EQ(handles.size(), static_cast<std::size_t>(1));
    EXPECT_TRUE(registry.IsHandleValid(handles.front()));
    SUCCEED() << "CUDA IPC export works on this platform";
  } catch (const stratakv::LogicError& e) {
    // Expected on platforms without CUDA IPC (WSL2): the source fails
    // loudly instead of stubbing.
    SUCCEED() << "CUDA IPC unsupported here (loud): " << e.what();
  }
  (void)cudaFree(device_buffer);
}

TEST(GpuTransportTest, UnknownTierFailsLoudly) {  if (!GpuAvailable()) GTEST_SKIP() << "no CUDA device";
  const ModelLayout layout = Qwen2Shaped();
  MetadataPlaneConfig plane_config;
  plane_config.blocks_per_group = 16;
  MetadataPlane plane(layout, std::make_unique<RadixIndex>(layout),
                      plane_config);
  CpuTierConfig tier_config;
  tier_config.blocks_per_group = 16;
  CpuTier cpu_tier(layout, tier_config);

  TierTable table;
  table.AddCpuTier(0, layout, cpu_tier);  // Tier 9 (GPU) never mapped.
  auto transport = std::make_unique<CudaStreamTransport>(std::move(table));
  auto scheduler = std::make_shared<TransferScheduler>(
      std::move(transport), IsolationMode::kInProcess,
      TransferScheduler::Config());
  TaskEngine engine(&plane, std::make_shared<TransportExecutor>(scheduler));

  const MetadataPlane::PutPlan plan =
      plane.StagePut(std::vector<TokenId>(32, TokenId{7}));  // 2 blocks
  const std::vector<BlockId> gpu_blocks = {0, 1};
  (void)engine.SubmitPut(plan, gpu_blocks);  // Queues fine (validation
  // is transport-side); the throw surfaces when the scheduler flushes
  // the graph into the transport's Enqueue.
  EXPECT_THROW(scheduler->Flush(), stratakv::LogicError);

  engine.Shutdown(std::chrono::milliseconds(1000));
  scheduler->Shutdown(std::chrono::milliseconds(1000));
}

// ---- Isolated-process mode (E4, ADR 0002) ----
// The engine process keeps metadata + GPU buffers; the worker owns
// the CPU tier bytes and the transport (GPU access via IPC handles).
// Skips on platforms without cross-process CUDA IPC (WSL2).

/// \brief Engine-side isolated stack bootstrap; nullptr + `skip`
/// message when this platform cannot do cross-process CUDA IPC.
std::unique_ptr<IpcTransportClient> TryMakeIsolatedClient(
    const ModelLayout& layout, std::uint32_t blocks_per_group,
    const GroupDescriptor& descriptor, std::uint32_t capacity,
    std::string* skip) {
#ifdef STRATAKV_TEST_WORKER
  // Test binaries live in a different dir than the worker.
  setenv("STRATAKV_TRANSFER_WORKER", STRATAKV_TEST_WORKER, /*overwrite=*/1);
#endif
  CpuTierConfig tier_config;
  tier_config.blocks_per_group = blocks_per_group;
  tier_config.state_objects_per_group = 8;
  tier_config.backing = HostBacking::kPinned;
  auto client = std::make_unique<IpcTransportClient>(layout, tier_config);
  try {
    client->RegisterGpuGroup(descriptor, capacity);
  } catch (const stratakv::LogicError& e) {
    if (IpcTransportClient::IsIpcUnsupported(e.what())) {
      *skip = e.what();
      return nullptr;
    }
    throw;
  }
  return client;
}

TEST(GpuIsolatedTest, BlockRoundTripThroughWorkerProcess) {
  if (!GpuAvailable()) GTEST_SKIP() << "no CUDA device";
  const ModelLayout layout = Qwen2Shaped();
  constexpr std::uint32_t kCapacity = 64;
  MetadataPlaneConfig plane_config;
  plane_config.blocks_per_group = kCapacity;
  MetadataPlane plane(layout, std::make_unique<RadixIndex>(layout),
                      plane_config);

  GpuGroupBuffers gpu(24, kCapacity * kPerLayer);
  GroupDescriptor descriptor;
  descriptor.group_name = "full_attn";
  descriptor.group = 0;
  descriptor.layout = GroupLayout::kLayerwise;
  descriptor.device_id = 0;
  for (std::size_t l = 0; l < 24; ++l) {
    BufferView view;
    view.base = gpu.layer(l);
    view.kind = DeviceKind::kGpu;
    view.device_id = 0;
    view.size_bytes = kCapacity * kPerLayer;
    descriptor.per_layer.push_back(view);
  }
  std::string skip_reason;
  auto client = TryMakeIsolatedClient(layout, kCapacity, descriptor,
                                      kCapacity, &skip_reason);
  if (client == nullptr) {
    GTEST_SKIP() << "cross-process CUDA IPC unsupported here: "
                 << skip_reason;
  }
  auto scheduler = std::make_shared<TransferScheduler>(
      std::move(client), IsolationMode::kIsolatedProcess,
      TransferScheduler::Config());
  TaskEngine engine(&plane, std::make_shared<TransportExecutor>(scheduler));

  // Golden pattern into GPU slots 0..3.
  std::vector<std::uint64_t> golden(kSlotBytes / 8);
  std::uint64_t rng = 0x1501D;
  for (auto& word : golden) {
    rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
    word = rng;
  }
  for (std::uint32_t b = 0; b < 4; ++b) {
    for (std::size_t l = 0; l < 24; ++l) {
      CUDA_TEST_CHECK(cudaMemcpy(
          static_cast<char*>(gpu.layer(l)) + b * kPerLayer,
          reinterpret_cast<const char*>(golden.data()) + l * kPerLayer,
          kPerLayer, cudaMemcpyHostToDevice));
    }
  }

  // Put through the worker; verify by loading back into GPU slots
  // 20..23 (the CPU bytes live in the worker — round-trip proves them).
  const std::vector<TokenId> tokens(64, TokenId{7});
  const auto plan = plane.StagePut(tokens);
  const std::vector<BlockId> src = {0, 1, 2, 3};
  const TaskHandle put = engine.SubmitPut(plan, src);
  ASSERT_EQ(engine.Wait(std::span<const TaskHandle>(&put, 1),
                        std::chrono::milliseconds(10000))
                .items.at(0)
                .state,
            TaskState::kSucceeded);
  const auto match = plane.MatchForGet(tokens);
  ASSERT_EQ(match.matched_blocks, 4u);
  std::vector<BlockId> dst = {20, 21, 22, 23};
  const TaskHandle get = engine.SubmitGet(match, dst);
  ASSERT_EQ(engine.Wait(std::span<const TaskHandle>(&get, 1),
                        std::chrono::milliseconds(10000))
                .items.at(0)
                .state,
            TaskState::kSucceeded);
  std::vector<std::uint64_t> gather(kSlotBytes / 8);
  for (std::uint32_t b = 0; b < 4; ++b) {
    for (std::size_t l = 0; l < 24; ++l) {
      CUDA_TEST_CHECK(cudaMemcpy(
          reinterpret_cast<char*>(gather.data()) + l * kPerLayer,
          static_cast<const char*>(gpu.layer(l)) +
              static_cast<std::uint64_t>(20 + b) * kPerLayer,
          kPerLayer, cudaMemcpyDeviceToHost));
    }
    ASSERT_EQ(HashBytes(gather.data(), kSlotBytes),
              HashBytes(golden.data(), kSlotBytes))
        << "block " << b << " corrupted through the worker round trip";
  }
  engine.Shutdown(std::chrono::milliseconds(2000));
  scheduler->Shutdown(std::chrono::milliseconds(2000));
}

TEST(GpuIsolatedTest, WorkerKillFailsOpsButEngineSurvives) {
  if (!GpuAvailable()) GTEST_SKIP() << "no CUDA device";
  const ModelLayout layout = Qwen2Shaped();
  constexpr std::uint32_t kCapacity = 64;
  constexpr std::uint32_t kPool = 8 * kCapacity;  // The whole batch
  // stages 8 x 64 blocks before anything completes.
  MetadataPlaneConfig plane_config;
  plane_config.blocks_per_group = kPool;
  MetadataPlane plane(layout, std::make_unique<RadixIndex>(layout),
                      plane_config);
  GpuGroupBuffers gpu(24, kCapacity * kPerLayer);
  GroupDescriptor descriptor;
  descriptor.group_name = "full_attn";
  descriptor.group = 0;
  descriptor.layout = GroupLayout::kLayerwise;
  descriptor.device_id = 0;
  for (std::size_t l = 0; l < 24; ++l) {
    BufferView view;
    view.base = gpu.layer(l);
    view.kind = DeviceKind::kGpu;
    view.device_id = 0;
    view.size_bytes = kCapacity * kPerLayer;
    descriptor.per_layer.push_back(view);
  }
  std::string skip_reason;
  auto client = TryMakeIsolatedClient(layout, kCapacity, descriptor,
                                      kCapacity, &skip_reason);
  if (client == nullptr) {
    GTEST_SKIP() << "cross-process CUDA IPC unsupported here: "
                 << skip_reason;
  }
  const pid_t worker_pid = client->worker_pid();
  auto scheduler = std::make_shared<TransferScheduler>(
      std::move(client), IsolationMode::kIsolatedProcess,
      TransferScheduler::Config());
  TaskEngine engine(&plane, std::make_shared<TransportExecutor>(scheduler));

  // A batch big enough to still be in flight when the worker dies
  // (8 x 64 blocks = ~100 MB through a ~12 GB/s link).
  std::vector<TaskHandle> puts;
  std::vector<MetadataPlane::PutPlan> plans;  // Kept for post-failure
  // aborts (the engine contract: failed puts stay staged for the
  // caller to abort; their reservations must return to the pool).
  std::vector<BlockId> big_src(64);
  std::iota(big_src.begin(), big_src.end(), BlockId{0});
  for (int task = 0; task < 8; ++task) {
    plans.push_back(plane.StagePut(
        std::vector<TokenId>(1024, TokenId{100 + task})));
    puts.push_back(engine.SubmitPut(plans.back(), big_src));
  }
  scheduler->Flush();
  std::this_thread::sleep_for(std::chrono::milliseconds(1));
  ASSERT_EQ(kill(worker_pid, SIGKILL), 0);

  // Every task must reach a terminal state (failures, not a hang).
  const auto result = engine.Wait(puts, std::chrono::milliseconds(5000));
  ASSERT_EQ(result.items.size(), puts.size());
  std::size_t failed = 0;
  for (const auto& item : result.items) {
    EXPECT_TRUE(item.state == TaskState::kFailed ||
                item.state == TaskState::kSucceeded);
    if (item.state == TaskState::kFailed) {
      ++failed;
    }
  }
  EXPECT_GT(failed, static_cast<std::size_t>(0));

  // The engine survives: metadata still works, the failed tasks'
  // reservations are recoverable (the documented abort path), a new
  // submission fails fast and loudly (inline op failure, no hang),
  // shutdown is clean.
  for (const auto& staged : plans) {
    if (staged.entry != nullptr) {
      (void)plane.AbortPut(staged.entry);
    }
  }
  const auto plan = plane.StagePut(std::vector<TokenId>(16, TokenId{9}));
  const TaskHandle doomed = engine.SubmitPut(plan, std::vector<BlockId>{0});  const auto quick = engine.Wait(std::span<const TaskHandle>(&doomed, 1),
                                 std::chrono::milliseconds(2000));
  EXPECT_EQ(quick.items.at(0).state, TaskState::kFailed);
  EXPECT_EQ(engine.num_in_flight(), 0u);
  engine.Shutdown(std::chrono::milliseconds(1000));
  scheduler->Shutdown(std::chrono::milliseconds(1000));
}

}  // namespace
