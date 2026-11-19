// stratakv_py.cpp — pybind11 module for the StrataKV native core.
//
// Phase D surface (design.md Python facade): Engine with match /
// stage_put / submit_get / submit_put / try_wait / wait / cancel /
// shutdown. Phase E (GPU builds, STRATAKV_BINDINGS_GPU) adds the real
// data plane: use_gpu_transport=True defers engine start until the
// adapter has registered its GPU groups (register_gpu_group with raw
// descriptors — torch stays out of the core path by design), then
// start() wires CpuTier + DeviceMemoryRegistry + TierTable +
// CudaStreamTransport + TransferScheduler behind the TaskEngine.
// CPU-only builds keep the InlineExecutor simulation (no bytes moved)
// and fail loudly on the GPU options. GIL policy per ADR 0004:
//   - match():           holds the GIL (measured tens of µs budget;
//                        MatchForGet p50 53 µs at 1k blocks).
//   - wait():            releases the GIL while blocked in C++.
//   - submit_get/put():  releases the GIL (execution is queued, never
//                        run inline on the caller).
//   - start():           releases the GIL (device allocations).
//   - cancel/try_wait:   hold the GIL (non-blocking, fast).
// No native thread ever calls into Python (completion delivery is
// poll-based: completion_fd + drain_completions on the caller's
// thread). Debug device helpers (debug_alloc_device etc.) exist for
// tests/standalone use only — real adapters register engine-owned
// memory, never StrataKV allocations.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <chrono>
#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "stratakv/common/exceptions.h"
#include "stratakv/common/handles.h"
#include "stratakv/common/model_layout.h"
#include "stratakv/control/executor.h"
#include "stratakv/control/task_engine.h"
#include "stratakv/data/cpu_tier.h"
#include "stratakv/data/registry.h"
#include "stratakv/data/transport.h"
#include "stratakv/data/tier_table.h"
#include "stratakv/metadata/index.h"
#include "stratakv/metadata/metadata_plane.h"

#if defined(STRATAKV_BINDINGS_GPU)
#include <cuda_runtime.h>

#include "stratakv/data/transport_cuda.h"

using stratakv::data::CudaStreamTransport;
#endif

namespace py = pybind11;

namespace {

using stratakv::common::BlockId;
using stratakv::common::BufferView;
using stratakv::common::CacheEntry;
using stratakv::common::DeviceKind;
using stratakv::common::FullAttnSpec;
using stratakv::common::LayerGroup;
using stratakv::common::LayerSpec;
using stratakv::common::LinearStateSpec;
using stratakv::common::MlaSpec;
using stratakv::common::ModelLayout;
using stratakv::common::SlidingWindowSpec;
using stratakv::common::TaskHandle;
using stratakv::common::TaskState;
using stratakv::common::TokenId;
using stratakv::control::InlineExecutor;
using stratakv::control::ITaskExecutor;
using stratakv::control::TaskEngine;
using stratakv::control::TaskEngineConfig;
using stratakv::data::CpuTier;
using stratakv::data::CpuTierConfig;
using stratakv::data::DeviceMemoryRegistry;
using stratakv::data::GroupDescriptor;
using stratakv::data::GroupLayout;
using stratakv::data::HostBacking;
using stratakv::data::TierTable;
using stratakv::data::TransferScheduler;
using stratakv::data::TransportExecutor;
using stratakv::metadata::MatchResult;
using stratakv::metadata::MetadataPlane;
using stratakv::metadata::MetadataPlaneConfig;
using stratakv::metadata::RadixIndex;

constexpr char kVersion[] = "0.1.0";

/// \brief Kind-tagged layout group description (avoids binding the C++
/// LayerSpec variant; converted in BuildLayout).
struct PyLayoutGroup {
  std::string name;
  std::string kind = "full_attn";  // full_attn|mla|sliding_window|linear_state
  std::vector<int> layer_indices;
  int tokens_per_block = 16;
  int num_kv_heads = 0;
  int head_size = 0;
  int kv_lora_rank = 0;
  int rope_dim = 0;
  int window_blocks = 0;
  std::uint64_t state_bytes = 0;
};

struct PyEngineConfig {
  std::uint32_t blocks_per_group = 1u << 20;
  std::uint32_t state_objects_per_group = 1024;
  std::uint64_t state_object_bytes = 1u << 16;
  std::uint32_t max_in_flight = 1024;
  /// \brief Per-op simulated transfer latency (InlineExecutor); 0 =
  /// complete synchronously. Scheduler-side integration testing only —
  /// no data is moved. Ignored when use_gpu_transport.
  int inline_delay_ms = 0;
  /// \brief Phase E: construct the real data plane (CpuTier +
  /// CudaStreamTransport) instead of the InlineExecutor simulation.
  /// Requires a STRATAKV_GPU build of this module; the engine then
  /// requires register_gpu_group(...) + start() before submissions.
  bool use_gpu_transport = false;
  /// \brief CPU tier page-locking: "pageable" | "pinned" | "registered"
  /// (GPU transport only; "pinned" = cudaMallocHost is the only
  /// kernel-safe kind on WSL2 — benchmark F4).
  std::string cpu_backing = "pageable";
  /// \brief Transport stream-pool size (GPU transport only).
  std::uint32_t transport_streams = 2;
};

LayerSpec BuildSpec(const PyLayoutGroup& g) {
  if (g.kind == "full_attn") return FullAttnSpec{g.num_kv_heads, g.head_size};
  if (g.kind == "mla") return MlaSpec{g.kv_lora_rank, g.rope_dim};
  if (g.kind == "sliding_window") {
    return SlidingWindowSpec{g.num_kv_heads, g.head_size, g.window_blocks};
  }
  if (g.kind == "linear_state") return LinearStateSpec{g.state_bytes};
  throw std::invalid_argument("unknown group kind: " + g.kind);
}

ModelLayout BuildLayout(const std::vector<PyLayoutGroup>& groups) {
  if (groups.empty()) {
    throw std::invalid_argument("Engine: at least one layout group required");
  }
  std::vector<LayerGroup> parsed;
  parsed.reserve(groups.size());
  for (const PyLayoutGroup& g : groups) {
    LayerGroup group;
    group.name = g.name;
    group.spec = BuildSpec(g);
    group.layer_indices = g.layer_indices;
    group.tokens_per_block =
        g.kind == "linear_state" ? 0 : g.tokens_per_block;
    parsed.push_back(std::move(group));
  }
  return ModelLayout::FromGroups(std::move(parsed));
}

std::string StateName(TaskState state) {
  switch (state) {
    case TaskState::kStaged: return "staged";
    case TaskState::kLaunched: return "launched";
    case TaskState::kSucceeded: return "succeeded";
    case TaskState::kFailed: return "failed";
    case TaskState::kCancelled: return "cancelled";
  }
  return "unknown";
}

/// \brief Native engine facade held by Python.
class PyEngine {
 public:
  PyEngine(std::vector<PyLayoutGroup> groups, PyEngineConfig cfg)
      : layout_(std::make_unique<ModelLayout>(BuildLayout(groups))) {
    MetadataPlaneConfig plane_config;
    plane_config.blocks_per_group = cfg.blocks_per_group;
    plane_config.state_objects_per_group = cfg.state_objects_per_group;
    plane_config.state_object_bytes = cfg.state_object_bytes;
    config_ = cfg;

    auto index = std::make_unique<RadixIndex>(*layout_);
    plane_ = std::make_unique<MetadataPlane>(*layout_, std::move(index),
                                             plane_config);
    if (!cfg.use_gpu_transport) {
      // Phase D simulation path: ready immediately, no bytes move.
      StartEngine(std::make_shared<InlineExecutor>(
          std::chrono::milliseconds(cfg.inline_delay_ms)));
      return;
    }
#if !defined(STRATAKV_BINDINGS_GPU)
    throw stratakv::LogicError(
        "Engine: use_gpu_transport requires a STRATAKV_PYTHON build "
        "configured with -DSTRATAKV_GPU=ON (this module is CPU-only; "
        "the InlineExecutor simulation remains available)");
#else
    gpu_mode_ = true;  // Engine constructed by start().
#endif
  }

#if defined(STRATAKV_BINDINGS_GPU)
  /// \brief Register one group's engine-owned GPU buffers (ADR 0005:
  /// raw descriptors, never torch pointers). LAYERWISE: one (addr,
  /// bytes) per layer of the group, capacity per-base slots; BLOCKWISE:
  /// exactly one flat buffer. Only before start().
  std::uint64_t RegisterGpuGroup(const std::string& name,
                                 std::uint32_t group,
                                 const std::string& layout_kind,
                                 std::vector<std::pair<std::uint64_t,
                                                       std::uint64_t>>
                                     buffers,
                                 std::uint32_t capacity_blocks,
                                 int device_id) {
    if (!gpu_mode_) {
      throw stratakv::LogicError(
          "Engine: register_gpu_group requires use_gpu_transport=True");
    }
    if (engine_ != nullptr) {
      throw stratakv::LogicError("Engine: already started");
    }
    if (buffers.empty()) {
      throw std::invalid_argument("register_gpu_group: no buffers");
    }
    GroupLayout layout = layout_kind == "blockwise"
                             ? GroupLayout::kBlockwise
                             : GroupLayout::kLayerwise;
    if (layout_kind != "blockwise" && layout_kind != "layerwise") {
      throw std::invalid_argument(
          "register_gpu_group: layout must be 'layerwise' or 'blockwise'");
    }
    GroupDescriptor descriptor;
    descriptor.group_name = name;
    descriptor.group = group;
    descriptor.layout = layout;
    descriptor.device_id = device_id;
    for (const auto& [address, bytes] : buffers) {
      BufferView view;
      view.base = reinterpret_cast<void*>(address);
      view.kind = DeviceKind::kGpu;
      view.device_id = device_id;
      view.size_bytes = bytes;
      descriptor.per_layer.push_back(view);
    }
    const auto registration = registry_->RegisterGroup(
        std::move(descriptor));  // Validates layout/view contract.
    registrations_.push_back({registration, capacity_blocks});
    return registration;
  }

  /// \brief Build the data plane (CpuTier, TierTable, transport,
  /// scheduler) and the TaskEngine. Requires >= 1 GPU registration.
  void Start() {
    if (!gpu_mode_) {
      throw stratakv::LogicError(
          "Engine: start() requires use_gpu_transport=True");
    }
    if (engine_ != nullptr) {
      throw stratakv::LogicError("Engine: already started");
    }
    if (registrations_.empty()) {
      throw stratakv::LogicError(
          "Engine: start() requires at least one register_gpu_group "
          "(ops address the GPU tier)");
    }
    HostBacking backing = HostBacking::kPageable;
    if (config_.cpu_backing == "pinned") {
      backing = HostBacking::kPinned;
    } else if (config_.cpu_backing == "registered") {
      backing = HostBacking::kRegistered;
    } else if (config_.cpu_backing != "pageable") {
      throw std::invalid_argument(
          "EngineConfig.cpu_backing must be pageable|pinned|registered");
    }
    CpuTierConfig tier_config;
    tier_config.blocks_per_group = config_.blocks_per_group;
    tier_config.state_objects_per_group = config_.state_objects_per_group;
    tier_config.backing = backing;
    tier_ = std::make_unique<CpuTier>(*layout_, tier_config);

    TierTable table;
    table.AddCpuTier(0, *layout_, *tier_);
    for (const auto& [registration, capacity] : registrations_) {
      table.AddGpuTier(1, *registry_, registration, *layout_, capacity);
    }

    CudaStreamTransport::Config transport_config;
    transport_config.streams = config_.transport_streams;
    auto transport = std::make_unique<CudaStreamTransport>(
        std::move(table), transport_config);
    scheduler_ = std::make_shared<TransferScheduler>(
        std::move(transport), stratakv::data::IsolationMode::kInProcess,
        TransferScheduler::Config());
    StartEngine(std::make_shared<TransportExecutor>(scheduler_));
  }
#endif  // STRATAKV_BINDINGS_GPU

  // ---- match (holds the GIL; sub-ms budget) ----

  struct PyMatch {
    std::uint64_t match_id = 0;
    std::uint32_t matched_blocks = 0;
    std::uint32_t matched_blocks_reserved = 0;
    std::vector<std::pair<std::uint32_t, BlockId>> blocks;  // (group, id)
    std::vector<std::pair<std::uint32_t, std::uint32_t>> states;
    std::vector<std::uint8_t> token_mask;
  };

  PyMatch Match(std::vector<TokenId> tokens) {
    MatchResult result = plane_->MatchForGet(tokens);
    const std::uint64_t id = next_id_++;
    PyMatch out;
    out.match_id = id;
    out.matched_blocks = result.matched_blocks;
    out.matched_blocks_reserved = result.matched_blocks_reserved;
    for (const auto& ref : result.block_refs) {
      out.blocks.emplace_back(ref.group, ref.block);
    }
    for (const auto& ref : result.state_refs) {
      out.states.emplace_back(ref.group, ref.object);
    }
    out.token_mask = std::move(result.token_mask);
    matches_.emplace(id, std::move(result));
    return out;
  }

  /// \brief Drop a match's pins without loading (request abort).
  /// Idempotent per id (the entry stays registered; the underlying
  /// plane Release is itself idempotent).
  void ReleaseMatch(std::uint64_t match_id) {
    auto it = matches_.find(match_id);
    if (it == matches_.end()) {
      throw stratakv::LogicError("Engine: unknown match id " +
                                 std::to_string(match_id));
    }
    plane_->Release(it->second);
  }

  // ---- put staging ----

  struct PyPutPlan {
    std::uint64_t plan_id = 0;
    std::uint32_t matched_blocks = 0;
    std::uint32_t new_blocks = 0;
    bool feasible = true;
  };

  PyPutPlan StagePut(std::vector<TokenId> tokens) {
    MetadataPlane::PutPlan plan = plane_->StagePut(tokens);
    const std::uint64_t id = next_id_++;
    PyPutPlan out;
    out.plan_id = id;
    out.matched_blocks = plan.matched_blocks;
    out.new_blocks = plan.new_blocks;
    out.feasible = plan.feasible;
    plans_.emplace(id, std::move(plan));
    return out;
  }

  void AbortPut(std::uint64_t plan_id) {
    auto it = plans_.find(plan_id);
    if (it == plans_.end()) {
      throw stratakv::LogicError("Engine: unknown plan id " +
                                 std::to_string(plan_id));
    }
    if (it->second.entry != nullptr) {
      (void)plane_->AbortPut(it->second.entry);
    }
    plans_.erase(it);
  }

  // ---- task submission (GIL released: execution is queued, never
  // run inline on the calling thread under the real transport) ----

  /// \brief The started engine (submissions/waits need it; match and
  /// staging work pre-start — they only touch the metadata plane).
  TaskEngine& Eng() {
    if (engine_ == nullptr) {
      throw stratakv::LogicError(
          "Engine: not started — call start() after registering GPU "
          "groups (use_gpu_transport engines defer construction)");
    }
    return *engine_;
  }

  std::uint64_t SubmitGet(std::uint64_t match_id,
                          std::vector<BlockId> gpu_slots,
                          std::vector<std::uint32_t> gpu_state_slots) {
    auto it = matches_.find(match_id);
    if (it == matches_.end()) {
      throw stratakv::LogicError("Engine: unknown match id " +
                                 std::to_string(match_id));
    }
    // Erase only after a successful submit: a throw (e.g. pre-start)
    // must not strand the pinned match with no handle to release it.
    MatchResult match = it->second;
    const TaskHandle handle =
        Eng().SubmitGet(match, gpu_slots, gpu_state_slots);
    matches_.erase(it);
    return handle.id();
  }

  std::uint64_t SubmitPut(std::uint64_t plan_id,
                          std::vector<BlockId> gpu_blocks,
                          std::vector<std::uint32_t> gpu_state_slots) {
    auto it = plans_.find(plan_id);
    if (it == plans_.end()) {
      throw stratakv::LogicError("Engine: unknown plan id " +
                                 std::to_string(plan_id));
    }
    // Same contract as SubmitGet: a failed submit leaves the staged
    // plan abortable via abort_put.
    MetadataPlane::PutPlan plan = it->second;
    const TaskHandle handle =
        Eng().SubmitPut(plan, gpu_blocks, gpu_state_slots);
    plans_.erase(it);
    return handle.id();
  }

  // ---- task state (wait releases the GIL) ----

  using PyWaitItem = std::tuple<std::uint64_t, std::string, std::string>;

  std::vector<PyWaitItem> StatesOf(const std::vector<std::uint64_t>& ids) {
    std::vector<TaskHandle> handles;
    handles.reserve(ids.size());
    for (std::uint64_t id : ids) {
      handles.emplace_back(id);
    }
    const auto result = Eng().TryWait(handles);
    std::vector<PyWaitItem> out;
    out.reserve(result.items.size());
    for (const auto& item : result.items) {
      out.emplace_back(item.handle.id(), StateName(item.state),
                       item.error);
    }
    return out;
  }

  std::vector<PyWaitItem> Wait(const std::vector<std::uint64_t>& ids,
                               double timeout_ms) {
    std::vector<TaskHandle> handles;
    handles.reserve(ids.size());
    for (std::uint64_t id : ids) {
      handles.emplace_back(id);
    }
    const auto result = Eng().Wait(
        handles,
        std::chrono::milliseconds(static_cast<std::int64_t>(timeout_ms)));
    std::vector<PyWaitItem> out;
    out.reserve(result.items.size());
    for (const auto& item : result.items) {
      out.emplace_back(item.handle.id(), StateName(item.state),
                       item.error);
    }
    return out;
  }

  bool Cancel(std::uint64_t id) { return Eng().Cancel(TaskHandle(id)); }

  int CompletionFd() { return Eng().CompletionFd(); }

  void DrainCompletions() { Eng().DrainCompletions(); }

  void Shutdown(double grace_ms) {
    if (engine_ == nullptr) {
      return;  // Nothing was ever started.
    }
    engine_->Shutdown(
        std::chrono::milliseconds(static_cast<std::int64_t>(grace_ms)));
  }

  std::uint32_t NumInFlight() const {
    return engine_ == nullptr ? 0 : engine_->num_in_flight();
  }
  std::uint32_t FreeBlocks(std::uint32_t group) const {
    return plane_->num_free_blocks(group);
  }
  std::uint32_t FreeStates(std::uint32_t group) const {
    return plane_->num_free_states(group);
  }

  /// \brief LRU eviction (engine-driven pressure relief; also used by
  /// the benchmark's put loop). \return Blocks reclaimed.
  std::uint32_t EvictBlocks(std::uint32_t max_blocks) {
    return static_cast<std::uint32_t>(
        plane_->EvictBlocks(max_blocks).blocks.size());
  }

 private:
  /// \brief Construct the TaskEngine over `executor` (inline
  /// simulation or the real transport stack).
  void StartEngine(std::shared_ptr<ITaskExecutor> executor) {
    TaskEngineConfig engine_config;
    engine_config.max_in_flight = config_.max_in_flight;
    executor_ = std::move(executor);
    engine_ = std::make_unique<TaskEngine>(plane_.get(), executor_,
                                           engine_config);
  }

  PyEngineConfig config_;
  bool gpu_mode_ = false;
  // Declaration order fixes destruction order: the engine dies first,
  // then the executor/scheduler/transport stack, then the tier and
  // plane whose memory the transport touched.
  std::unique_ptr<ModelLayout> layout_;
  std::unique_ptr<MetadataPlane> plane_;
  std::unique_ptr<DeviceMemoryRegistry> registry_ =
      std::make_unique<DeviceMemoryRegistry>();
  struct GpuRegistration {
    DeviceMemoryRegistry::RegistrationId id;
    std::uint32_t capacity_blocks;
  };
  std::vector<GpuRegistration> registrations_;
  std::unique_ptr<CpuTier> tier_;
  std::shared_ptr<TransferScheduler> scheduler_;
  std::shared_ptr<ITaskExecutor> executor_;
  std::unique_ptr<TaskEngine> engine_;
  std::uint64_t next_id_ = 1;
  std::map<std::uint64_t, MatchResult> matches_;
  std::map<std::uint64_t, MetadataPlane::PutPlan> plans_;
};

}  // namespace

PYBIND11_MODULE(_core, m) {
  m.doc() = "StrataKV native core (Phase D: Engine facade).";
  m.attr("__version__") = kVersion;

  m.def("version", [] { return std::string(kVersion); },
        "Return the native core version string.");
  m.def("spec_names",
        [] {
          return std::vector<std::string>{
              stratakv::common::SpecName(FullAttnSpec{}),
              stratakv::common::SpecName(MlaSpec{}),
              stratakv::common::SpecName(SlidingWindowSpec{}),
              stratakv::common::SpecName(LinearStateSpec{}),
          };
        },
        "Return the supported per-group KV geometry kinds.");
#if defined(STRATAKV_BINDINGS_GPU)
  // Test/standalone device helpers ONLY (real adapters register
  // engine-owned memory; StrataKV never allocates GPU memory for the
  // engine — ADR 0005). Present == this module was built with
  // STRATAKV_GPU (feature marker for the Python facade).
  m.def("debug_alloc_device",
        [](std::uint64_t bytes) -> std::uint64_t {
          void* p = nullptr;
          cudaError_t err = cudaMalloc(&p, bytes);
          if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("debug_alloc_device: ") +
                cudaGetErrorString(err));
          }
          return reinterpret_cast<std::uint64_t>(p);
        },
        py::arg("bytes"), "cudaMalloc for tests (returns raw address).");
  m.def("debug_free_device",
        [](std::uint64_t address) {
          cudaError_t err = cudaFree(reinterpret_cast<void*>(address));
          if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("debug_free_device: ") +
                cudaGetErrorString(err));
          }
        },
        py::arg("address"), "cudaFree for tests.");
  m.def("debug_write_device",
        [](std::uint64_t address, const py::bytes& data) {
          const char* raw = PYBIND11_BYTES_AS_STRING(data.ptr());
          const ssize_t size = PYBIND11_BYTES_SIZE(data.ptr());
          if (raw == nullptr) {
            throw std::runtime_error("debug_write_device: bad bytes");
          }
          cudaError_t err = cudaMemcpy(reinterpret_cast<void*>(address),
                                       raw, static_cast<std::size_t>(size),
                                       cudaMemcpyHostToDevice);
          if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("debug_write_device: ") +
                cudaGetErrorString(err));
          }
        },
        py::arg("address"), py::arg("data"),
        "Host bytes -> device (tests; synchronous).");
  m.def("debug_read_device",
        [](std::uint64_t address, std::uint64_t bytes) -> py::bytes {
          std::string out;
          out.resize(bytes);
          cudaError_t err = cudaMemcpy(out.data(),
                                       reinterpret_cast<void*>(address),
                                       bytes, cudaMemcpyDeviceToHost);
          if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("debug_read_device: ") +
                cudaGetErrorString(err));
          }
          return py::bytes(out);
        },
        py::arg("address"), py::arg("bytes"),
        "Device bytes -> host (tests; synchronous).");
#endif  // STRATAKV_BINDINGS_GPU
  m.def("device_kinds",
        [] {
          return std::vector<std::string>{
              stratakv::common::ToString(stratakv::common::DeviceKind::kGpu),
              stratakv::common::ToString(
                  stratakv::common::DeviceKind::kCpuPinned),
              stratakv::common::ToString(stratakv::common::DeviceKind::kCpu),
              stratakv::common::ToString(stratakv::common::DeviceKind::kSsd),
          };
        },
        "Return the supported buffer device kinds.");

  py::register_exception<stratakv::NotEnoughSpaceError>(
      m, "NotEnoughSpaceError", PyExc_RuntimeError);
  py::register_exception<stratakv::LogicError>(m, "LogicError",
                                               PyExc_RuntimeError);

  py::class_<PyLayoutGroup>(
      m, "LayoutGroup",
      "Kind-tagged model layout group (converted to the native "
      "LayerSpec variant).")
      .def(py::init<>())
      .def_readwrite("name", &PyLayoutGroup::name)
      .def_readwrite("kind", &PyLayoutGroup::kind)
      .def_readwrite("layer_indices", &PyLayoutGroup::layer_indices)
      .def_readwrite("tokens_per_block", &PyLayoutGroup::tokens_per_block)
      .def_readwrite("num_kv_heads", &PyLayoutGroup::num_kv_heads)
      .def_readwrite("head_size", &PyLayoutGroup::head_size)
      .def_readwrite("kv_lora_rank", &PyLayoutGroup::kv_lora_rank)
      .def_readwrite("rope_dim", &PyLayoutGroup::rope_dim)
      .def_readwrite("window_blocks", &PyLayoutGroup::window_blocks)
      .def_readwrite("state_bytes", &PyLayoutGroup::state_bytes);

  py::class_<PyEngineConfig>(m, "EngineConfig")
      .def(py::init<>())
      .def_readwrite("blocks_per_group",
                     &PyEngineConfig::blocks_per_group)
      .def_readwrite("state_objects_per_group",
                     &PyEngineConfig::state_objects_per_group)
      .def_readwrite("state_object_bytes",
                     &PyEngineConfig::state_object_bytes)
      .def_readwrite("max_in_flight", &PyEngineConfig::max_in_flight)
      .def_readwrite("inline_delay_ms", &PyEngineConfig::inline_delay_ms)
      .def_readwrite("use_gpu_transport",
                     &PyEngineConfig::use_gpu_transport)
      .def_readwrite("cpu_backing", &PyEngineConfig::cpu_backing)
      .def_readwrite("transport_streams",
                     &PyEngineConfig::transport_streams);

  py::class_<PyEngine::PyMatch>(m, "MatchInfo")
      .def_readonly("match_id", &PyEngine::PyMatch::match_id)
      .def_readonly("matched_blocks",
                    &PyEngine::PyMatch::matched_blocks)
      .def_readonly("matched_blocks_reserved",
                    &PyEngine::PyMatch::matched_blocks_reserved)
      .def_readonly("blocks", &PyEngine::PyMatch::blocks)
      .def_readonly("states", &PyEngine::PyMatch::states)
      .def_readonly("token_mask", &PyEngine::PyMatch::token_mask);

  py::class_<PyEngine::PyPutPlan>(m, "PutPlan")
      .def_readonly("plan_id", &PyEngine::PyPutPlan::plan_id)
      .def_readonly("matched_blocks",
                    &PyEngine::PyPutPlan::matched_blocks)
      .def_readonly("new_blocks", &PyEngine::PyPutPlan::new_blocks)
      .def_readonly("feasible", &PyEngine::PyPutPlan::feasible);

  py::class_<PyEngine>(
      m, "Engine",
      "Native StrataKV engine: MetadataPlane + TaskEngine over either "
      "the InlineExecutor simulation (default; no bytes moved) or — "
      "with use_gpu_transport=True in a STRATAKV_GPU build — the real "
      "CpuTier + CudaStreamTransport data plane (register_gpu_group + "
      "start before submissions).")
      .def(py::init<std::vector<PyLayoutGroup>, PyEngineConfig>(),
           py::arg("groups"), py::arg("config") = PyEngineConfig{})
#if defined(STRATAKV_BINDINGS_GPU)
      .def("register_gpu_group", &PyEngine::RegisterGpuGroup,
           py::arg("name"), py::arg("group"), py::arg("layout"),
           py::arg("buffers"), py::arg("capacity_blocks"),
           py::arg("device_id") = 0,
           "Register one group's engine-owned GPU buffers as raw "
           "(address, bytes) descriptors — torch stays at the adapter "
           "(ADR 0005). 'layerwise': one buffer per layer; 'blockwise': "
           "one flat buffer. Only before start().")
      .def("start", &PyEngine::Start,
           py::call_guard<py::gil_scoped_release>(),
           "Build the data plane + TaskEngine (GIL released; device "
           "allocations). Requires >= 1 GPU registration.")
#endif
      .def("match", &PyEngine::Match, py::arg("token_ids"),
           "Pinned readiness-filtered match (GIL held; sub-ms budget). "
           "Returns MatchInfo; the pin lives until submit_get or "
           "release_match.")
      .def("release_match", &PyEngine::ReleaseMatch,
           py::arg("match_id"),
           "Drop a match's pins (request abort). Idempotent per id.")
      .def("plan_put", [](PyEngine& self, std::vector<TokenId> tokens) {
             return self.StagePut(std::move(tokens));
           },
           py::arg("token_ids"),
           "Read-only put feasibility preview.")
      .def("stage_put", &PyEngine::StagePut, py::arg("token_ids"),
           "Reserve CPU blocks/states and stage the not-ready entry.")
      .def("abort_put", &PyEngine::AbortPut, py::arg("plan_id"),
           "Abort a staged put; recycles its reservation.")
      .def("submit_get", &PyEngine::SubmitGet, py::arg("match_id"),
           py::arg("gpu_slots"), py::arg("gpu_state_slots") = std::vector<std::uint32_t>{},
           py::call_guard<py::gil_scoped_release>(),
           "Launch the load for a match (GIL released).")
      .def("submit_put", &PyEngine::SubmitPut, py::arg("plan_id"),
           py::arg("gpu_blocks"), py::arg("gpu_state_slots") = std::vector<std::uint32_t>{},
           py::call_guard<py::gil_scoped_release>(),
           "Launch the offload for a staged plan (GIL released).")
      .def("try_wait", &PyEngine::StatesOf, py::arg("task_ids"),
           "Non-blocking task-state poll (GIL held; fast).")
      .def("wait", &PyEngine::Wait, py::arg("task_ids"),
           py::arg("timeout_ms"),
           py::call_guard<py::gil_scoped_release>(),
           "Blocking wait with timeout; GIL released while blocked.")
      .def("cancel", &PyEngine::Cancel, py::arg("task_id"))
      .def("completion_fd", &PyEngine::CompletionFd,
           "Pollable fd signalled on task completion (ADR 0004).")
      .def("drain_completions", &PyEngine::DrainCompletions,
           "Clear the completion fd (callback delivery context).")
      .def("shutdown", &PyEngine::Shutdown, py::arg("grace_ms") = 100.0,
           py::call_guard<py::gil_scoped_release>(),
           "Cancel outstanding tasks and drain (GIL released).")
      .def("num_in_flight", &PyEngine::NumInFlight)
      .def("num_free_blocks", &PyEngine::FreeBlocks, py::arg("group"))
      .def("num_free_states", &PyEngine::FreeStates, py::arg("group"))
      .def("evict_blocks", &PyEngine::EvictBlocks, py::arg("max_blocks"),
           "Evict up to max_blocks LRU-first (engine-driven pressure "
           "relief); returns blocks reclaimed. Works pre-start.");
}
