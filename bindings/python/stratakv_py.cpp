// stratakv_py.cpp — pybind11 module for the StrataKV native core.
//
// Phase D surface (design.md Python facade): Engine with match /
// submit_get / submit_put / try_wait / wait / cancel / shutdown,
// driving MetadataPlane + TaskEngine through the InlineExecutor
// simulation transport (no bytes are moved; Phase E installs the CUDA
// transport). GIL policy per ADR 0004:
//   - match():           holds the GIL (measured tens of µs budget;
//                        MatchForGet p50 53 µs at 1k blocks).
//   - wait():            releases the GIL while blocked in C++.
//   - submit_get/put():  releases the GIL (inline execution may sleep).
//   - cancel/try_wait:   hold the GIL (non-blocking, fast).
// No native thread ever calls into Python (completion delivery is
// poll-based: completion_fd + drain_completions on the caller's
// thread).

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
#include "stratakv/metadata/index.h"
#include "stratakv/metadata/metadata_plane.h"

namespace py = pybind11;

namespace {

using stratakv::common::BlockId;
using stratakv::common::CacheEntry;
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
using stratakv::control::TaskEngine;
using stratakv::control::TaskEngineConfig;
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
  /// no data is moved.
  int inline_delay_ms = 0;
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

    auto delay = std::chrono::milliseconds(cfg.inline_delay_ms);
    executor_ = std::make_shared<InlineExecutor>(delay);
    auto index = std::make_unique<RadixIndex>(*layout_);
    plane_ = std::make_unique<MetadataPlane>(*layout_, std::move(index),
                                             plane_config);
    TaskEngineConfig engine_config;
    engine_config.max_in_flight = cfg.max_in_flight;
    engine_ = std::make_unique<TaskEngine>(plane_.get(), executor_,
                                           engine_config);
  }

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

  // ---- task submission (GIL released: inline execution may sleep) ----

  std::uint64_t SubmitGet(std::uint64_t match_id,
                          std::vector<BlockId> gpu_slots) {
    auto it = matches_.find(match_id);
    if (it == matches_.end()) {
      throw stratakv::LogicError("Engine: unknown match id " +
                                 std::to_string(match_id));
    }
    MatchResult match = std::move(it->second);
    matches_.erase(it);
    const TaskHandle handle = engine_->SubmitGet(match, gpu_slots);
    return handle.id();
  }

  std::uint64_t SubmitPut(std::uint64_t plan_id,
                          std::vector<BlockId> gpu_blocks) {
    auto it = plans_.find(plan_id);
    if (it == plans_.end()) {
      throw stratakv::LogicError("Engine: unknown plan id " +
                                 std::to_string(plan_id));
    }
    MetadataPlane::PutPlan plan = std::move(it->second);
    plans_.erase(it);
    const TaskHandle handle = engine_->SubmitPut(plan, gpu_blocks);
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
    const auto result = engine_->TryWait(handles);
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
    const auto result = engine_->Wait(
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

  bool Cancel(std::uint64_t id) { return engine_->Cancel(TaskHandle(id)); }

  int CompletionFd() const { return engine_->CompletionFd(); }

  void DrainCompletions() { engine_->DrainCompletions(); }

  void Shutdown(double grace_ms) {
    engine_->Shutdown(
        std::chrono::milliseconds(static_cast<std::int64_t>(grace_ms)));
  }

  std::uint32_t NumInFlight() const { return engine_->num_in_flight(); }
  std::uint32_t FreeBlocks(std::uint32_t group) const {
    return plane_->num_free_blocks(group);
  }
  std::uint32_t FreeStates(std::uint32_t group) const {
    return plane_->num_free_states(group);
  }

 private:
  std::unique_ptr<ModelLayout> layout_;
  std::shared_ptr<InlineExecutor> executor_;
  std::unique_ptr<MetadataPlane> plane_;
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
      .def_readwrite("inline_delay_ms", &PyEngineConfig::inline_delay_ms);

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
      "Native StrataKV engine: MetadataPlane + TaskEngine driven "
      "through the InlineExecutor simulation transport (Phase D; no "
      "bytes are moved).")
      .def(py::init<std::vector<PyLayoutGroup>, PyEngineConfig>(),
           py::arg("groups"), py::arg("config") = PyEngineConfig{})
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
           py::arg("gpu_slots"),
           py::call_guard<py::gil_scoped_release>(),
           "Launch the load for a match (GIL released).")
      .def("submit_put", &PyEngine::SubmitPut, py::arg("plan_id"),
           py::arg("gpu_blocks"),
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
      .def("num_free_states", &PyEngine::FreeStates, py::arg("group"));
}
