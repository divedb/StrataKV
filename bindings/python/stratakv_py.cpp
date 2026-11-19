// stratakv_py.cpp — pybind11 module for the StrataKV native core.
//
// Phase B surface: build introspection only. The full Engine facade
// (match / submit / wait / cancel with GIL policy per ADR 0004) lands in
// Phase D; exposing it piecemeal before its semantics are pinned would
// create an unstable API.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>
#include <vector>

#include "stratakv/common/types.h"

namespace py = pybind11;

namespace {

constexpr char kVersion[] = "0.1.0";

}  // namespace

PYBIND11_MODULE(_core, m) {
  m.doc() = "StrataKV native core (Phase B: introspection surface).";

  m.attr("__version__") = kVersion;

  m.def("version", [] { return std::string(kVersion); },
        "Return the native core version string.");

  m.def("spec_names",
        [] {
          return std::vector<std::string>{
              stratakv::common::SpecName(stratakv::common::FullAttnSpec{}),
              stratakv::common::SpecName(stratakv::common::MlaSpec{}),
              stratakv::common::SpecName(
                  stratakv::common::SlidingWindowSpec{}),
              stratakv::common::SpecName(
                  stratakv::common::LinearStateSpec{}),
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
              stratakv::common::ToString(
                  stratakv::common::DeviceKind::kSsd),
          };
        },
        "Return the supported buffer device kinds.");
}
