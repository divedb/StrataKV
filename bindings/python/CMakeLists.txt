# Python bindings target. Requires the pinned pybind11 submodule
# (scripts/fetch_third_party.sh); there is intentionally no offline shim
# for bindings — a partial Python surface would violate the no-silent-
# stubs rule, so the target fails loudly when the dependency is absent.

if (NOT EXISTS "${PROJECT_SOURCE_DIR}/third_party/pybind11/CMakeLists.txt")
  message(FATAL_ERROR
    "STRATAKV_PYTHON=ON requires the pybind11 submodule under "
    "third_party/. Run scripts/fetch_third_party.sh (network required).")
endif()

add_subdirectory("${PROJECT_SOURCE_DIR}/third_party/pybind11"
  "${CMAKE_BINARY_DIR}/third_party/pybind11-build" EXCLUDE_FROM_ALL)

pybind11_add_module(stratakv_bindings stratakv_py.cpp)
set_target_properties(stratakv_bindings PROPERTIES OUTPUT_NAME "_core")

stratakv_target_defaults(stratakv_bindings)
stratakv_target_warnings(stratakv_bindings)
target_link_libraries(stratakv_bindings PRIVATE stratakv_core)
