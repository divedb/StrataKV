cmake_minimum_required(VERSION 3.22)

project(stratakv
  VERSION 0.1.0
  DESCRIPTION "StrataKV: a high-performance C++20 KV cache runtime for LLM inference"
  LANGUAGES CXX)

# --- Options (all target-scoped effects; no global compiler flags) ---
option(STRATAKV_GPU "Enable GPU-dependent components (Phase E and later)" OFF)
option(STRATAKV_PYTHON "Build pybind11 Python bindings" OFF)
option(STRATAKV_BUILD_TESTS "Build unit tests" ON)
option(STRATAKV_ENABLE_SANITIZERS "Enable ASan+UBSan on CPU targets" OFF)
option(STRATAKV_GTEST_SHIM
  "Offline bootstrap: use the bundled minimal GoogleTest-compatible shim
   instead of the pinned googletest submodule. CI must use the submodule."
  OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(CompilerWarnings)
include(Sanitizers)
include(Xxhash)

add_subdirectory(src)

if (STRATAKV_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests/cpp)
endif()

option(STRATAKV_BUILD_BENCHMARKS "Build benchmark executables" ON)
if (STRATAKV_BUILD_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

if (STRATAKV_PYTHON)
  add_subdirectory(bindings/python)
endif()

# --- Install / export ---
include(GNUInstallDirs)
configure_file(cmake/stratakvConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/stratakvConfig.cmake" @ONLY)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/stratakvConfig.cmake"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/stratakv")
install(EXPORT stratakv-targets
  NAMESPACE stratakv::
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/stratakv")
