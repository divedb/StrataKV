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

# --- GPU support (Phase E lineage; roadmap §2 E-environments) ---
# First CUDA consumer is the Phase 0 capability baseline benchmark
# (benchmarks/gpu_baseline_bench.cu). stratakv_data gains CUDA sources
# (transport, IPC source, pinning) when the STRATAKV_GPU build lands.
if(STRATAKV_GPU)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
      set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING
        "CUDA architectures to compile for")
    else()
      # CMake < 3.24 has no 'native'; set your GPU's SM version here.
      set(CMAKE_CUDA_ARCHITECTURES 89 CACHE STRING
        "CUDA architectures to compile for")
    endif()
  endif()
endif()

add_subdirectory(src)

if (STRATAKV_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests/cpp)
  if (STRATAKV_PYTHON)
    # Phase D acceptance: pytest drives the native engine from Python.
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    add_test(NAME stratakv_python_tests
      COMMAND ${CMAKE_COMMAND} -E env
        PYTHONPATH=${PROJECT_SOURCE_DIR}/python
        STRATAKV_REQUIRE_NATIVE=1
        ${Python3_EXECUTABLE} -m pytest ${PROJECT_SOURCE_DIR}/tests/python -q)
    # Phase F: connector suite (vLLM adapter against the fake-vLLM
    # harness; real bytes on STRATAKV_GPU builds, simulation on CPU).
    add_test(NAME stratakv_connector_tests
      COMMAND ${CMAKE_COMMAND} -E env
        PYTHONPATH=${PROJECT_SOURCE_DIR}/python:${PROJECT_SOURCE_DIR}/integrations/vllm/v0_17:${PROJECT_SOURCE_DIR}/integrations/vllm/tests
        STRATAKV_REQUIRE_NATIVE=1
        ${Python3_EXECUTABLE} -m pytest
        ${PROJECT_SOURCE_DIR}/integrations/vllm/tests -q)
  endif()
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
