# Unit tests. Prefers the pinned googletest submodule; falls back to the
# documented offline bootstrap shim with -DSTRATAKV_GTEST_SHIM=ON; fails
# loudly otherwise (no silent stubs).

set(STRATAKV_UNIT_TEST_SOURCES
  types_test.cpp
  model_layout_test.cpp
  handles_test.cpp
  hasher_test.cpp
  allocator_test.cpp
  radix_index_test.cpp
  metadata_plane_test.cpp
  metadata_fuzz_test.cpp
  task_engine_test.cpp
  registry_test.cpp
  cpu_tier_test.cpp
  tier_table_test.cpp
  transfer_scheduler_test.cpp)

if (EXISTS "${PROJECT_SOURCE_DIR}/third_party/googletest/CMakeLists.txt")
  add_subdirectory("${PROJECT_SOURCE_DIR}/third_party/googletest"
    "${CMAKE_BINARY_DIR}/third_party/googletest-build" EXCLUDE_FROM_ALL)
  add_executable(stratakv_unit_tests ${STRATAKV_UNIT_TEST_SOURCES})
  target_link_libraries(stratakv_unit_tests PRIVATE stratakv_core
    stratakv_metadata stratakv::xxhash gtest gtest_main)
elseif (STRATAKV_GTEST_SHIM)
  message(STATUS "Using offline GoogleTest bootstrap shim (STRATAKV_GTEST_SHIM=ON)")
  add_executable(stratakv_unit_tests
    ${STRATAKV_UNIT_TEST_SOURCES} shim_main.cpp)
  target_include_directories(stratakv_unit_tests PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}")
  target_compile_definitions(stratakv_unit_tests PRIVATE STRATAKV_GTEST_SHIM=1)
else()
  message(FATAL_ERROR
    "googletest submodule not present under third_party/. Run "
    "scripts/fetch_third_party.sh (requires network) or reconfigure with "
    "-DSTRATAKV_GTEST_SHIM=ON for the offline bootstrap shim.")
endif()

stratakv_target_defaults(stratakv_unit_tests)
stratakv_target_warnings(stratakv_unit_tests)
stratakv_target_sanitizers(stratakv_unit_tests)
target_link_libraries(stratakv_unit_tests PRIVATE stratakv_core
  stratakv_metadata stratakv_control stratakv_data stratakv::xxhash)
# CPU-only builds must fail loudly on page-locked tier backings; the GPU
# build exercises them for real in stratakv_gpu_tests (below).
if (NOT STRATAKV_GPU)
  target_compile_definitions(stratakv_unit_tests PRIVATE STRATAKV_TESTS_CPU_ONLY)
endif()

# GPU tests (Phase E): real CUDA round trips through the TaskEngine ->
# TransferScheduler -> CudaStreamTransport stack, tier backings, and the
# IPC handle source. Requires STRATAKV_GPU; skips at runtime when no
# device is usable. Sanitizers are not applied (nvcc).
if (STRATAKV_GPU AND EXISTS "${PROJECT_SOURCE_DIR}/third_party/googletest/CMakeLists.txt")
  add_executable(stratakv_gpu_tests gpu_transport_test.cpp)
  stratakv_target_defaults(stratakv_gpu_tests)
  target_compile_features(stratakv_gpu_tests PRIVATE cxx_std_20)
  target_compile_options(stratakv_gpu_tests PRIVATE -Wall -Wextra -Wshadow -Wunused)
  target_link_libraries(stratakv_gpu_tests PRIVATE stratakv_core
    stratakv_metadata stratakv_control stratakv_data stratakv::xxhash
    CUDA::cudart gtest gtest_main)
  add_test(NAME stratakv_gpu_tests COMMAND stratakv_gpu_tests)
endif()

# hasher_test.cpp includes the vendored xxhash.h (one-shot golden
# digests); exempt just that cast warning there.
set_source_files_properties(hasher_test.cpp PROPERTIES
  COMPILE_OPTIONS "-Wno-old-style-cast")

add_test(NAME stratakv_unit_tests COMMAND stratakv_unit_tests)
