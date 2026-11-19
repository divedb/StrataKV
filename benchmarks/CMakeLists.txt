# Phase C (C4): index match-latency benchmark. Executable only — NOT a
# ctest entry (latency numbers are read from the printed table per the
# benchmark policy in roadmap.md; hard gates would flake on shared CI
# runners). Run: ./build/benchmarks/stratakv_index_bench

add_executable(stratakv_index_bench index_bench.cpp)
stratakv_target_defaults(stratakv_index_bench)
stratakv_target_warnings(stratakv_index_bench)
stratakv_target_sanitizers(stratakv_index_bench)
target_link_libraries(stratakv_index_bench PRIVATE stratakv::metadata)

# Phase 0: GPU data-plane capability baseline (roadmap §5 policy; the
# record every later phase compares against). Requires STRATAKV_GPU=ON.
# Timing executable only, NOT a ctest entry, like index_bench above.
# Sanitizers are not applied: nvcc device TUs do not support ASan/UBSan.
if(STRATAKV_GPU)
  add_executable(stratakv_gpu_baseline gpu_baseline_bench.cu)
  stratakv_target_defaults(stratakv_gpu_baseline)
  target_compile_features(stratakv_gpu_baseline PRIVATE cxx_std_20)
  # Direct -std for CUDA TUs: keeps CMake 3.22 (project minimum) happy
  # with CUDA 12 toolchains it predates.
  target_compile_options(stratakv_gpu_baseline PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-std=c++20>
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wshadow -Wunused>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wshadow,-Wunused>)
  target_link_libraries(stratakv_gpu_baseline PRIVATE
    CUDA::cudart stratakv::xxhash)

  # Phase E (E3): full-stack per-task cost through TaskEngine ->
  # TransferScheduler -> CudaStreamTransport (acceptance-bar evidence).
  # Plain C++ TU; cudart needed directly (transport_cuda.h includes
  # <cuda_runtime.h>; stratakv_data's cudart link is PRIVATE).
  add_executable(stratakv_gpu_transport_bench gpu_transport_bench.cpp)
  stratakv_target_defaults(stratakv_gpu_transport_bench)
  stratakv_target_warnings(stratakv_gpu_transport_bench)
  # gcc-13 false positive in the vendored xxhash.h (XXH_INLINE_ALL);
  # same one-off suppression pattern as the metadata TUs.
  set_source_files_properties(gpu_transport_bench.cpp PROPERTIES
    COMPILE_OPTIONS "-Wno-array-bounds")
  # The isolated-mode bench run spawns the worker (E4).
  target_compile_definitions(stratakv_gpu_transport_bench PRIVATE
    STRATAKV_TEST_WORKER="${CMAKE_BINARY_DIR}/src/stratakv_transfer_worker")
  target_link_libraries(stratakv_gpu_transport_bench PRIVATE
    stratakv::control stratakv::data stratakv::xxhash CUDA::cudart)

  # Cross-process CUDA IPC feasibility probe (E4 prerequisite, ADR
  # 0002): environment documentation tool, not a ctest entry.
  add_executable(stratakv_ipc_probe ipc_probe.cu)
  target_compile_options(stratakv_ipc_probe PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-std=c++20 -Xcompiler=-Wall,-Wextra,-Wshadow,-Wunused>)
  target_link_libraries(stratakv_ipc_probe PRIVATE CUDA::cudart)
endif()
