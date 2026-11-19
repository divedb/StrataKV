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
  target_compile_features(stratakv_gpu_baseline PRIVATE
    cxx_std_20 cuda_std_20)
  target_compile_options(stratakv_gpu_baseline PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wshadow -Wunused>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wshadow,-Wunused>)
  target_link_libraries(stratakv_gpu_baseline PRIVATE
    CUDA::cudart stratakv::xxhash)
endif()
