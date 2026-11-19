# Phase C (C4): index match-latency benchmark. Executable only — NOT a
# ctest entry (latency numbers are read from the printed table per the
# benchmark policy in roadmap.md; hard gates would flake on shared CI
# runners). Run: ./build/benchmarks/stratakv_index_bench

add_executable(stratakv_index_bench index_bench.cpp)
stratakv_target_defaults(stratakv_index_bench)
stratakv_target_warnings(stratakv_index_bench)
stratakv_target_sanitizers(stratakv_index_bench)
target_link_libraries(stratakv_index_bench PRIVATE stratakv::metadata)
