# Benchmarks

## index_bench (Phase C4 — metadata plane, CPU-only)

`stratakv_index_bench` measures RadixIndex/MetadataPlane match latency on
a Qwen3-Next-shaped hybrid layout (full_attn + linear_state,
tokens_per_block 16) at 1k-block scale. Not a ctest entry: latency tables
are read by humans per the benchmark policy in `roadmap.md` (hard gates
would flake on shared CI runners); the budget verdict line is printed on
every run.

Run:

```bash
cmake -S . -B build && cmake --build build -j
./build/benchmarks/stratakv_index_bench
```

Reference (Apple Silicon dev box, Release, 2026-11-19):

| scenario | p50 µs | p95 µs | p99 µs | matched |
| --- | --- | --- | --- | --- |
| deep_warm_hit_1k | 53.1 | 59.1 | 74.2 | 1024 |
| deep_miss_1k | 33.4 | 38.0 | 43.9 | 0 |
| deep_partial_512_of_1k | 38.5 | 43.7 | 50.4 | 512 |
| single_node_warm_hit_1k | 41.6 | 47.2 | 52.4 | 1024 |

Roadmap budget: p50 < 100 µs at 1k blocks — PASS with ~2× headroom.
`deep_partial_512_of_1k` requires a state-bundle boundary at 512 (ADR
0003); without one a hybrid partial match correctly truncates to 0.

## Phase A3 harness — not yet ported

The reproducible end-to-end benchmark harness currently lives with the
audited FlexKV environment: `~/Desktop/FlexKV/examples/vllm_adaption/serve_qwen2_clean.py`
plus the recorded baselines (`flexkv_clean_baseline.log`,
`docs/fluxkv/PHASE0_AUDIT.md`). It will be generalized into this
directory at Phase A3 with:

- pinned workload JSON (prompts, block counts, concurrency, hit rates);
- metric collectors: match p50/p95/p99, per-task fixed overhead,
  effective GB/s, TTFT/TPOT deltas, scheduler stall;
- environment capture (commits, drivers, clocks) in every report.
