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

Reference numbers:

| environment | deep_warm_hit_1k p50 µs | deep_miss_1k p50 µs | log |
| --- | --- | --- | --- |
| Apple Silicon dev box, Release, 2026-11-19 | 53.1 | 33.4 | (session record) |
| WSL2 / RTX 4080 SUPER, Release, 2026-11-19 | 26.0 | 22.1 | `results/2026-11-19_wsl2_rtx4080super/index_bench.log` |

Roadmap budget: p50 < 100 µs at 1k blocks — PASS with headroom on both.
`deep_partial_512_of_1k` requires a state-bundle boundary at 512 (ADR
0003); without one a hybrid partial match correctly truncates to 0.

## gpu_baseline (Phase 0 — raw CUDA data-plane floor, STRATAKV_GPU=ON)

`stratakv_gpu_baseline` measures the CUDA floor under the data plane on
the current GPU environment: peak contiguous bandwidth; task-shaped
transfers at the E0 block geometry (24 layers × 8 KiB = 192 KiB block)
across three methods (contig / per-(block,layer) memcpy / strided
kernel) × host memory kinds (pinned / registered / pageable) ×
directions × {1,4,16,64} blocks in sync and pipelined timing modes;
per-task fixed overhead (0-byte extrapolation); micro costs. Every row
is XXH3-verified against a golden pattern before timing. The harness
probes SM access to registered/pageable memory (forked child; WSL2
faults) and gates kernel scenarios on the result.

Methodology, environment record, results, and findings:
[`phase0_gpu_baseline.md`](phase0_gpu_baseline.md). Baseline log (WSL2 /
RTX 4080 SUPER, 2026-11-19): `results/2026-11-19_wsl2_rtx4080super/phase0_gpu_baseline.log`.

Run:

```bash
cmake -S . -B build-gpu -DCMAKE_BUILD_TYPE=Release -DSTRATAKV_GPU=ON
cmake --build build-gpu -j && ./build-gpu/benchmarks/stratakv_gpu_baseline
```

Headline (pinned, 4-block task, SYNC mode): contig 55.5 µs, kernel
74.9 µs, memcpy24 409.6 µs; per-task fixed overhead ≈ 22–32 µs — vs
FlexKV v1.0.0's 10–20 ms per task on the same payload class (E0).

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
