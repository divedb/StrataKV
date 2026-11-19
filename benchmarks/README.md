# Benchmarks (Phase A3 harness — not yet ported)

The reproducible benchmark harness currently lives with the audited FlexKV
environment: `~/Desktop/FlexKV/examples/vllm_adaption/serve_qwen2_clean.py`
plus the recorded baselines (`flexkv_clean_baseline.log`,
`docs/fluxkv/PHASE0_AUDIT.md`). It will be generalized into this
directory at Phase A3 with:

- pinned workload JSON (prompts, block counts, concurrency, hit rates);
- metric collectors: match p50/p95/p99, per-task fixed overhead,
  effective GB/s, TTFT/TPOT deltas, scheduler stall;
- environment capture (commits, drivers, clocks) in every report.

Until then this README is the entire contents (no silent stubs).
