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

## gpu_transport_bench (Phase E3 — full stack, STRATAKV_GPU=ON)

`stratakv_gpu_transport_bench` measures per-task cost through the REAL
stack — MetadataPlane → TaskEngine → TransferScheduler →
CudaStreamTransport (pinned CpuTier, LAYERWISE GPU registration,
in-process mode) — at the Phase 0 geometry. Sync mode is
Submit→Wait per task (the FlexKV per-task cost definition); pipelined
mode is K=8 tasks with one flush+wait. Every scenario is XXH3-verified
before timing. Run like gpu_baseline (same build).

Baseline (WSL2 / RTX 4080 SUPER, 2026-11-19,
`results/2026-11-19_wsl2_rtx4080super/gpu_transport_bench.log`):

| task (4 blocks = 768 KiB, sync p50) | µs | vs bar 1000 µs | vs FlexKV v1.0.0 |
| --- | --- | --- | --- |
| put | 184.8 | PASS (5.4× headroom) | ~59–114× lower |
| get | 194.6 | PASS (5.1× headroom) | ~56–108× lower |

Full-stack 0-byte intercept: put 135.6 µs / get 163.6 µs (raw CUDA
floor ≈ 22–32 µs; the delta is scheduler queueing, per-batch scratch
upload, the 50 µs completion-poll cadence, and engine bookkeeping —
tuning headroom for later phases). Pipelined (K=8): put ≈ 10 GB/s,
get ≈ 17.7 GB/s per task at 64 blocks. **Roadmap Phase E acceptance
(per-task fixed overhead ≤ 1 ms at 4-block payloads): PASS in
in-process mode** (the E4 isolation experiment still owes the
isolated-process comparison).

## gpu_transport_py_bench (Phase E — the Python layer, STRATA_GPU build)

`benchmarks/gpu_transport_py_bench.py` drives the same real stack from
Python through the typed facade (`register_gpu_group`/`start`,
`stage_put`/`submit_put`/`wait`, `match`/`submit_get`/`wait`) at the
same geometry — the delta vs `stratakv_gpu_transport_bench` prices
exactly the pybind + facade layer the future vLLM connector pays per
step. Byte-verified round trips precede every timed scenario.
Requires the module built with `-DSTRATAKV_PYTHON=ON
-DDSTRATAKV_GPU=ON`. Run: `PYTHONPATH=python python3
benchmarks/gpu_transport_py_bench.py`.

Baseline (WSL2 / RTX 4080 SUPER, CPython 3.12, 2026-11-19,
`results/2026-11-19_wsl2_rtx4080super/gpu_transport_py_bench.log`):
4-block sync p50 **put 200.4 µs / get 221.4 µs** vs C++ 184.8/194.6 —
the Python layer adds only ~16–27 µs per task (acceptance bar PASS
from Python, 4.5× headroom); intercepts 157.4/192.4 µs vs C++
135.6/163.6. Match through the facade: 2.8 µs @1 block → 42.4 µs @64
blocks — token-list conversion dominates at scale; the connector
should pass numpy arrays eventually (design.md's facade sketch already
anticipates `npt.NDArray` input).

## Environments compared (2026-11-19: E-Local WSL2 vs E0 remote host)

The remote E0-class host (RTX 3080 Ti, native Linux, driver 580.76.05,
CUDA 12.8; StrataKV rsynced to `/root/autodl-tmp/StrataKV`, built
Release/sm_86 — local repo remains the single source of truth) runs
the same committed binaries; logs under
`results/2026-11-19_remote_rtx3080ti/`:

| metric (same harness, both PASS the ≤1 ms bar) | E-Local (WSL2, 4080 SUPER) | E0 host (native, 3080 Ti) |
| --- | --- | --- |
| pinned H2D / D2H peak | 22.0 / 24.0 GB/s | 12.3 / 12.5 GB/s (link-limited) |
| D2D | 278 GB/s | 407 GB/s |
| 4-block pinned/kernel sync (h2d/d2h) | 74.9 / 82.9 µs | 73.2 / 72.0 µs |
| raw 0-byte intercept (h2d/d2h) | 31.9 / 15.4 µs | 9.5 / 8.3 µs |
| empty CUDA task | 22.3 µs | **5.2 µs** |
| full stack 4-block put/get | 184.8 / 194.6 µs | **147.3 / 152.4 µs** |
| full-stack 0-byte intercept (put/get) | 135.6 / 163.6 µs | 103.3 / 111.0 µs |
| SM access to `cudaHostRegister`'d memory | FAULTS (F4) | works |
| cross-process CUDA IPC open | fails (ADR 0002 note) | **works** (E4 unblocked) |

Two structural findings: the WSL2 GPU-PV path carries ~4× higher CUDA
fixed costs than native (empty task 22.3 vs 5.2 µs — why the 3080 Ti
*outscores* the faster-linked 4080 SUPER on the full stack despite
half the bandwidth), and the F4 registered-memory constraint is
WSL2-specific (native Linux runs registered+kernel scenarios fine;
pageable+SM faults everywhere). The 3080 Ti's lower intercepts make it
the better E4/isolation-mode substrate — as the roadmap assumed.

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
