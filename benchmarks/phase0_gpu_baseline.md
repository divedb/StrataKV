# Phase 0 GPU baseline — WSL2 / RTX 4080 SUPER (2026-11-19)

First performance record on the local GPU environment (GPU acceleration
newly available under WSL2), taken before any StrataKV CUDA data-plane
code exists. Purpose per `roadmap.md` §5 (benchmark policy): every later
measurement compares against a recorded baseline on recorded hardware.
This file is the methodology + interpretation; the raw, committed logs
are authoritative:

- `results/2026-11-19_wsl2_rtx4080super/phase0_gpu_baseline.log` — GPU
  data-plane capability baseline (harness:
  `benchmarks/gpu_baseline_bench.cu`)
- `results/2026-11-19_wsl2_rtx4080super/index_bench.log` — Phase C
  metadata-plane match latency on the same box

All claims below cite that log. Harness commit `563cb8f` plus the
benchmark/CMake changes in the same working tree as this file.

## Environment

| Component | Value |
| --- | --- |
| OS | WSL2 (kernel 6.6.87.2-microsoft-standard-WSL2), Ubuntu 24.04 userland |
| GPU | NVIDIA GeForce RTX 4080 SUPER, 16 GB, compute capability 8.9 (sm_89) |
| Driver | 610.88 (CUDA UMD 13.3); `nvidia-smi` 610.57.01 |
| CUDA toolkit | 12.0 (V12.0.140) — **deviation from the 12.8 pin** (see below) |
| Host compiler | gcc 13.3.0 (works with nvcc 12.0 on this setup) |
| CMake | 3.28.3 |
| CPU / RAM | 32 cores, 31 GiB |
| Other GPU tenants | Xwayland (display); GPU idle-clocked between sections |
| Clocks (post-run sample) | sm 465 MHz idle / max 2595 MHz; mem 11501 MHz; no ECC |

Toolkit deviation: `docs/toolchains.md` pins CUDA 12.8 (verified on the
remote RTX 3080 Ti host). The local toolkit is 12.0 — oldest available
here; nvcc 12.0 + gcc 13.3 compiles and runs everything in this baseline
correctly (verified kernels, byte-identical transfers). Phase E code
must avoid CUDA >12.0 API surface until the local toolkit is upgraded,
or the pin is relaxed after re-verification.

## Methodology

**Geometry.** One model-wide KV block = 24 layers × 8 KiB = 192 KiB
(Qwen2-0.5B-shaped: 2 (K,V) × 2 kv_heads × 64 head_size × 2 B fp16 × 16
tokens/block per layer). 4 blocks = 768 KiB — the payload class FlexKV
v1.0.0 spent 10.85–21.04 ms per task on (E0 ground truth, `roadmap.md`
§0). Sizes swept: 1/4/16/64 blocks (roadmap §5 payload sweep).

**Layouts** mirror the real tier shapes (roadmap Appendix B): GPU side
LAYERWISE (24 separate per-layer pools), CPU side one flat BLOCKWISE
pool laid out `[block][layer][slice]`. An extra flat GPU buffer backs
the contiguous ceiling scenario.

**Methods.**

- `contig` — one `cudaMemcpyAsync` of the whole payload between flat
  buffers; the copy-engine ceiling for the payload size.
- `memcpy24` — one 8 KiB `cudaMemcpyAsync` per (block, layer) — the
  per-layer copy decomposition FlexKV v1.0.0's transfers reduce to.
- `kernel` — one strided gather/scatter kernel launch per task (the
  `transfer.cu`-harvest design point; 64-bit words, 256-thread blocks).

**Host memory kinds.** `pinned` (`cudaMallocHost`), `registered`
(`cudaHostRegister` on aligned malloc — the CpuTier pinning plan, E2),
`pageable`.

**Timing modes.**

- SYNC — one submit → `cudaEventRecord` → `cudaEventSynchronize` round
  per task; the per-task wall time comparable to FlexKV's per-task
  cost (their measurement also spanned submit→completion).
- PIPELINED — K tasks enqueued back-to-back, one sync (the scheduler's
  drain-and-batch shape; K adaptive, 8–256).

`std::chrono::steady_clock`; 3 warmups + adaptive iterations (~200 ms
per scenario, 10–400), p50/p95/p99 reported. Decimal GB/s.

**Correctness.** Every scenario row is verified before timing:
destinations scrubbed (0xA5 / kernel-scrubbed), one op executed, then
XXH3-hashed against a deterministic splitmix64 golden pattern (H2D
destinations gathered back through trusted synchronous `cudaMemcpy`s).
A failed hash aborts the run — no timing counts unless bytes verify.
The run ends with `all verifications passed`.

**Platform probe (SM access).** On this WSL2 platform SM/kernel access
to `cudaHostRegister`'d and to pageable host memory faults with an
illegal memory access, while copy-engine (`cudaMemcpyAsync`) access to
the same memory works (bytes verified — see F4 below). The harness
probes each memory kind in a forked child **before the parent touches
CUDA** (the fault is sticky for the process lifetime, and on WSL2 even
`cudaDeviceReset` recovery fails with "device busy/unavailable"), then
gates kernel scenarios on the probe result, printing `SKIP` rows. The
header line `SM access probed: …` records the outcome per run.

**Reproduce.**

```bash
cmake -S . -B build-gpu -DCMAKE_BUILD_TYPE=Release -DSTRATAKV_GPU=ON
cmake --build build-gpu -j
./build-gpu/benchmarks/stratakv_gpu_baseline          # GPU baseline
./build-cpu/benchmarks/stratakv_index_bench           # metadata plane
```

Capture `nvidia-smi`, `uname -a`, and the git commit into the log
header when recording results (see the committed log's header block).

## Results (p50 unless noted; full tables in the log)

**Peak contiguous bandwidth, 256 MiB:**

| path | GB/s |
| --- | --- |
| H2D pinned | 22.0 |
| D2H pinned | 24.0 |
| H2D pageable | 18.6 |
| D2H pageable | 12.2 |
| D2D (device memcpy) | 278.2 |

PCIe 4.0 ×16 class, as expected for this GPU.

**Per-task cost, SYNC mode, pinned, at the E0 payload class:**

| task (4 blocks = 768 KiB) | p50 µs | GB/s | vs FlexKV v1.0.0 (E0: 10,850–21,040 µs) |
| --- | --- | --- | --- |
| h2d contig | 55.5 | 14.2 | ~195–378× lower |
| h2d kernel | 74.9 | 10.5 | ~145–281× lower |
| h2d memcpy24 | 409.6 | 1.92 | ~26–51× lower |
| d2h contig | 60.9 | 12.9 | — |
| d2h kernel | 82.9 | 9.5 | — |
| d2h memcpy24 | 375.0 | 2.1 | — |

**Per-task fixed overhead** (least squares on pinned/kernel SYNC rows
over 1/4/16/64 blocks; intercept = extrapolated 0-byte task):

| dir | intercept µs | marginal GB/s |
| --- | --- | --- |
| h2d | 31.9 | 21.1 |
| d2h | 15.4 | 10.8 |

Measured empty task (event record + sync only): **22.3 µs** p50.

**Micro costs (p50):** empty event task 21.9 µs; no-op kernel task
29.4 µs; `cudaMemcpyAsync` 8 KiB submit-only 7.96 µs/call; kernel
launch submit-only 5.73 µs/call.

**Metadata plane on this box** (`index_bench`, warm-hit 1k blocks):
p50 26.0 µs / p95 27.9 / p99 45.1 — roadmap budget p50 < 100 µs PASS
with ~4× headroom (Apple-Silicon reference in `benchmarks/README.md`:
53.1 µs).

## Findings

- **F1 — the fixed-cost problem is software, not hardware.** The raw
  submit→completion floor on this environment is ~20–35 µs per task
  (empty task 22 µs; fit intercepts 15–32 µs), against FlexKV v1.0.0's
  10–20 ms on the E0 remote host. The Phase E acceptance bar (per-task
  fixed overhead ≤ 1 ms at 4-block payloads) has ~30× headroom before
  any StrataKV code is written. If StrataKV's transport lands anywhere
  near its own floor, the bar is met by construction; the risk to watch
  is added software layers (queueing, completion dispatch), not CUDA.
- **F2 — the per-(block,layer) memcpy decomposition is the v1.0.0
  tax, independent of IPC.** `memcpy24` caps at ~2 GB/s even pinned and
  in-process: 24 submits/block at ~8 µs API cost each is submit-bound
  (micro table), and pipelining does not help it (log: 2.0–2.1 GB/s in
  both modes). At 4 blocks it is already 5–7× slower than one kernel
  launch. Phase E's strided-kernel (or copy-engine) transport is
  justified by this number alone; the FlexKV layerwise-decomposition
  heritage must not be inherited for CPU-tier transfers.
- **F3 — D2H via SM gather is bandwidth-halved; prefer the copy engine
  where layout permits.** D2H kernel gather tops out at ~10.7 GB/s
  (64 blocks) vs 22.9 GB/s for contiguous D2H on the same buffers —
  the SM-side gather from 24 scattered per-layer GPU pools does not
  saturate the link, while H2D scatter reaches ~20 GB/s. Phase E
  transport design should keep BLOCKWISE-contiguous GPU pools where
  the model/adapter allows (making D2H a single copy-engine op), or
  batch per-layer copies into large contiguous segments; the decision
  gets re-measured by the E4 isolation experiment.
- **F4 — WSL2 SM-access constraint (environment contract).** `cudaHostRegister`
  returns success, and copy-engine access to registered memory is
  correct and fast (verified), but SM access to registered or pageable
  host memory faults (illegal memory access; reproduced minimally in
  isolation). Consequences: on WSL2 the CPU tier must back
  kernel-served transfers with `cudaMallocHost` pools (or route
  registered/pagable memory exclusively through copy-engine memcpys),
  and `cudaDeviceReset` cannot recover a faulted context ("device
  busy"). This constrains roadmap E2's `cudaHostRegister` pinning plan
  and ADR 0002's isolated-mode IPC flow; native-Linux hosts must be
  probed before assuming the same limitation (the harness probe does
  this automatically).
- **F5 — pageable staging is catastrophic for small copies, D2H
  worst.** Pageable `memcpy24` D2H runs at 0.4 GB/s (staging tax per
  8 KiB op); even contiguous pageable D2H halves bandwidth vs pinned
  (12.2 vs 24.0 GB/s). All CPU-tier pools that receive streamed
  transfers must be pinned on this environment.
- **F6 — metadata plane comfortably inside budget on this box**
  (26 µs warm-hit p50 vs 100 µs budget), so matching will not be the
  bottleneck when the GPU phases start measuring end-to-end.

## Comparability notes

- E0 (FlexKV v1.0.0, remote RTX 3080 Ti, PCIe 4.0 ×16) and this
  baseline (RTX 4080 SUPER, same link class) differ in GPU, driver, and
  — materially — WSL2 vs native. Link-bandwidth numbers are comparable
  only as same-class; the per-task fixed-cost comparison (F1) is
  software-dominated and therefore robust across this hardware delta.
- The Phase A ladder (A1–A4, gate G1) runs the *same workload* through
  FlexKV E1/E2 on the remote host; when those logs exist, the table in
  "Results" gains E1/E2 columns and this file stays the
  local-environment reference.
- SLOs remain unset until M1 per the benchmark policy; nothing here is
  a gate, it is the floor record.
