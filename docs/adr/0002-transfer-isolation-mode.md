# ADR 0002 — Transfer isolation mode (in-process vs isolated process)

- Status: **Accepted (final — gate G2 passed 2026-11-19; numbers below)**
- Date: 2026-11-19 (decision), 2026-11-19 (experiment)
- Deciders: StrataKV design (roadmap Phase E, task E4)

## Context

FlexKV v1.0.0 (audited) runs transfer workers in a separate process fed by
cross-process queues. The clean-run baseline measured a fixed ~10–20 ms
per-task cost for ~0.6–0.8 MB payloads, attributed to the submit/complete
round trips, not to bandwidth or matching (matching was 0.31–0.46 ms,
in-process). Upstream FlexKV later moved to in-process layerwise transfers
(`FLEXKV_ENABLE_LAYERWISE_TRANSFER`, Jun 2026) claiming latency wins.

Isolation trade-offs observed during the audit:

- **Isolated process**: a transfer crash cannot corrupt or freeze the
  engine; GPU access requires CUDA IPC registration (validated pattern:
  `cuda_ipc_demo`); costs cross-process queue latency and a second CUDA
  context.
- **In-process threads**: no IPC, shared CUDA context, direct stream use;
  but transfer faults (or driver-level stalls) can take down the engine,
  and the GIL/engine event loop must be carefully protected.

## Decision

1. The data plane is built behind `ITransport` +
   `TransferScheduler(IsolationMode)`, so both modes share all logic.
2. Interim default: isolated process (safety while the runtime is young).
3. Phase E task E4 measures both modes on the Phase A harness (throughput,
   TTFT/p99, memory, contention, fault injection) and finalizes this ADR
   at gate G2. No default is changed without those numbers.

## Final decision (gate G2, 2026-11-19) — keep isolated-process as the default

The E4 experiment ran the full stack on the E0 remote host (RTX 3080
Ti, native Linux, CUDA 12.8, driver 580.76.05) in both modes, same
harness, same session
(`benchmarks/results/2026-11-19_remote_rtx3080ti/e4_isolation_experiment.log`;
harness `stratakv_gpu_transport_bench --mode {in-process,isolated}`):

| metric (sync submit→Wait, p50) | in-process | isolated | premium |
| --- | --- | --- | --- |
| 4-block put (768 KiB) | 147.9 µs | 195.6 µs | +47.7 µs |
| 4-block get | 155.7 µs | 207.3 µs | +51.6 µs |
| 0-byte intercept (put/get) | 107.6 / 108.6 µs | 146.0 / 150.0 µs | +38 / +41 µs |
| 64-block put | 1229 µs | 1357 µs | +10% |
| marginal bandwidth | ~11.3 GB/s | ~10.4 GB/s | ~-7% |
| Phase E bar (≤ 1000 µs at 4 blocks) | PASS | PASS | both 5× under |

Fault injection (test `GpuIsolatedTest.WorkerKillFailsOpsButEngineSurvives`,
same host): SIGKILL of the worker mid-batch (8 × 64 blocks in flight)
→ every task reaches a terminal state (failures, never a hang), staged
reservations recover via the documented abort path, subsequent
submissions fail fast and loudly, the engine shuts down cleanly. The
engine process survived every transfer-worker death thrown at it.

Memory overhead: the worker adds its own CUDA context plus the
transport's 8 MiB device scratch and its own (host-side, pinned) CPU
tier. Per-process GPU totals are not observable inside the autodl
container (nvidia-smi hides process/usage metrics there); the context
cost is the standard ~300–500 MiB per CUDA process and is the operator
accounting number until measurable on a bare-metal host.

**Decision:** isolated-process stays the default — the safety is
proven (fault containment works; the engine never hangs) and the
premium (~50 µs/task, single-digit-% bandwidth) is far below any
budget this system has (5× under the Phase E bar, ~200× below FlexKV
v1.0.0's per-task cost, whose isolation overhead was 10–20 ms). 
In-process remains a first-class configuration for
latency-sensitive deployments (the `IsolationMode` seam and both
transports ship; per-workload choice stays available). A future ADR
may revisit if a measured workload needs the last 50 µs.

## Consequences

- The registration flow (`DeviceMemoryRegistry`) must work for both
  modes; in-process mode still uses explicit descriptors (no implicit
  torch pointer capture) to keep the contract identical.
- Fault-injection tests are required before in-process can become the
  default anywhere.

## Note added 2026-11-19 (E-Local probe)

A cross-process CUDA IPC feasibility probe on the local WSL2 host
(export via `cudaIpcGetMemHandle` succeeds; `cudaIpcOpenMemHandle` in
a second process fails with "invalid resource handle", byte-verified
two-way probe) shows **the isolated-process mode cannot be built or
tested on WSL2** — consistent with NVIDIA's long-standing WSL IPC
limitation. The E4 experiment therefore requires a native-Linux GPU
host (the E0-class remote). The in-process measurement (Phase E
acceptance: put 184.8 / get 194.6 µs at 4 blocks; Python layer +16–27
µs) stands; the G2 decision keeps its interim isolated default until
E4 runs there.

## Note added 2026-11-19 (E0-host probe, same day)

The probe now exists in-tree (`benchmarks/ipc_probe.cu`) and was run
on the E0-class remote (RTX 3080 Ti, native Linux, driver 580.76.05):
**cross-process CUDA IPC works end to end** (child opened the handle,
verified the parent's bytes, wrote its own pattern, parent observed
it — log in `benchmarks/results/2026-11-19_remote_rtx3080ti/`). E4's
isolated-process mode is implementable and testable there; the same
host also runs the full stack faster than E-Local (4-block put 147.3 /
get 152.4 µs — native CUDA fixed costs are ~4× lower than WSL2's,
outweighing its half-bandwidth link).
