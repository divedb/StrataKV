# ADR 0002 — Transfer isolation mode (in-process vs isolated process)

- Status: Accepted (interim default: isolated; final decision deferred to
  gate G2 after the Phase E experiment)
- Date: 2026-11-19
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
