# StrataKV Implementation Roadmap

Status: detailed implementation plan derived from the StrataKV design
proposal (`README.md`) and the completed Phase 0 audit of the FlexKV
checkout (2026-11-19). Supersedes the "FluxKV" draft naming; same mission.
Per-phase class APIs and final-architecture sequence diagrams live in the
companion `design.md`.

Evidence base for this plan (produced during the audit session):

- `~/Desktop/FlexKV/docs/fluxkv/PHASE0_AUDIT.md` — commit/version/topology audit
- `~/Desktop/FlexKV/flexkv_clean_baseline.log` (remote `/root/autodl-tmp/FlexKV/`) — uncontaminated baseline
- `~/Desktop/FlexKV/flexkv_pdb_validation_trace.log` — full connector execution trace (debugger-contaminated; useful for flow, not latency)
- `~/Desktop/FlexKV/examples/vllm_adaption/serve_qwen2_clean.py` — reproducible baseline harness

---

## 0. Ground truth already established (do not re-litigate)

| Fact | Evidence |
| --- | --- |
| Local FlexKV checkout is **v1.0.0** (`7cf7118`, 2025-09-11), patch-era, running patched vLLM 0.10.1.1 | `git describe --tags`; patch byte-identical v1.0.0↔v1.2.1 |
| Clean-run match latency: **0.31–0.46 ms** (get/put, 4–5 blocks) | clean baseline task logs |
| Clean-run per-task transfer cost: **10.85–21.04 ms** for ~0.6–0.8 MB (≈30–50 MB/s effective) | same |
| The reported "constant 40 ms match" was **debugger contamination** (~130× inflation) | pdb vs clean comparison; timers span breakpoint stops |
| Fixed-poll-interval hypothesis **falsified** (hot-loop sleeps are 0.1–1 ms) | code audit + clean numbers |
| Residual fixed per-task cost lives in the **cross-process submit/complete round trip** (separate transfer process), not in matching or compute | topology + code path audit |
| Metadata plane (radix index, hashing) and transfer kernels are **already native** (`csrc/`: `radix_tree.cpp`, `hash.cpp`, `transfer.cu`, `tp_transfer_thread_group.cpp`, pybind11 `bindings.cpp`) | repo tree |
| Upstream FlexKV already shipped: library mode (v1.0.0, Nov 2025), v1.2.1 (Dec 2025), **vLLM mainline** (v0.17.2+, Mar 2026), **in-process layerwise + CE transfers** (Jun–Aug 2026) | upstream README, verified live |

**Consequence for scoping:** StrataKV cannot justify itself on "make
matching fast" (already sub-ms) or "remove IPC" (upstream already did).
Its legitimate grounds are the gaps upstream still has or handles
per-model-family: heterogeneous/hybrid KV as a first-class object model, a
framework-independent core with a stable API, an event-driven control
plane, and engineering rigor (testing, sanitizers, CI, docs).

---

## 1. Principles (binding for all phases)

1. **Evidence before redesign.** Every phase ends with a measurement compared against a recorded baseline on identical hardware/workload.
2. **Plane separation.** Metadata (index, leases), control (tasks, completion), data (transfers) are separate modules with narrow interfaces; only the Python boundary knows pybind11; only adapters know vLLM.
3. **Heterogeneity from day one.** The model-description layer is a per-layer-group spec table (`FullAttn | Mla | SlidingWindow | LinearState`), never a single global tuple. Block pools and state-object pools are both first-class.
4. **Ownership contracts.** Engine owns GPU memory; StrataKV borrows visibility via descriptors/handles. Registrations declare lifetimes; StrataKV never frees what it does not own.
5. **No silent stubs, no invented numbers.** Unimplemented paths raise. Performance claims cite the harness run that produced them.
6. **Harvest, don't rewrite.** FlexKV's native index/hash/transfer kernels are candidates for porting under StrataKV interfaces (see Appendix A harvest map).

---

## 2. Environments

| ID | Contents | Purpose |
| --- | --- | --- |
| **E0** | Remote RTX 3080 Ti host; FlexKV v1.0.0 checkout; patched vLLM 0.10.1.1 `.venv`; `serve_qwen2_clean.py` harness | Historical control; flow tracing. **Live again (2026-11-19):** StrataKV builds/tests/benches there (logs `benchmarks/results/2026-11-19_remote_rtx3080ti/`; full stack 4-block put 147.3/get 152.4 µs — acceptance PASS; cross-process CUDA IPC works → E4 substrate). NOTE: the old FlexKV checkout/venvs are GONE (fresh container) — A1/A2 need environment reconstruction. |
| **E1** | Same host; `git worktree` at **v1.2.1**; cloned `.venv` with flexkv repointed (vLLM patch identical) | Middle baseline: what did Sept→Dec 2025 recover? |
| **E2** | Upstream FlexKV `main` + **vLLM ≥ 0.17.2** (unpatched, mainline connector); fresh venv; system deps (`liburing`, `libxxhash`, `libhiredis`) | Modern reference; the "adopt vs build" gate input |
| **E-Local** | Local WSL2 host (RTX 4080 SUPER, sm_89, CUDA 12.0, driver 610.88; toolchain deviations in `docs/toolchains.md`) | Phase 0 raw-GPU floor + local dev GPU: `benchmarks/phase0_gpu_baseline.md` (recorded 2026-11-19; per-task fixed overhead ≈ 22–32 µs vs E0's 10–20 ms; WSL2 forbids SM access to registered/pageable host memory) |

E0 is done. E1 and E2 are the first deliverables of Phase A below. GPU
acceleration became available locally (WSL2) on 2026-11-19; the Phase 0
local baseline is recorded and the Phase E deferred items (CUDA
transport, IPC source, pinning, E4) can proceed here, with the platform
constraint above shaping the pinning plan.

---

## 3. Phases

### Phase A — Close the baseline ladder (gate G1: adopt vs build)

**Goal:** three comparable baselines (E0/E1/E2) on one harness before any StrataKV code.

| # | Task | Deliverable | Size |
| --- | --- | --- | --- |
| A1 | E1 worktree + venv clone + rebuild + identical workload | `flexkv_v1.2.1_baseline.log` | S |
| A2 | E2 environment; run mainline connector equivalent of the same workload (Qwen2-0.5B, block 16, same prompt/max-tokens; layerwise/CE toggles on and off) | `flexkv_upstream_baseline.log` (4 variants) | M |
| A3 | Extend harness: p50/p95/p99 over ≥50 requests, payload sweep (1/4/16/64 blocks), concurrency 1/4/16, hit-rate 0/50/100%, cold/warm; capture TTFT/TPOT + task-internal timings | `bench/` harness v2 (Python, reusable for StrataKV) | M |
| A4 | Decision memo: per-metric table E0→E1→E2; if upstream already meets the residual-cost and API-stability bars, StrataKV narrows to "gap work on top of upstream" (hybrid object model, framework independence) instead of a parallel runtime | `docs/adr/0001-adopt-vs-build.md` | S |

**Exit gate G1:** written recommendation with numbers. No core code before G1.

### Phase B — Repository bootstrap and object model (no GPU yet)

**Status: ✅ implemented 2026-11-19** (B1–B5; offline adaptation: pinned
clones via `scripts/fetch_third_party.sh` instead of live submodules due
to no GitHub egress; unit tests ran on the documented GoogleTest bootstrap
shim `STRATAKV_GTEST_SHIM=ON`, 17/17 passing in Debug, Release, and
ASan+UBSan builds). Deferred to Phase A (gate G1): A1–A4 baseline ladder
tasks requiring the GPU host.

**Goal:** skeleton repo, build system, CI, and the type system everything else keys off.

| # | Task | Deliverable | Size |
| --- | --- | --- | --- |
| B1 | Repo per README structure (`include/stratakv/{metadata,control,data,common}`, `src/…`, `bindings/python`, `python/stratakv`, `integrations/vllm`, `tests/{cpp,python}`, `benchmarks`, `third_party`, `docs/adr`); top-level `CMakeLists.txt` with targets `stratakv_core`, `stratakv_bindings`, `stratakv_tests`, options `STRATAKV_GPU`, `STRATAKV_PYTHON` | compiling empty targets, Debug/Release, install/export | M |
| B2 | Toolchain pinning doc (CUDA 12.8, gcc/clang versions, Python 3.12); submodules added pinned: GoogleTest, pybind11, xxhash (first harvest) | `docs/toolchains.md`, `third_party/` | S |
| B3 | clang-format (Google), clang-tidy config, CI (build + unit + sanitizer job) | `.clang-format`, `.clang-tidy`, CI green | M |
| B4 | **Core object model** (headers + unit tests, no impl beyond in-memory): `LayerSpec` variant table; `TierSpec` (GPU descriptor / CPU pool / SSD path); `BlockRef` (tier, block id, refcount); `StateRef` (sidecar object id); `CacheEntry` (bundle of BlockRefs + StateRefs at a prefix boundary); `BufferView` (non-owning, device id + size + strides); `TaskHandle` | `include/stratakv/common/types.h` + tests | M |
| B5 | ADRs: 0002 in-process vs isolated transfer; 0003 hybrid sidecar & no-partial-caching invariant; 0004 GIL/thread dispatch policy; 0005 IPC-handle ownership & lifetime; 0006 endpoint naming (`registration_endpoint` / `control_endpoint` — retire the `server_recv_port` fossil) | `docs/adr/` | S |

**Acceptance:** `cmake --build` + `ctest` green in CI; object model unit tests cover heterogeneous spec tables (construct a Qwen3-Next-shaped 3-group table and a DeepSeek-V4-shaped one).

### Phase C — Metadata plane (index + allocators)

**Status: ✅ implemented 2026-11-19** (C1–C5; offline adaptation: the
Phase D Python oracle for C5 is deferred until the pybind11 facade
exists — the fuzz target instead verifies against an independent C++
oracle (flat-scan model) exactly in no-evict modes and by invariants +
allocator conservation under eviction, with `STRATAKV_FUZZ_ITERS` to
extend; verified to 100k ops/mode. xxHash pinned as the
`third_party/xxhash` submodule (v0.8.2 — the same library FlexKV links,
so chained digests match by construction). Benchmark (C4): warm-hit
p50 53.1 µs at 1k blocks on the dev box vs the 100 µs budget —
`benchmarks/README.md`. Parity (C1): the FlexKV `radixtree.py __main__`
golden scenario replays with identical match lengths and eviction
traces; three documented divergences (OOB child-hash read, contiguous
ready accounting, span-bounded CommitReady ancestor marking) are unit-
tested in `tests/cpp/radix_index_test.cpp`.)

**Goal:** prefix index and tier allocation with the semantics from ADR 0003.

| # | Task | Deliverable | Size |
| --- | --- | --- | --- |
| C1 | Port/harvest FlexKV's C++ radix tree + xxHash chained hashing (`csrc/radix_tree.cpp`, `csrc/hash.cpp`) behind `IIndex` interface; keep the chained-prefix-hash contract (hash(i) covers tokens ≤ block i) | `stratakv_metadata` target + GoogleTest vs golden vectors from FlexKV runs | M |
| C2 | Node payload generalization: `CacheEntry` bundles (BlockRef arrays per group + StateRef array) instead of FlexKV's single physical-block array; `is_ready` per entry; split/merge/shrink semantics preserved | tests incl. partial-node split, mid-node match | M |
| C3 | Allocators: block-id free-list per tier (harvest `Mempool` semantics: numpy-style free-mask → `std::vector` bitset) **plus** fixed-size state-object pool for LinearState sidecars; LRU + grace-time eviction honoring `lock_cnt` | tests: alloc/recycle/evict/lock interactions | M |
| C4 | Match API: `MatchResult Match(Span<const token_id>, tier readiness filter)` — returns token-count, entry refs, ready-prefix boundary; p50 < 100 µs at 1k blocks (unit bench, CPU-only) | `benchmarks/index_bench` | M |
| C5 | Fuzz: random insert/match/evict interleavings vs a Python oracle model; no leaks (ASan/UBSan), no double-recycle | fuzz target in CI | M |

**Acceptance:** index returns identical match lengths to FlexKV's tree on recorded golden sequences; heterogeneous-entry tests pass.

### Phase D — Control plane (task engine + completion)

**Status: ✅ implemented 2026-11-19** (D1–D4; offline adaptations:
pybind11 could not be cloned — GitHub egress dropped mid-session — so
the pin is vendored in-tree from the official v2.13.6 wheel under
`third_party/pybind11` (standard pip layout, `find_package(pybind11
CONFIG)`), with conversion to a real submodule noted for when egress
returns. The design.md `TaskEngine(meta, data::TransferScheduler*)`
signature is realized as `TaskEngine(meta, ITaskExecutor*)` — the
executor seam Phase E's scheduler will implement; Phase D ships
`InlineExecutor` (simulation, per-op delay injection; no bytes moved)
plus test fakes. Additions beyond design.md, needed by the abort path
of the cancellation diagram: `IIndex::RemoveEntry` +
`MetadataPlane::AbortPut`, with aborted-span fragments marked evictable
(`aborted` node state). Acceptance verified: pytest drives the engine
end to end (10 tests, incl. hybrid truncation and GIL-heartbeat under
simulated 100 ms latency); 71 C++ tests incl. 10k-task stress with
random cancels and shutdown-mid-flight, green in Debug/Release/ASan+
UBSan; CI gains a Python job.)

**Goal:** event-driven task engine replacing per-step polling.

| # | Task | Deliverable | Size |
| --- | --- | --- | --- |
| D1 | Task state machine (`Created → Staged → Launched → Succeeded/Failed/Cancelled`), graph/op model (harvest FlexKV's `TransferOpGraph` shape), backpressure (bounded queues, reject semantics) | core target + tests | M |
| D2 | Completion machinery: background thread(s) + `condition_variable`/`eventfd`; explicit thread-sync vs cross-process notification split; cancellation/shutdown/timeout/error propagation tested under stress | tests: 10k tasks, random cancels, shutdown mid-flight | L |
| D3 | pybind11 facade v1: submit/query/wait/cancel + match; `gil_scoped_release` on all blocking entry points per ADR 0004; callbacks dispatched on a documented single context (never raw background threads) | `stratakv_bindings` + pytest | M |
| D4 | Python package `stratakv` with type hints, exceptions mapping, no torch import in core path (raw descriptors only) | packaging + smoke test | S |

**Acceptance:** pytest suite drives the engine from Python; GIL released during waits (verified by a Python-side heartbeat thread not starving).

### Phase E — Data plane (registration, CPU tier, transfer engine)

**Status: ✅ complete 2026-11-19 — gate G2 passed (ADR 0002 final)** (the
offline portion landed 2026-11-19 morning — `DeviceMemoryRegistry` with
ADR 0005 lifetime semantics, `CpuTier` byte ownership,
`TransferScheduler` drain-and-batch with fake-transport wiring. The
local GPU then enabled the GPU portion: real `CudaIpcHandleSource`,
page-locked tier backings (`cudaMallocHost`/`cudaHostRegister`/pageable
with loud CPU-build failures) and the model-wide block-slot fix;
`CudaStreamTransport` (TierTable id→buffer geometry, one strided-kernel
launch per op on SM-safe memory with copy-engine fallback per F2/F4,
scratch-arena parameter upload, stream pool, single-worker event
completion); `SubmitGet`/SubmitPut` gained `gpu_state_slots` (state ops
previously lacked GPU-side addressing); the Python facade drives the
real stack (`use_gpu_transport`, `register_gpu_group`, `start()`), and
`ITaskExecutor::Flush()` became the per-step launch point (diagram 2
cadence). **E4 isolation experiment (both modes on the E0 remote
host, `e4_isolation_experiment.log`): 4-block put/get p50 147.9/155.7 µs
in-process vs 195.6/207.3 µs isolated — the isolation premium is
~50 µs/task; both pass the ≤ 1 ms acceptance bar 5× under; fault
injection (worker SIGKILL mid-batch) leaves the engine alive with
clean task failures and recoverable reservations. ADR 0002 final:
isolated-process stays the default, in-process stays selectable.**
Found and fixed along the way: a TaskEngine race (terminal state
visible before CommitPut/Release effects — shredded entries
nondeterministically; effects now atomic under the engine mutex),
non-PIC static libs breaking Linux Python builds, and CMake 3.22/CUDA
12 portability (documented in `docs/toolchains.md`).)

**Goal:** GPU registration + tier transfers, benchmarked in both isolation modes per ADR 0002.

| # | Task | Deliverable | Size |
| --- | --- | --- | --- |
| E1 | `DeviceMemoryRegistry`: register per-group GPU buffers as raw descriptors + CUDA IPC handle export (mirror of the validated `cuda_ipc_demo` flow); layout declared per group (LAYERWISE/BLOCKWISE stride tables); ownership + lifetime validation | tests on E0/E2 GPUs | M |
| E2 | CPU tier: one pinned pool per group (+ state-object pool), allocation hooked to C3; host registration chunking policy documented | tests + `host_dma` notes | M |
| E3 | Transfer engine v1: stream pool, strided copy kernels (harvest `transfer.cu` concepts), op batching (harvest worker.py's drain-and-batch loop), CUDA-event completion feeding D2 | `stratakv_data` target | L |
| E4 | **Isolation experiment (gate G2):** same harness, mode=in-process vs mode=isolated-process: throughput, TTFT/p99, fault-injection (transfer crash: engine survives?), memory overhead | `docs/adr/0002` finalized with numbers | M |
| E5 | Optional backends behind `ITransport`: GDS/io_uring (SSD tier), NIXL — only if a benchmarked need exists | ADR per backend | deferred |

**Acceptance:** per-task fixed overhead ≤ 1 ms in the winning isolation mode at 4-block payloads (vs 10–20 ms measured on FlexKV v1.0.0); report win/loss either way.

### Phase F — vLLM adapter and hybrid support

**Status: F1 ✅ / F2 ✅ / F3 ✅ / F4 pending (2026-11-20)** — F1:
`integrations/vllm/v0_17/stratakv_connector.py` implements the
mainline connector hook set over the native facade (worker role:
multi-group registration from raw descriptors — torch optional at the
boundary; scheduler role: match/submit/one-poll-per-step;
`build_connector_meta` returns empty by design). v1 topology is
single-process vLLM (both roles share the engine); the cross-process
registration endpoint stays future work with the real environment.
Tested WITHOUT vLLM via a fake-vLLM harness (duck-typed tensors over
the debug device helpers; scheduler-step driver) — ctest
`stratakv_connector_tests` runs in simulation on CPU builds and with
REAL bytes on GPU builds. F2 (byte leg): cold-put → warm-get round
trip through the hooks is byte-identical per block on both GPU hosts,
including a REAL-torch test on the E0 remote (torch 2.8 present there;
verified via `torch.equal`). Model-output parity passes on vLLM 0.34.0 / torch 2.14.0 with
Qwen2.5-0.5B-Instruct on the RTX 3080 Ti: two rounds, prefix cache
disabled, native puts/gets/match hits, identical completions. Worker
completion polling fixes empty-step load progress; complete-block
selection excludes partial tails. Transfer failures surface native errors.
F3: hybrid layouts (multi-group + LinearState sidecars +
`HybridSidecarMixin` slot bookkeeping + SWA store-all) enforced at the
connector boundary — the ADR 0003 truncation test passes through
vLLM's hook order; `num_states` added to the facade PutPlan to size
`gpu_state_slots`. Found+fixed in the connector: try_wait tuple
unpacking compared the error field instead of the state (completions
never surfaced), group-index vs group-name keying in the get path, and
simulation-mode registration guards. F4 (real hybrid model): pending real-model and vLLM HMA validation;
the vLLM installation is now available.

**Goal:** thin connector + the hybrid model upstream lacks generically.

| # | Task | Deliverable | Size |
| --- | --- | --- | --- |
| F1 | Adapter for **E2 vLLM (mainline-era)** first (upstream connector API), 0.10.1.1 only if E0 parity is required; version-isolated under `integrations/vllm/<version>/` | connector + integration test | M |
| F2 | Full-attention parity suite: same workloads as Phase A harness; correctness = byte-identical KV after offload+reload (checksum per block) and identical model outputs | parity report | M |
| F3 | Hybrid v1: multi-group registration (per-spec groups), state sidecar snapshot/restore ops, SWA window-bounded storage (store-all v1, window-trim later); enforce **no-partial-caching invariant** (match counts only fully-loadable bundles) | tests with a synthetic hybrid spec | L |
| F4 | Real hybrid model validation on E2 (Qwen3-Next-class if available; else DeepSeek-V4-class per upstream support) | benchmark + correctness report | L |

### Phase G — Hardening and release

| # | Task | Deliverable | Size |
| --- | --- | --- | --- |
| G1 | Sanitizers on CPU paths in CI; race detector on task engine | CI jobs | M |
| G2 | Lifecycle/leak suite: repeated init/shutdown, registration churn, task storms; crash/restart behavior (isolated mode must not take the engine down) | tests | M |
| G3 | Observability: per-phase latency histograms, counters; optional Prometheus export | metrics module | M |
| G4 | Docs: API reference (Doxygen), integration guide, benchmarking guide, ADR index; versioned release + CHANGELOG | v0.1.0 tag | S |

---

## 4. Milestones and gates

| Milestone | Contents | Gate |
| --- | --- | --- |
| M0 ✅ | E0 audit + clean baseline (done) | — |
| M1 | A1–A4: baseline ladder closed | **G1** adopt-vs-build decision |
| M2 ✅ | B complete: skeleton + object model + CI | build/test green |
| M3 ✅ | C+D complete: metadata + control plane, CPU-only | index parity + engine stress pass |
| M4 ✅ | E1–E4: data plane + isolation experiment | **G2** isolation decision with numbers (ADR 0002 final: isolated default, +~50 µs/task premium, fault containment proven) |
| M5 | F1–F2: vLLM parity on full attention | byte-exact reload, workload parity |
| M6 | F3–F4 + G: hybrid + hardening | v0.1.0 |

Sequencing rule: no phase starts before the previous gate is written. If
G1 says "contribute atop upstream," M2–M6 re-scope to the gap list
(hybrid object model, framework-independent core, event-driven control)
layered on upstream rather than parallel — the plan's structure survives,
the substrate changes.

---

## 5. Benchmark policy (all phases)

- One harness (Phase A3 lineage), pinned workloads recorded as JSON; environments recorded (commit, driver, clocks).
- Metrics: match p50/p95/p99; per-task fixed overhead (extrapolated to 0-byte payload); effective GB/s at 4/16/64 blocks; TTFT/TPOT delta vs no-connector; scheduler stall per step.
- Every claim table cites log file + harness commit. Regressions are reported, never dropped.
- SLOs are set **after** M1, from the E2 reference — not invented now.

---

## 6. Non-goals (explicitly deferred)

- Distributed KV reuse across nodes (GMS/Redis, leases at cluster scope) — reference upstream's distributed design instead.
- DP server mode / cross-instance coordination beyond what the adapter inherits from vLLM.
- Mooncake/NIXL RDMA transports (kept behind `ITransport` seam; enabled only on demonstrated need).
- torch-integrated tensor API (descriptors only; torch stays at the adapter).

---

## Appendix A — FlexKV harvest map

| FlexKV asset (v1.0.0) | Disposition |
| --- | --- |
| `csrc/radix_tree.cpp`, `csrc/hash.cpp` (xxHash chained) | **Port** behind `IIndex` (C1) |
| `Mempool` free-list semantics | **Port** to C++ allocators (C3) |
| `TransferOpGraph` shape, worker drain-and-batch loop | **Harvest design** for D1/E3 |
| `csrc/transfer.cu`, `tp_transfer_thread_group.cpp` | **Reference** for strided kernels; port only what benchmarking justifies |
| CUDA IPC registration flow (validated in `cuda_ipc_demo/`) | **Adopt** pattern for `DeviceMemoryRegistry` (E1) |
| ZMQ `KVServer`/`KVDPClient`/`KVTPClient` control plane | **Discard** (fossil of client-server era; replaced by D-plane design) |
| Per-step `try_wait` polling model | **Discard** (replaced by event-driven completion) |
| `vllm_v1_adapter.py` flow (task staging, slot-mapping) | **Reference** semantics for F1 parity |

## Appendix B — Known pitfalls recorded during the audit

- Wall-clock task timers spanning debugger stops produce ~40 ms/stop artifacts; never measure latency under pdb.
- Multi-line first statements double-stop in pdb (CPython line events); harmless.
- `KVManager`'s two endpoint names alias one string in v1.0.0 library mode; treat names as unreliable, trace binds.
- v1.0.0 adapter hardcodes `dp_client_id=0` and has no `dp_size` in config — dp>1 unreachable via vLLM in this snapshot.
- GPU-side per-layer tensors (24 handles) force per-layer copy decomposition (~8 KB × 24 per block); CPU tier should stay one flat pool (BLOCKWISE) regardless.
