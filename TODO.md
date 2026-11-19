# Technical investigation and prioritized improvement backlog

Investigation date: 2026-11-20. Documentation only; no implementation changed.

## 1. Evidence, scope, and confidence

StrataKV audited at `2cccd41111a3f507612e6dc6655a3eb693f4725f`.
Upstream repositories were cloned read-only into temporary directories and
inspected at these exact revisions:

- LMCache: `e0e64d0e70b1bb64176b9e9303abdbbb86433104` ([snapshot](https://github.com/LMCache/LMCache/tree/e0e64d0e70b1bb64176b9e9303abdbbb86433104)).
- FlexKV: `542b11c8f22ee74c34ceeb6c8b63db7832e31ece` ([snapshot](https://github.com/taco-project/FlexKV/tree/542b11c8f22ee74c34ceeb6c8b63db7832e31ece)).

These are retrieved source snapshots, **not claims that every feature is in a
stable release or compatible with our vLLM installation**. FlexKV's snapshot
README explicitly describes proposed restore changes requiring a matching
SGLang companion branch. LMCache's live documentation spans legacy in-process
and newer multiprocess architectures; features must be checked per mode.
Neither upstream was built or benchmarked in this investigation.

Labels used below:

- **V — verified implementation fact:** observed directly in source. This can
  establish missing behavior without proving its production impact.
- **D — upstream documented capability:** implementation/design documentation
  supports the claim; not independently reproduced here.
- **R — risk inferred from code:** a specific failure mechanism, requiring a
  regression reproducer before claiming an observed failure.
- **H — optimization hypothesis:** expected benefit depends on measurement.

Priority: P0 correctness/validity prerequisite; P1 high practical impact;
P2 workload-dependent extension; P3 research/deferred. Complexity S/M/L/XL is
relative engineering scope, not a promised duration. Feasibility considers our
single 12-GB remote GPU, existing native Linux IPC tests, and WSL limitations.

## 2. Architectural comparison

| Dimension | LMCache snapshot | FlexKV snapshot | StrataKV today |
| --- | --- | --- | --- |
| Runtime boundary | Standalone MP server plus engine integration; legacy in-process implementation is separate | Direct-call library interfaces, transfer workers and optional in-process/layerwise paths; integration-dependent | C++ metadata/control/data planes; Python facade and single-process vLLM singleton |
| Cache identity | Object identity includes chunk hash, model name and KV rank; grouped objects for hybrid layouts | Chained hashes/radix indexes, namespaces for isolation | Chained XXH64 token-prefix hashes inside one engine; no explicit semantic namespace |
| Storage | L1 managers and pluggable L2, async store/prefetch controllers | CPU/SSD/remote tiers with best-effort inclusive write-through population | CPU pools plus borrowed GPU buffers; SSD enum/interface concepts do not constitute an SSD backend |
| Hybrid reuse | Explicit engine groups versus physical transfer groups; group/rank completeness and window-aware fetch planning | DeepSeek-specific compressed KV/SWA/state groups and mapping validation | Generic FullAttn/MLA/SWA/LinearState types; synthetic tests; real hybrid restore unvalidated |
| Scheduling | Lookup/prefetch handles and storage locks; backend-dependent async behavior | Op graphs, batched transfers, layerwise readiness; bounded remote reads in inspected branch | Graph construction and queue batching; all-layer request readiness; dependencies copied but scheduler does not enforce them |
| GPU transport | Engine-specific native transfer paths and device abstraction | SM and CE paths, host registration boundaries, optional layerwise overlap | CUDA strided kernel with CE fallback; one submission/completion worker and stream pool |
| Operations | Controller/coordinator APIs, storage plugins, cache observability | Multiple engine integrations, routing/events, policies, monitoring | Task/phase metrics and Prometheus text; no service/controller or operational cache lifecycle API |

### LMCache: mechanisms and trade-offs

The MP architecture separates serving-engine lifetime from the cache daemon.
L1 holds reusable objects; controllers move data between L1 and L2 and expose
prefetch completion. This supports sharing and reuse across engine restarts,
but daemon/registration/protocol failures become deployment concerns. A cache
server is a larger operational system than a linked transfer library. Storage
availability, successful allocation, and completed loading must remain distinct.
Sources: [MP architecture](https://docs.lmcache.ai/mp/architecture.html),
[L2 storage](https://docs.lmcache.ai/mp/l2_storage.html).

The inspected storage-manager contract represents each object-group/KV-rank
combination as a row of chunk keys. Prefix fetching folds all required rows
using their individual windows; full and whole-chunk policies have different
completeness rules. Prefetch retains read locks and requires explicit release.
This is relevant to StrataKV's no-partial-caching invariant: completeness is
not simply the minimum of identical block counts once window sizes and physical
block sizes differ. Source: [grouped storage contract](https://github.com/LMCache/LMCache/blob/e0e64d0e70b1bb64176b9e9303abdbbb86433104/docs/design/v1/distributed/storage_manager.md).

LMCache distinguishes engine block-ID spaces from physical transfer identities
including dtype, format, stride, block size and engine group. Several transfer
groups can map to one engine group; shared layers are excluded rather than
copied twice. Logical tokens per block and physical slots per block are separate.
Its design requires cache chunks divisible by participating group block sizes.
This is a more concrete hybrid registration contract than StrataKV currently
implements. It adds mapping and validation complexity, but prevents shape-equal
buffers with different address semantics from being conflated. Source:
[hybrid registration design](https://github.com/LMCache/LMCache/blob/e0e64d0e70b1bb64176b9e9303abdbbb86433104/docs/design/integration/vllm/hybrid-kv-cache-groups.md).

CacheBlend reuses non-prefix text segments with selective recomputation; it is
not equivalent to exact prefix-cache restoration. Adopt only with a separate
quality contract and benchmark suite. Compression must also be mode-qualified:
the inspected MP CacheGen documentation says it is not implemented there and
points to the legacy repository. Do not advertise compression as universally
available based on the project name. Sources:
[CacheBlend](https://docs.lmcache.ai/kv_cache_optimizations/cacheblend.html),
[MP CacheGen status](https://github.com/LMCache/LMCache/blob/e0e64d0e70b1bb64176b9e9303abdbbb86433104/docs/source/mp/cachegen.rst).

### FlexKV: mechanisms and trade-offs

Current FlexKV is not accurately represented by the historical v1.0.0 timing
alone. It has a directly callable library and a mainline vLLM connector path;
layerwise loading can overlap transfer of a later layer with current-layer
computation where supported. Library API placement does not imply every transfer
path is free of IPC. Source: [vLLM guide](https://github.com/taco-project/FlexKV/blob/542b11c8f22ee74c34ceeb6c8b63db7832e31ece/docs/vllm_adapter/README_en.md).

Its tier model is approximately inclusive: put attempts to fill missing lower
tiers, and eviction removes only the selected tier's copy. Eviction itself does
not migrate bytes downward. Lower-tier misses/failures therefore break strict
inclusion. Policies include LRU, LFU, segmented LRU, FIFO, MRU and FILO. This
trades write amplification and duplicate capacity for independent tier hits;
it is not interchangeable with an exclusive victim cache. Source:
[eviction semantics](https://github.com/taco-project/FlexKV/blob/542b11c8f22ee74c34ceeb6c8b63db7832e31ece/docs/eviction_policy/README_en.md).

The inspected restore work prepares admitted requests before launching a batch.
Remote reads have both concurrency and padded-byte budgets; credits and source
leases survive until accepted work drains, including cancellation. CE changes
respect registered host boundaries and use pitched copies where eligible.
These are useful mechanisms to evaluate, but the README labels relevant work
as proposed and companion-dependent; do not assume it works with stock vLLM
or all hybrid adapters. Sources:
[branch status](https://github.com/taco-project/FlexKV/blob/542b11c8f22ee74c34ceeb6c8b63db7832e31ece/README.md),
[performance controls](https://github.com/taco-project/FlexKV/blob/542b11c8f22ee74c34ceeb6c8b63db7832e31ece/docs/cache_tier_performance_controls.md).

DeepSeek-V4 support includes SWA pages and attention/indexer state sidecars,
using physical SWA page IDs and group-specific byte layouts. Layer-ready events
must include relevant state writes. The design records restrictions and model
accuracy validation still needed for some paths; byte movement alone does not
prove recurrent-state semantics. StrataKV should therefore stop treating
“hybrid support” as an upstream absence and focus on a reusable, validated
contract. Source: [state I/O design](https://github.com/taco-project/FlexKV/blob/542b11c8f22ee74c34ceeb6c8b63db7832e31ece/docs/dsv4_compress_state_io_en.md).

## 3. StrataKV strengths and evidence limits

Preserve the framework-independent C++ plane separation, explicit borrowed
buffer descriptors, bundle readiness, pinned matches, failure propagation,
bounded task admission, and CPU sanitizer coverage. The real Qwen2.5 full-
attention two-round test establishes output parity for that configuration.
GPU tests establish byte round trips and recovery through a fresh isolated
worker after SIGKILL. They do not establish general hybrid correctness,
multi-GPU agreement, sustained capacity, or safe reuse during in-flight cancel.

Recorded 4-block native results are about 148/156 microseconds in-process and
196/207 microseconds isolated on the RTX 3080 Ti (768 KiB, synchronous harness).
They are not a comparison against current LMCache/FlexKV, nor proof of TTFT or
TPOT improvements. The ADR's GPU-context memory estimate is not a measured
allocation. Sources: [ADR 0002](docs/adr/0002-transfer-isolation-mode.md),
[validation](docs/phase-g-validation.md), [results](benchmarks/results/2026-11-19_remote_rtx3080ti/e4_isolation_experiment.log).

## 4. Prioritized, actionable gap analysis

### P0 — establish safe ownership and truthful API contracts

- [ ] **C1: Separate logical cancellation from physical transfer completion.**
  **V:** `TaskEngine::Cancel` calls terminal effects immediately, releasing get
  pins; `TransferScheduler::Cancel` removes only unflushed work. Launched CUDA
  copies can continue. **R:** eviction/reallocation can reuse source storage
  while DMA still reads it; cancelled put destinations have the analogous risk
  if callers abort early. Impact: correctness, potentially catastrophic under
  pressure. Complexity L; feasible with deterministic fake transport first.
  Add a physical completion/drained state or transfer lease independent of task
  status; release buffers and reservations only after accepted operations retire.
  Keep immediate user-visible cancellation but delay reuse. Test cancel→evict→
  overwrite with a paused transport, then real CUDA/IPC byte checks. Audit
  partial launch failure and shutdown using the same contract.
  Evidence: `src/control/task_engine.cpp` (`Cancel`, `FinishLocked`, `ApplyEffects`),
  `src/data/transport.cpp` (`Cancel`), `src/data/transport_cuda.cu` (`Run`).

- [ ] **C2: Make failure/exception paths recover reservations.**
  **V:** `PyEngine::SubmitPut` erases `plans_` after submission; `AbortPut`
  requires that plan ID. A later failed task cannot use that public abort path.
  `TaskEngine::Submit` calls executor `Execute` after publishing a live record
  without an exception guard. **R:** a synchronous enqueue exception strands
  an in-flight task/reservation. Complexity M; high feasibility.
  Retain task→reservation ownership until success or physical failure cleanup;
  convert submission exceptions into terminal results with original diagnostics.
  Test pre-enqueue validation failure, partial batch acceptance, worker death,
  repeated failure until pool exhaustion, and full pool conservation afterward.
  Evidence: `bindings/python/stratakv_py.cpp` (`AbortPut`, `SubmitPut`),
  `src/control/task_engine.cpp` (`Submit`). Depends on C1.

- [ ] **C3: Fix Python facade retention and misleading preview semantics.**
  **V:** `ReleaseMatch` releases the pin but leaves `matches_` entries;
  connector cold misses call it repeatedly. `plan_put` is documented as a
  read-only preview but binds to `StagePut`. `submit_get`/`submit_put` release
  the GIL while accessing maps also used by GIL-held methods, without an
  explicit facade mutex. **R:** concurrent Python callers race on these maps;
  existing core TSan tests do not exercise this boundary. Complexity M.
  Erase released matches with a specified repeat-release policy; implement
  actual preview; serialize map ownership independently of the GIL. Test
  million-miss bounded retention, preview pool invariance, and concurrent
  release/submit/shutdown through bindings. Evidence: `bindings/python/stratakv_py.cpp`.

- [ ] **C4: Make metadata staging transactional or enforce one writer.**
  **V:** `PlanLocked` has no encompassing mutex; it separately checks capacity,
  allocates across groups and inserts. On null insertion it clears reserved ID
  vectors without recycling them. The code calls this unreachable under a
  single-scheduler protocol, while the public header describes thread safety.
  **R:** concurrent staging and intermediate allocation exceptions leak or
  violate invariants. Complexity M/L. Choose a documented single-writer
  boundary or transaction lock plus RAII reservation rollback; test identical
  concurrent puts and exhaustion midway through heterogeneous reservations.
  Evidence: `src/metadata/metadata_plane.cpp`, `include/stratakv/metadata/metadata_plane.h`.

- [ ] **C5: Correct vLLM partial-hit/preemption/error contracts.**
  **V:** `get_num_new_matched_tokens` ignores `num_computed_tokens` and returns
  total matched blocks; zero-external allocation is a no-op only when no match
  exists. A failed get raises instead of reporting failed receive/recompute.
  **R:** local-prefix hits, rescheduling, or preemption can overstate new tokens
  or load the wrong range. Complexity M; high impact on production use.
  Track token intervals and allocate/load only the external suffix; explicitly
  gate zero-token loads; wire failed receive metadata and safe recomputation
  after DMA drain. Test local+external hits, exact block-aligned prompts,
  preemption/resume, aborted requests, and injected transfer failure against
  the pinned vLLM contract. Evidence: `integrations/vllm/v0_17/stratakv_connector.py`.

### P1 — usable capacity and general integration

- [ ] **I1: Add capacity-pressure handling and bounded admission.**
  **V:** StagePut throws when pools lack capacity; the connector does not call
  eviction or handle saturation. There is no byte-budget admission across
  transfer queues; task count alone does not bound payload. Impact high for
  long-lived serving; complexity M; feasible on existing GPU.
  Use high/low watermarks, evict only physically unleased bundles, reserve
  bytes before submit, and treat cache-store pressure as a skipped offload
  rather than failed inference. Differentiate temporary pinned capacity from
  permanent oversize. Test >10× pool capacity, mixed request sizes and slow
  transport, recording memory high-water marks, hit rate and p99 latency.
  Evidence: `MetadataPlane::PlanLocked`, connector `request_finished`,
  `TaskEngineConfig`, `TransferScheduler`. Requires C1/C2.

- [ ] **I2: Implement real hybrid registration and state mapping.**
  **V:** metadata uses one token block size for matching/allocation; connector
  takes the first token group's block size. State slots are assigned from an
  adapter free list, not derived from the engine's actual recurrent-state
  allocator; warm load indexes `state_slots[request_id]`. Real vLLM HMA hooks
  (`SupportsHMA`) are absent. Impact high for the project's stated differentiator;
  complexity XL; synthetic work feasible, real large-model coverage hardware-
  constrained. Separate logical token coverage, physical slots, dtype/packed
  bytes, group block-ID spaces and checkpoint boundaries. Require explicit
  recurrent state source/destination mappings. Preserve bundle completeness;
  add window-aware retention only after correct full restore. Reject unsupported
  layouts early. Acceptance: real hybrid cold/warm outputs, state checksums,
  uneven group blocks, mixed dtype, shared layers, window rollover, partial hits.
  Evidence: connector `from_vllm`, `_tokens_per_block`, `assign_state_slot`,
  `update_state_after_alloc`; metadata `PlanLocked`; `TierTable::AddGpuTier`.

- [ ] **I3: Expose actual isolation and multiprocess engine support.**
  **V:** Python `Start()` constructs `CudaStreamTransport` and `kInProcess`
  unconditionally; isolated native tests do not make isolated vLLM available.
  Both connector roles rely on `_SHARED` in one process. Complexity L/XL;
  impact high for deployment compatibility, not automatically speed.
  First expose explicit transport mode with fail-fast unsupported topology;
  then design worker registration/control RPC with generation IDs, ownership
  leases and teardown acknowledgments. Test separate scheduler/worker PIDs,
  worker crash/restart and stale handles. TP/PP require rank-specific layouts,
  completion aggregation and allocation agreement; do not infer them from IPC.
  Evidence: `bindings/python/stratakv_py.cpp::Start`, connector `_SHARED`;
  ADR 0005/0006. Keep ADR/default/config documentation consistent.

- [ ] **I4: Enforce layout and semantic cache identity.**
  **V:** token hashing lacks model/adapter/tenant identity; GPU registration
  primarily passes base+byte size, and tier stride derives from size/capacity.
  **R:** noncontiguous or packed layouts and multiple semantic contexts could
  reuse invalid bytes. Separate-engine use limits current cross-model exposure;
  do not claim a demonstrated cross-tenant leak. Complexity M/L.
  Validate strides, page padding, dtype, quantization metadata and bounds.
  Add versioned namespace(model revision, adapter, positional settings, format,
  TP/PP rank, tenant policy) before shared persistence or multi-model service.
  Test equal tokens under different identities and malformed descriptors.
  Upstream reference: [FlexKV namespaces](https://github.com/taco-project/FlexKV/blob/542b11c8f22ee74c34ceeb6c8b63db7832e31ece/docs/namespace/README_en.md),
  [LMCache ObjectKey](https://github.com/LMCache/LMCache/blob/e0e64d0e70b1bb64176b9e9303abdbbb86433104/lmcache/v1/distributed/api.py).

### P1/P2 — performance changes to measure, not assume

- [ ] **P1: Close the contemporary benchmark ladder before architectural expansion.**
  Priority P1, complexity M/L, feasibility high for full attention. Run current
  pinned LMCache/FlexKV/StrataKV and no-offload vLLM on identical hardware,
  model/dtype, prefix-cache setting, memory budget and payload. Separate useful
  token count from physical transferred bytes. Sweep 1/4/16/64 blocks,
  concurrency 1/4/16, hit rates 0/50/100%, cold/warm, capacity pressure and
  mixed prompt lengths; run ≥50 measured requests after warmup with repeats.
  Record TTFT/TPOT/p50/p95/p99, throughput, CPU utilization, GPU memory, pinned
  bytes, queue/host/device times and errors. Report unsupported combinations.
  Revisit ADR 0001 with evidence; historical 10–20 ms is not a modern baseline.

- [ ] **P2: Prevent submission traffic from starving completions.**
  Priority P1; complexity M; **V:** CUDA `Run()` keeps launching while its
  queue is nonempty and polls events only when empty; otherwise it waits up to
  50 microseconds between checks. **H:** sustained load delays reclamation and
  raises p99/CPU pressure. Poll after a bounded launch budget; compare adaptive
  waits, event callbacks and dedicated polling. Measure queue occupancy and
  completion delay under sustained arrival, not only synchronous one-task loops.
  Evidence: `src/data/transport_cuda.cu::Run`; do not merely delete sleeps.

- [ ] **P3: Add optional layerwise restore and early offload.**
  Priority P2; potential high TTFT benefit for large reusable contexts;
  complexity L. **V:** vLLM forward hooks are no-ops, saves occur at request
  finish and receives become ready after all groups complete. Add layer/group
  completion fences and stream ordering, preserve state dependencies, and
  batch admitted requests before launch. Publish stores only once source KV
  is immutable and complete. Compare full-load and layerwise paths under real
  compute overlap; track TPOT interference as well as TTFT. Requires C1/I2.

- [ ] **P4: Make CE/SM transfer policy measurable and selectable.**
  Priority P2; complexity M/L. **V:** strided kernels are used for SM-accessible
  memory, CE fallback loops over IDs/layer chunks. **H:** many small copies
  cost launches; SM copies may contend with attention, while CE may increase
  CPU packing traffic. Add explicit direction-specific policy, contiguous-run
  coalescing and eligible pitched copies; keep registration-boundary checks.
  Benchmark fragmented and contiguous IDs under simultaneous inference.
  Cache launch descriptors only with registration generation/lifetime checks.
  Evidence: `src/data/transport_cuda.cu::LaunchBatch`, `GroupAddressing`.

- [ ] **P5: Resolve graph semantics before multistage storage.**
  Priority P1 before SSD/layerwise, complexity M. **V:** task engine copies
  dependencies but scheduler submits only `graph.ops()`. Reject nonempty
  dependencies now or implement ready-node scheduling with predecessor failure
  propagation and cancellation. Test SSD→CPU→GPU/state ordering with delayed
  fake operations. Today's independent GPU/CPU groups do not prove DAG support.

- [ ] **P6: Measure instrumentation and metadata overhead.**
  Priority P2; complexity S/M; high feasibility. **V:** hashes are rebuilt from
  full token spans; phase histograms take mutexes and clocks, and task completion
  performs metadata effects under the engine lock. **H:** contention may dominate
  small transfers or high concurrency. Add instrumentation-on/off benchmark,
  lock-wait/device-event measurements, bytes and hit/miss/eviction counters.
  Profile before adding incremental hashing or sharded histograms. Preserve
  atomic publication of metadata readiness; simply moving effects outside the
  lock would regress a previously fixed race. Sources: hasher, TaskEngine,
  `common/metrics.h`, existing benchmark logs.

### P2/P3 — capacity tiers and broader functionality

- [ ] **E1: Introduce one durable local tier behind an object-store API.**
  Priority P2; expected high capacity benefit only when CPU misses dominate;
  complexity L. `ITransport` is a byte-movement seam, not a persistent index,
  ownership or recovery protocol. Define key/schema/version, checksum, atomic
  publish, eviction policy and recovery first; start with file/NVMe asynchronous
  reads into pinned staging. Prefetch under byte budgets; preserve locks until
  data is resident. Compare cost of loading with saved prefill compute. Evaluate
  io_uring/GDS only after a measured bottleneck and device support check.
  Decide inclusive write-through versus victim storage explicitly. Requires
  C1–C4, I4, P5. Keep disk full, restart, torn writes and checksum failure tests.

- [ ] **E2: Add workload-aware retention/admission before remote scale-out.**
  Priority P2; complexity M. Compare LRU with segmented LRU/admission filtering
  on repeated-prefix versus one-off long prompts. Charge bytes, not just blocks,
  across unequal groups; pin complete bundles. Measure hit-token rate, avoided
  compute, bytes written and tail latency. A policy catalogue without trace
  evidence is not a performance win.

- [ ] **E3: Distributed reuse / PD disaggregation as an explicit scope change.**
  Priority P3, complexity XL, limited local feasibility. LMCache/FlexKV provide
  mechanisms StrataKV intentionally defers. Reuse a mature backend such as NIXL
  or Mooncake rather than inventing RDMA/control services; require namespace,
  remote leases, credit limits, retries, cancellation and rank compatibility.
  KV-aware routing and cache events become useful only with multiple engines.
  Obtain multi-node correctness/performance evidence before changing non-goals.

- [ ] **E4: Compression and non-prefix reuse remain separate experiments.**
  Priority P3, complexity L/XL. Lossless compression is worthwhile only when
  reduced transport/storage cost exceeds codec latency and GPU contention.
  Measure ratio and end-to-end performance on real KV; packed low-precision KV
  needs scale/format metadata. CacheBlend-style selective recomputation changes
  the exactness contract: use quality evaluations and full recomputation as the
  reference. Do not mix approximate reuse into byte-exact prefix-cache tests.

## 5. Recommended execution order and decision gates

1. C1–C5 reproducers and ownership fixes; clarify supported concurrency/layouts.
2. I1 capacity behavior and I4 identity/descriptor validation; run sustained tests.
3. P1 contemporary baseline ladder, including current topology limitations.
4. I2 real hybrid validation and I3 explicit process support, sized to hardware.
5. P2/P4 experiments, followed by P3 only when transfer is on the critical path.
6. P5 then E1 if measured CPU miss cost warrants it; E2 trace-driven tuning.
7. E3/E4 require workload evidence and an explicit scope/quality decision.

Release criteria should include bounded memory under repeated miss/failure,
physical DMA lifetime safety, a supported vLLM version/topology matrix, sustained
capacity tests, real hybrid output parity, and a written current-baseline
comparison. Passing unit tests or having a metrics exporter alone does not close
those gates. No numerical speedup is predicted by this document.
