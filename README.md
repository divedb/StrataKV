A High-Performance C++20 KV Cache Runtime for LLM Inference
Status: design proposal and implementation prompt. This README is not a claim that FluxKV has been implemented or benchmarked. The intended repository location is ~/Desktop/FlexKV/README.md.

Mission
Build a modular, production-oriented KV-cache runtime with a standalone C++20 core, a pybind11 API, and a thin Python adapter for vLLM. Do not blindly rewrite FlexKV. Audit the checked-out version and preserve mature native implementations. Focus on measured latency, task completion, metadata lookup, transfer orchestration, and interoperability.
Research findings (reviewed November 2026)
1. FlexKV upstream has evolved. Its current README says it switched from client-server to a directly callable library in November 2025, and its vLLM connector landed in vLLM mainline in March 2026 (vLLM 0.17.2 onward). The earlier claim that all calls must cross a separate transfer process may only apply to an older checkout or a particular configuration. Inspect the local Git commit and runtime topology first. FlexKV upstream.
2. PagedAttention / vLLM establishes block-oriented KV memory management and sharing; the connector must respect block mappings, ownership and scheduler semantics. Kwon et al., SOSP 2023.
3. Mooncake demonstrates a KV-centric architecture, disaggregated prefill/decode, tiered memory/storage and transfer-aware scheduling. Its design motivates separating metadata decisions from asynchronous movement, not necessarily moving everything in-process. Qin et al., arXiv:2407.00079; FAST '25.
4. LMCache demonstrates modular connector boundaries, batched transfers, compute/I/O overlap and explicit KV-cache management APIs across storage tiers. Cheng et al., arXiv:2510.09665; source.
5. FlashOffloading (November 2026) explores host-DRAM-centric offloading, sparse lookup and overlapping compute with loads. Treat its results as workload- and hardware-specific, not guaranteed gains for FluxKV. arXiv:2611.07542.
6. NIXL / Mooncake Transfer Engine are useful transport design references; compare their capabilities and interfaces before introducing another transport abstraction. NIXL repository; Mooncake repository.
What the reported trace does—and does not—prove
An earlier trace reportedly showed prefix match times of 40.26 / 40.60 / 40.98 ms for 4–5 blocks and task execution of 250–276 ms for approximately 0.75–1 MB. These are user-provided historical observations, not independently reproduced benchmarks. The near-constant match latency suggests fixed overhead (e.g. polling, synchronization, queueing or IPC), but does not prove a specific root cause. The rough end-to-end effective rate of ~3–4 MB/s is not equivalent to measured PCIe bandwidth; queueing and completion time may dominate. Investigate before claiming a speedup or choosing a rewrite.
Proposed architecture
vLLM EngineCore / Scheduler (Python)
    |
    vLLM KVConnector adapter (thin, version-isolated Python)
    |
    pybind11 facade (ownership / exception / GIL policy)
    |
    +----------------- FluxKV C++20 Core ------------------+
    | Metadata plane: prefix index, block lookup, leases   |
    | Control plane: task engine, queue, completion events  |
    | Data plane: transfer scheduler, GPU/CPU/SSD backends  |
    | Observability: tracing, metrics, latency histograms   |
    +-------------------------------------------------------+
                          |
              Optional transport backends
          CUDA streams / SSD / NIXL / Mooncake
The C++ core must not depend on Python, pybind11, vLLM or libtorch. Bindings depend on the core; the Python connector depends on bindings. Keep metadata lookup logically distinct from data movement. Do not assume that in-process GPU transfer is always superior: benchmark it against isolated processes and document fault isolation, CUDA context, stream synchronization and ownership tradeoffs.
Engineering standards
- Language: C++20, following the Google C++ Style Guide. Use RAII, explicit ownership, const correctness, move semantics and clear error handling. Prefer composition and small interfaces. Avoid unnecessary heap allocation, global mutable state, and unmeasured abstractions.
- Documentation: English Doxygen triple-slash comments for every public class, function, method and interface. Use /// \brief, /// \param, /// \return, and /// \tparam where appropriate. Specify thread safety, ownership, buffer lifetime and asynchronous completion semantics.
- Build: modern target-based CMake, explicit C++20, separate core/bindings/tests/integrations targets, Debug/Release, install/export support and options for optional GPU, Python and vLLM components. No global include paths or global compiler flags when target-scoped equivalents exist.
- Dependencies: third-party source repositories in third_party/ as Git submodules pinned to commits. Prefer add_subdirectory() where supported. No FetchContent/ExternalProject network fetch during configuration; no Conan/vcpkg. Document system dependencies (CUDA toolkit, Python, drivers) and reproducible toolchains separately; do not vendor SDKs indiscriminately.
- Python: pybind11 with a stable Pythonic API, type hints and clear exceptions. Release the GIL around blocking native work only when safe; reacquire it before touching Python objects. Do not make Python callbacks on arbitrary background threads without a documented dispatch strategy.
- GPU buffers: avoid libtorch ABI dependencies in the core. Use explicit device-memory descriptors, sizes, device IDs and stream/context contracts. py::array_t is for compatible host arrays; it is not a general zero-copy CUDA tensor interface. Validate buffer lifetimes, device identity, synchronization and IPC handle ownership.
- Quality: GoogleTest, pytest, clang-format (Google style), clang-tidy, sanitizer-enabled CPU builds, GPU tests where supported, benchmarks, CI, and no silent stubs.
Documentation example
/// \brief Submits an asynchronous KV cache transfer.
///
/// The source and destination buffers must remain valid until completion.
/// The caller is responsible for satisfying device and stream contracts.
///
/// \param source Non-owning descriptor for the source device buffer.
/// \param destination Non-owning descriptor for the destination buffer.
/// \return Handle for querying or awaiting completion.
TaskHandle SubmitTransfer(const DeviceBuffer& source,
                          const DeviceBuffer& destination);
Suggested repository structure
CMakeLists.txt
cmake/
include/fluxkv/{cache,task,transfer,metadata}/
src/{cache,task,transfer,metadata}/
bindings/python/
python/fluxkv/
integrations/vllm/
tests/cpp/
tests/python/
benchmarks/
third_party/
docs/
This is a proposal, not a claim that these paths already exist.
Profiling and acceptance plan
Phase 0 — Audit and baseline (mandatory). Record git rev-parse HEAD, vLLM/PyTorch/CUDA versions, GPU/CPU/SSD topology and whether the running configuration uses separate processes. Trace get_num_new_matched_tokens, get_match, SharedOpPool, _process_empty_graph, task queue, wakeup, transfer enqueue, CUDA event completion and callback delivery only if these symbols exist in the checked-out version. Instrument monotonic timestamps at each boundary. Test hit/miss, block counts, payload sizes, concurrent requests and cold/warm cache. Collect p50/p95/p99, scheduler stall time, bytes transferred, actual copy duration and effective throughput. Form and falsify a fixed-poll-interval hypothesis.

Phase 1 — Minimal fixes. Remove measured unnecessary sleeps, extra round trips or per-block Python boxing. Prefer batching, direct metadata lookup and explicit notifications. Re-run identical workloads before redesigning components.

Phase 2 — C++ task engine. Migrate task state machines and completion queue to C++20 if profiles justify it. Use event-driven notification (condition_variable, eventfd or event-loop-compatible mechanism as appropriate); distinguish thread synchronization from cross-process notification. Keep vLLM scheduler mutations on its permitted execution context. Ensure cancellation, shutdown, timeout, error propagation and backpressure are correct.

Phase 3 — Transfer architecture experiments. Compare current architecture with optional in-process transfer threads and CUDA streams. Measure throughput, TTFT, tail latency, memory overhead, contention, isolation, and failure behavior. Keep process isolation when operational benefits outweigh latency gains.

Phase 4 — Stable API and integrations. Add pybind11 bindings and a thin version-isolated Python KVConnector; prefer the supported upstream vLLM connector where feasible. Make C++ core independently testable and usable by other inference frameworks.

Phase 5 — Production hardening. Concurrency stress tests, sanitizer checks where applicable, lifecycle and resource-leak tests, crash/restart behavior, metrics, CI and reproducible benchmarks.
Benchmark acceptance criteria
Set numeric SLOs after establishing a reproducible baseline. Do not invent speedup targets. Compare the same hardware, model, prompt distribution, cache-hit rate, payload sizes, concurrency, transfer mode and software versions. Report both wins and regressions; a lower microbenchmark latency is not sufficient if TTFT, TPOT, throughput or reliability regress.
Implementation prompt for a coding agent
You are a senior C++ systems engineer specializing in LLM inference, CUDA data movement, asynchronous runtimes, and Python interoperability.

Build FluxKV, a high-performance C++20 KV-cache runtime, by incrementally modernizing and reusing the existing FlexKV implementation rather than performing a blanket rewrite.

Before changing code, inspect the actual repository commit and architecture. Identify which components already run in C++/CUDA, how vLLM integration works in the checked-out version, whether IPC or polling remains in the critical path, and what existing tests and benchmarks prove. Explicitly compare local behavior with current upstream FlexKV, which reports a directly callable library design and upstream vLLM integration. Do not rely on historical file names or assumptions without verification.

Research and design: use PagedAttention, Mooncake, LMCache, FlashOffloading and NIXL as references, clearly separating their published findings from design hypotheses. Propose a metadata/control/data-plane decomposition, stable task and buffer APIs, event-driven completion, optional transport backends and a thin vLLM Python connector. Explain tradeoffs around in-process versus out-of-process transfers, GIL handling, CUDA contexts, lifetime and thread safety.

Mandatory standards: strictly follow Google C++ Style and modern C++20; use target-based CMake; manage third-party source dependencies as pinned Git submodules under third_party/; use English Doxygen /// \brief, /// \param, /// \return, /// \tparam comments for public APIs; use pybind11 only at the Python boundary; keep the core independent of Python, vLLM and libtorch; use GoogleTest, pytest, clang-format, clang-tidy and appropriate sanitizers.

Execution order: (1) audit and baseline; (2) identify and fix the root cause of measured fixed latency; (3) implement a native event-driven task engine only if justified; (4) benchmark optional in-process transfers against the existing architecture; (5) expose a stable pybind11 API; (6) integrate through a thin, version-aware vLLM Python adapter; (7) harden and document. Implement in small, testable increments and preserve existing behavior unless changes are explicitly justified.

Deliverables: architecture decision record, version/commit audit, performance traces and reproducible benchmark harness, CMake targets, Git submodules, C++20 implementation, Python bindings/type hints, vLLM adapter, unit/integration tests, build/run documentation and measured before/after results. Do not claim performance improvements without evidence, invent unavailable APIs, or hide incomplete work behind placeholders.

References
- FlexKV upstream
- vLLM / PagedAttention (SOSP 2023)
- Mooncake (arXiv:2407.00079)
- Mooncake (FAST 2025)
- LMCache (arXiv:2510.09665)
- FlashOffloading (arXiv:2611.07542)
- NIXL
- Google C++ Style Guide