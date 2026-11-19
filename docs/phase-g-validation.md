# Phase G validation — 2026-11-20

Source: Phase G changes based on `ab4f03d`; the commit containing this
report records the tested implementation. Logs are under
[`benchmarks/results/2026-11-20_phase_g`](../benchmarks/results/2026-11-20_phase_g/).
These are correctness checks, not performance measurements.

| Check | Environment | Result | Log |
| --- | --- | --- | --- |
| CPU native + Python + fake connector | local, GCC 13.3, Python 3.12, Debug | 3/3 CTest suites passed | `local-cpu.log` |
| ASan + UBSan + leak detection | local, outside sandbox | 94 C++ tests passed | `local-asan.log` |
| ThreadSanitizer | native Linux 5.4, GCC 12.3 | 94 C++ tests passed, no suppressions | `remote-tsan12.log` |
| GPU + native + Python + connector | RTX 3080 Ti, CUDA 12.8, vLLM 0.34, torch 2.14 | 4/4 CTest suites passed, 49.00 s | `remote-gpu-vllm.log` |
| C++ API generation | Doxygen 1.9.1, CMake 3.22.1 | HTML index generated | `remote-doxygen.log` |

The GPU run includes real byte copies, isolated-worker SIGKILL and explicit
fresh-worker recovery, and two-round Qwen2.5-0.5B-Instruct output parity
with native prefix hits. No local GPU result is claimed: the local CUDA
driver/runtime combination is currently incompatible.

ASan initially ran all tests but LeakSanitizer failed under sandbox ptrace;
the unsandboxed rerun passed with leak detection enabled. Local GCC 13 TSan
could not initialize (`unexpected memory mapping`). Native Linux GCC 11
TSan reported a double mutex lock during the concurrent completion/wait
test (`remote-tsan11-failed.log`); GCC 12 passes the complete suite. CI uses
GCC 12 on Ubuntu 22.04 to match that successful configuration. This report
does not claim a successful hosted CI run or establish the cause of the
older runtime's report.

New regression coverage exercises concurrent completion/cancellation,
coherent metrics while submitting/reclaiming 4,000 tasks, bounded result
retention, explicit failure/rejection/cancellation accounting, dispatcher
replacement while dispatching, 100 engine lifecycles and 100 registration
cycles. Python checks the public metrics/reclamation API and Prometheus
text output. Task histograms cover accepted submission to terminal state;
metadata-match and transport-stage instrumentation remains open.

Release remains pending the baseline decision, real hybrid-model
validation, and finer phase instrumentation. See [roadmap](../roadmap.md).
