# Hardening and release validation

Use separate CPU build directories for each sanitizer:

```sh
cmake -S . -B build-asan -DCMAKE_BUILD_TYPE=Debug -DSTRATAKV_ENABLE_SANITIZERS=ON
cmake --build build-asan -j 4
ctest --test-dir build-asan --output-on-failure --timeout 180
cmake -S . -B build-tsan -DCMAKE_BUILD_TYPE=Debug -DSTRATAKV_ENABLE_TSAN=ON
cmake --build build-tsan -j 4
TSAN_OPTIONS=halt_on_error=1 ctest --test-dir build-tsan --output-on-failure --timeout 180
```

TSan cannot be combined with ASan/UBSan or CUDA. Its runtime needs a
compatible virtual address space; an `unexpected memory mapping` failure
before tests start is an environment failure, not a successful race check.
Do not suppress reports to turn a failed check green.

The C++ suite covers concurrent submission/snapshot/drain/reclamation,
completion/cancellation races, dispatcher replacement during dispatch,
repeated engine construction/shutdown, registration churn and stale IPC
handles. Existing tests cover 10k-task storms and GPU worker SIGKILL,
terminal failure propagation, post-crash submissions failing promptly, and explicit recovery using a fresh
worker with the engine-owned allocations still alive.
Run the GPU suite on native Linux with CUDA IPC support; CPU simulation
cannot establish crash containment or byte parity.

Release remains gated on Phase A's baseline decision and Phase F4's real
hybrid-model validation. CI configuration is not evidence of a completed
hosted CI run. Keep local/remote commands, logs and environment details
alongside any release claims; tag v0.1.0 only after the outstanding gates
are satisfied.
