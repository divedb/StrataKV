# Toolchains and reproducibility

## Supported toolchains (Phase B, CPU-only)

| Component | Minimum | Notes |
| --- | --- | --- |
| CMake | 3.22 | Target-based; no global compiler flags |
| Compiler | clang ≥ 15 or gcc ≥ 12 | C++20; developed on Apple clang 14 and gcc 12+ |
| Python | 3.12 | Only for `STRATAKV_PYTHON=ON` bindings |
| CUDA toolkit | 12.8 | Only when `STRATAKV_GPU=ON` (Phase E+); verified against driver 595.71.05 / RTX 3080 Ti (sm_86) |

GPU builds are intentionally out of scope until Phase E; the object model
and metadata plane are CPU-only and must stay buildable on any host.
Exception: the Phase 0 capability baseline
(`benchmarks/gpu_baseline_bench.cu`) builds under `STRATAKV_GPU=ON`
ahead of the full Phase E data plane; see below for the verified local
GPU toolchain and its deviations.

## Third-party pins

See `third_party/README.md`. Pins are exact release tags fetched by
`scripts/fetch_third_party.sh` (shallow clones):

- googletest `v1.14.0`
- pybind11 `v2.13.6`
- xxhash `v0.8.2` (Phase C harvest)

No FetchContent/ExternalProject fetching at configure time, no Conan/vcpkg.

## Offline development (current environment)

GitHub network access was unavailable when Phase B started and is now
restored; googletest has been converted to a real Git submodule (pin
`v1.14.0`), and tests run against genuine GoogleTest by default. Notes:

1. **Unit tests**: default configuration uses the pinned submodule. If the
   submodule is missing and network is unavailable, configure with
   `-DSTRATAKV_GTEST_SHIM=ON` to use the bundled bootstrap shim
   (`tests/cpp/gtest_shim.h`) — development convenience only.
2. **Python bindings**: no shim exists. `STRATAKV_PYTHON=ON` fails loudly
   until `third_party/pybind11` is fetched (script or submodule).
3. **Formatting/lint**: `clang-format` and `clang-tidy` binaries are not
   installed locally; configs are present (`.clang-format`,
   `.clang-tidy`) and enforced in review. Install via brew/apt when
   available.

## Verified local builds

```bash
# Real GoogleTest (default since the googletest submodule landed)
cmake -S . -B build-gtest -DCMAKE_BUILD_TYPE=Debug
cmake --build build-gtest && ctest --test-dir build-gtest

# Sanitizers with real GoogleTest
cmake -S . -B build-gtest-asan -DCMAKE_BUILD_TYPE=Debug \
  -DSTRATAKV_ENABLE_SANITIZERS=ON
cmake --build build-gtest-asan && ctest --test-dir build-gtest-asan

# Offline shim fallback (no submodule, no network)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DSTRATAKV_GTEST_SHIM=ON
cmake --build build && ctest --test-dir build
```

All configurations: 17/17 tests passing (2026-11-19).

## Verified GPU build (WSL2, Phase 0 baseline session, 2026-11-19)

Host: WSL2 kernel 6.6.87.2, RTX 4080 SUPER (sm_89), driver 610.88
(CUDA UMD 13.3), nvcc CUDA 12.0 (V12.0.140), gcc 13.3.0, CMake 3.28.3.

```bash
cmake -S . -B build-gpu -DCMAKE_BUILD_TYPE=Release -DSTRATAKV_GPU=ON
cmake --build build-gpu -j && ctest --test-dir build-gpu
./build-gpu/benchmarks/stratakv_gpu_baseline
```

Deviations and environment contracts (details and evidence in
`benchmarks/phase0_gpu_baseline.md`):

1. **Toolkit 12.0, not 12.8** (12.8 unavailable locally). nvcc 12.0 +
   gcc 13.3 — beyond nvcc 12.0's official gcc ceiling — compiles and
   runs all baseline kernels correctly on this box; treat with mild
   suspicion and keep CUDA >12.0 API surface out of Phase E until the
   toolkit is upgraded or the pin is re-verified.
2. **CMAKE_CUDA_ARCHITECTURES** defaults to `native` (CMake ≥ 3.24;
   here: sm_89). On older CMake, set your SM version explicitly.
3. **WSL2 SM-access constraint**: kernels must not touch
   `cudaHostRegister`'d or pageable host memory here (illegal memory
   access; copy-engine `cudaMemcpyAsync` to the same memory is fine).
   `cudaHostRegister` succeeds and `cudaDeviceReset` cannot recover a
   faulted context on this platform. The baseline harness probes this
   in a forked child before any parent CUDA state exists.
4. **WSL2 CUDA IPC**: handle *export* (`cudaIpcGetMemHandle`) works,
   but *cross-process open* (`cudaIpcOpenMemHandle` in another
   process) fails — verified by a two-way probe (ADR 0002 note).
   Isolated-process transport work must happen on a native-Linux host.
5. Sanitizers are not applied to CUDA targets (nvcc device TUs do not
   support ASan/UBSan); CPU targets keep the sanitizer machinery.
