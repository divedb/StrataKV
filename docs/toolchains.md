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
