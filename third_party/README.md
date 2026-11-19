# Third-party sources

Pinned via `scripts/fetch_third_party.sh` (shallow clones at exact release
tags). Preferred long-term form: Git submodules with the identical pins,
converted when GitHub network access is available (the repo currently
carries `.gitmodules` as the source of truth for URLs and paths).

| Directory | Upstream | Pin | Used by |
| --- | --- | --- | --- |
| `googletest/` | google/googletest | `v1.14.0` | `stratakv_unit_tests` |
| `pybind11/` | pybind/pybind11 | `v2.13.6` | `stratakv_bindings` (STRATAKV_PYTHON=ON) |
| `xxhash/` | Cyan4973/xxHash | `v0.8.2` | metadata plane chained hashing (Phase C harvest) |

No FetchContent/ExternalProject, no package managers — per the engineering
standards in the root README.

Note: while the googletest submodule is unavailable, unit tests can run
against the documented offline bootstrap shim
(`tests/cpp/gtest_shim.h`, `-DSTRATAKV_GTEST_SHIM=ON`). The shim is a
development convenience only; CI uses the pinned googletest.
