# Third-party sources

| Directory | Upstream | Pin | Form | Used by |
| --- | --- | --- | --- | --- |
| `googletest/` | google/googletest | `v1.14.0` | **Git submodule** (gitlink) | `stratakv_unit_tests` |
| `pybind11/` | pybind/pybind11 | `v2.13.6` | Fetch script clone (convert to submodule next) | `stratakv_bindings` (STRATAKV_PYTHON=ON) |
| `xxhash/` | Cyan4973/xxHash | `v0.8.2` | **Git submodule** (gitlink) | metadata plane chained hashing (Phase C harvest) |

All pins are exact; `scripts/fetch_third_party.sh` materializes the
fetch-script entries and skips anything already present. No
FetchContent/ExternalProject, no package managers — per the engineering
standards in the root README.

Note: while the googletest submodule is absent (e.g., shallow clones of
this repo without `--recurse-submodules` and without network), unit tests
can run against the documented offline bootstrap shim
(`tests/cpp/gtest_shim.h`, `-DSTRATAKV_GTEST_SHIM=ON`). The shim is a
development convenience only; CI and normal development use the pinned
googletest.
