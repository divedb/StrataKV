# pybind11 (vendored)

Pinned exactly at **v2.13.6** (the pin recorded in
`scripts/fetch_third_party.sh` history and `third_party/README.md`).

Provenance: the official `pybind11-2.13.6-py3-none-any.whl` from PyPI
(GitHub egress was unavailable; the wheel is the same bits the
`sdist`/repo tag produce). Layout is the standard pip install tree:

- `pybind11/include/` — C++ headers
- `pybind11/share/cmake/pybind11/` — CMake config consumed via
  `find_package(pybind11 CONFIG)` from `bindings/python/CMakeLists.txt`

License: BSD-3-Clause (`pybind11/LICENSE`).

When GitHub egress returns, convert to the preferred Git submodule form
(`git submodule add` at tag v2.13.6, replacing this directory) and keep
the pin identical — see the note in `third_party/README.md`.
