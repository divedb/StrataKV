#!/usr/bin/env bash
# Fetch pinned third-party sources into third_party/.
#
# Preferred form is Git submodules pinned to commits; network egress has
# been intermittent, so googletest and xxhash are real submodules now,
# while pybind11 is vendored in-tree from the pinned v2.13.6 wheel
# (third_party/pybind11/README.md; convert to a submodule on next touch,
# keeping the pin identical).
#
# Pins (do not float):
#   googletest  v1.14.0   (test framework)
#   pybind11    v2.13.6   (Python bindings)
#   xxhash      v0.8.2    (chained prefix hashing; first harvest, Phase C)
set -Eeuo pipefail

readonly ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

# Real Git submodules; initialize them instead of cloning.
if [[ ! -d "${ROOT}/third_party/googletest/.git" ]]; then
  git submodule update --init -- "${ROOT}/third_party/googletest"
fi
if [[ ! -d "${ROOT}/third_party/xxhash/.git" ]]; then
  git submodule update --init -- "${ROOT}/third_party/xxhash"
fi

echo "Done. Reconfigure CMake to pick up the dependencies."
