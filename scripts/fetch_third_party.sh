#!/usr/bin/env bash
# Fetch pinned third-party sources into third_party/.
#
# Preferred form is Git submodules pinned to commits. Network egress has
# since been restored, so googletest and xxhash are real submodules now;
# pybind11 is still materialized as a shallow clone at its exact release
# tag (convert to a submodule on next touch, keeping the pin identical).
#
# Pins (do not float):
#   googletest  v1.14.0   (test framework)
#   pybind11    v2.13.6   (Python bindings)
#   xxhash      v0.8.2    (chained prefix hashing; first harvest, Phase C)
set -Eeuo pipefail

readonly ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

fetch_pinned() {
  local dir="$1" url="$2" tag="$3"
  if [[ -d "${ROOT}/third_party/${dir}/.git" ]]; then
    echo "third_party/${dir} already present, skipping"
    return 0
  fi
  echo "Fetching ${dir} @ ${tag}"
  git clone --depth 1 --branch "${tag}" "${url}" "${ROOT}/third_party/${dir}"
}

# Real Git submodules; initialize them instead of cloning.
if [[ ! -d "${ROOT}/third_party/googletest/.git" ]]; then
  git submodule update --init -- "${ROOT}/third_party/googletest"
fi
if [[ ! -d "${ROOT}/third_party/xxhash/.git" ]]; then
  git submodule update --init -- "${ROOT}/third_party/xxhash"
fi
fetch_pinned pybind11   https://github.com/pybind/pybind11.git   v2.13.6

echo "Done. Reconfigure CMake to pick up the dependencies."
