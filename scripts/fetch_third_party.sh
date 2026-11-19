#!/usr/bin/env bash
# Fetch pinned third-party sources into third_party/.
#
# Preferred form is Git submodules pinned to commits; this environment
# currently has no GitHub network access, so the same pins are expressed
# as shallow clones at exact release tags. When network is available,
# convert to `git submodule add` and keep the pins identical.
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

fetch_pinned googletest https://github.com/google/googletest.git v1.14.0
fetch_pinned pybind11   https://github.com/pybind/pybind11.git   v2.13.6
fetch_pinned xxhash     https://github.com/Cyan4973/xxHash.git   v0.8.2

echo "Done. Reconfigure CMake to pick up the dependencies."
