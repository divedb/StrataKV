# Python bindings target. Requires the pinned pybind11 dependency under
# third_party/ (submodule form preferred; vendored wheel form accepted —
# see third_party/README.md); there is intentionally no offline shim
# for bindings — a partial Python surface would violate the no-silent-
# stubs rule, so the target fails loudly when the dependency is absent.

set(STRATAKV_PYBIND11_DIR "${PROJECT_SOURCE_DIR}/third_party/pybind11")

if (EXISTS "${STRATAKV_PYBIND11_DIR}/CMakeLists.txt")
  # Submodule form: build pybind11 from source.
  add_subdirectory("${STRATAKV_PYBIND11_DIR}"
    "${CMAKE_BINARY_DIR}/third_party/pybind11-build" EXCLUDE_FROM_ALL)
elseif (EXISTS
    "${STRATAKV_PYBIND11_DIR}/pybind11/share/cmake/pybind11/pybind11Config.cmake")
  # Vendored wheel form: pip install tree (headers + CMake config).
  # FindPython with a Development component must come first so pybind11
  # operates in FindPython mode (python3_add_library availability).
  find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
  set(pybind11_DIR
    "${STRATAKV_PYBIND11_DIR}/pybind11/share/cmake/pybind11")
  find_package(pybind11 CONFIG REQUIRED)
else()
  message(FATAL_ERROR
    "STRATAKV_PYTHON=ON requires the pybind11 pin under third_party/. "
    "Run scripts/fetch_third_party.sh or vendor the v2.13.6 wheel per "
    "third_party/pybind11/README.md.")
endif()

pybind11_add_module(stratakv_bindings stratakv_py.cpp)
set_target_properties(stratakv_bindings PROPERTIES OUTPUT_NAME "_core")
# Build the extension INTO the package source tree so `import stratakv`
# resolves the package and its native module together (dev-tree .so is
# gitignored; packaging relocates it properly).
set_target_properties(stratakv_bindings PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/python/stratakv")

stratakv_target_defaults(stratakv_bindings)
stratakv_target_warnings(stratakv_bindings)
target_link_libraries(stratakv_bindings
  PRIVATE stratakv_core stratakv_metadata stratakv_control)
