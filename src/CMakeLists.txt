add_library(stratakv_core
  common/types.cpp
  common/model_layout.cpp
  common/handles.cpp)
add_library(stratakv::core ALIAS stratakv_core)

# Metadata plane (Phase C): chained xxHash, radix index, allocators,
# facade. xxhash stays a private dependency (public headers never
# include xxhash.h).
add_library(stratakv_metadata
  metadata/hasher.cpp
  metadata/radix_index.cpp
  metadata/allocator.cpp
  metadata/metadata_plane.cpp)
add_library(stratakv::metadata ALIAS stratakv_metadata)

# Control plane (Phase D): task engine, completion, executor seam.
add_library(stratakv_control
  control/task_engine.cpp
  control/executor.cpp)
add_library(stratakv::control ALIAS stratakv_control)

# Data plane (Phase E): registry, CPU tier, scheduler; the STRATAKV_GPU
# build adds the CUDA transport, IPC source, and page-locked backings.
set(STRATAKV_DATA_SOURCES
  data/registry.cpp
  data/cpu_tier.cpp
  data/host_memory.cpp
  data/tier_table.cpp
  data/transport.cpp)
if (STRATAKV_GPU)
  list(APPEND STRATAKV_DATA_SOURCES
    data/transport_cuda.cu
    data/ipc_cuda.cu
    data/ipc_transport.cpp
    data/ipc_protocol.cpp
    data/host_memory_cuda.cpp)
endif()
add_library(stratakv_data ${STRATAKV_DATA_SOURCES})
add_library(stratakv::data ALIAS stratakv_data)

# PIC on the static libs: the pybind11 module links them into a shared
# object, and Linux ld rejects non-PIC objects there (macOS tolerates
# them — latent until the first Linux STRATAKV_PYTHON build).
set_target_properties(stratakv_core stratakv_metadata stratakv_control
  stratakv_data PROPERTIES POSITION_INDEPENDENT_CODE ON)

stratakv_target_defaults(stratakv_core)
stratakv_target_warnings(stratakv_core)
stratakv_target_sanitizers(stratakv_core)

stratakv_target_defaults(stratakv_metadata)
stratakv_target_warnings(stratakv_metadata)
stratakv_target_sanitizers(stratakv_metadata)

stratakv_target_defaults(stratakv_control)
stratakv_target_warnings(stratakv_control)
stratakv_target_sanitizers(stratakv_control)

stratakv_target_defaults(stratakv_data)
stratakv_target_warnings(stratakv_data)
stratakv_target_sanitizers(stratakv_data)

if (STRATAKV_GPU)
  # Marks page-locked backings available to CpuTier/host_memory (the
  # CUDA TUs link cudart; host TUs in this target see the define).
  target_compile_definitions(stratakv_data PRIVATE STRATAKV_HAVE_CUDA=1)
  # Direct -std (not the cuda_std_20 feature): CMake 3.22 — the
  # project's declared minimum — predates CUDA 12 toolchains and
  # rejects the feature name against them.
  target_compile_options(stratakv_data
    PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-std=c++20>)
  target_link_libraries(stratakv_data PRIVATE CUDA::cudart)
endif()

target_include_directories(stratakv_core
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

# Isolated-mode transfer worker (E4, ADR 0002): a separate process the
# IpcTransportClient spawns; looked up next to the running executable
# or via $STRATAKV_TRANSFER_WORKER.
if (STRATAKV_GPU)
  add_executable(stratakv_transfer_worker data/transfer_worker_main.cpp)
  stratakv_target_defaults(stratakv_transfer_worker)
  stratakv_target_warnings(stratakv_transfer_worker)
  target_link_libraries(stratakv_transfer_worker PRIVATE
    stratakv::data CUDA::cudart)
endif()

target_link_libraries(stratakv_metadata
  PUBLIC
    stratakv::core
  PRIVATE
    # xxhash is compiled into this target (XXH_INLINE_ALL); installed
    # consumers must not need it.
    $<BUILD_INTERFACE:stratakv::xxhash>)

# The vendored xxhash.h uses old-style casts; suppress that one warning
# for the two TUs that include it (our own code keeps the strict set).
set_source_files_properties(metadata/hasher.cpp PROPERTIES
  COMPILE_OPTIONS "-Wno-old-style-cast")

target_link_libraries(stratakv_control
  PUBLIC
    stratakv::metadata)

target_link_libraries(stratakv_data
  PUBLIC
    stratakv::control)

install(TARGETS stratakv_core stratakv_metadata stratakv_control
  stratakv_data
  EXPORT stratakv-targets
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}")
