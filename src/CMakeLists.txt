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

stratakv_target_defaults(stratakv_core)
stratakv_target_warnings(stratakv_core)
stratakv_target_sanitizers(stratakv_core)

stratakv_target_defaults(stratakv_metadata)
stratakv_target_warnings(stratakv_metadata)
stratakv_target_sanitizers(stratakv_metadata)

stratakv_target_defaults(stratakv_control)
stratakv_target_warnings(stratakv_control)
stratakv_target_sanitizers(stratakv_control)

target_include_directories(stratakv_core
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

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

install(TARGETS stratakv_core stratakv_metadata stratakv_control
  EXPORT stratakv-targets
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}")
