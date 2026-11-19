add_library(stratakv_core
  common/types.cpp
  common/model_layout.cpp
  common/handles.cpp)
add_library(stratakv::core ALIAS stratakv_core)

stratakv_target_defaults(stratakv_core)
stratakv_target_warnings(stratakv_core)
stratakv_target_sanitizers(stratakv_core)

target_include_directories(stratakv_core
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

install(TARGETS stratakv_core
  EXPORT stratakv-targets
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}")
