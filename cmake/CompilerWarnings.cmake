# Target-scoped defaults: C++20, no GNU extensions.
function(stratakv_target_defaults target)
  target_compile_features(${target} PUBLIC cxx_std_20)
  set_target_properties(${target} PROPERTIES CXX_EXTENSIONS OFF)
endfunction()

# Target-scoped warning set (Google style baseline).
function(stratakv_target_warnings target)
  target_compile_options(${target} PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -Wshadow
    -Wnon-virtual-dtor
    -Wold-style-cast
    -Wcast-align
    -Wunused
    -Werror=return-type)
endfunction()
