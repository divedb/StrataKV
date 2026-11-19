# Target-scoped defaults: C++20, no GNU extensions.
function(stratakv_target_defaults target)
  target_compile_features(${target} PUBLIC cxx_std_20)
  set_target_properties(${target} PROPERTIES CXX_EXTENSIONS OFF)
endfunction()

# Target-scoped warning set (Google style baseline). CUDA TUs get the
# host-compiler subset through -Xcompiler; device-code warnings come
# from nvcc defaults (nvcc rejects raw gcc flags on its own line).
function(stratakv_target_warnings target)
  target_compile_options(${target} PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:
      -Wall
      -Wextra
      -Wpedantic
      -Wshadow
      -Wnon-virtual-dtor
      -Wold-style-cast
      -Wcast-align
      -Wunused
      -Werror=return-type>
    $<$<COMPILE_LANGUAGE:CUDA>:
      -Xcompiler=-Wall,-Wextra,-Wshadow,-Wunused,-Werror=return-type>)
endfunction()
