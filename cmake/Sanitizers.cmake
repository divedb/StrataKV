# Target-scoped sanitizers (CPU paths only; GPU targets are excluded in
# Phase E when STRATAKV_GPU is introduced).
function(stratakv_target_sanitizers target)
  if (STRATAKV_ENABLE_SANITIZERS)
    message(STATUS "Sanitizers enabled for ${target}")
    target_compile_options(${target} PRIVATE
      -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(${target} PRIVATE -fsanitize=address,undefined)
  endif()
endfunction()
