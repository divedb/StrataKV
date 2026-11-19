# Interface target for the pinned xxHash submodule
# (third_party/xxhash @ v0.8.2; chained prefix hashing, Phase C harvest).
#
# Consumption is single-header style: linkers get XXH_INLINE_ALL and the
# vendored include directory; they then `#include <xxhash.h>`. The header
# is deliberately not part of any installed StrataKV interface — public
# StrataKV headers never include xxhash.h (Phase C keeps the hasher
# dependency private to the metadata plane implementation).

if(NOT EXISTS "${PROJECT_SOURCE_DIR}/third_party/xxhash/xxhash.h")
  message(FATAL_ERROR
    "xxhash submodule not present under third_party/. Run "
    "scripts/fetch_third_party.sh or "
    "`git submodule update --init third_party/xxhash`.")
endif()

add_library(stratakv_xxhash INTERFACE)
add_library(stratakv::xxhash ALIAS stratakv_xxhash)
target_include_directories(stratakv_xxhash INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/third_party/xxhash>)
target_compile_definitions(stratakv_xxhash INTERFACE XXH_INLINE_ALL=1)
