#include "host_memory.h"

#include <cstdlib>

namespace stratakv::data::host_memory {

void* AllocatePageable(std::size_t alignment, std::uint64_t bytes) {
  if (bytes == 0) {
    return nullptr;
  }
  void* base = nullptr;
  // posix_memalign (no size-multiple constraint, unlike
  // std::aligned_alloc).
  if (::posix_memalign(&base, alignment,
                       static_cast<std::size_t>(bytes)) != 0 ||
      base == nullptr) {
    throw std::runtime_error("host_memory: pageable reservation failed");
  }
  return base;
}

void FreePageable(void* base) { std::free(base); }

}  // namespace stratakv::data::host_memory
