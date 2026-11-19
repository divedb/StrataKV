// GPU-build host-memory backings (Phase E2): cudaMallocHost pools and
// cudaHostRegister page-locking. Compiled into stratakv_data only under
// STRATAKV_GPU (see src/CMakeLists.txt).

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "host_memory.h"

namespace stratakv::data::host_memory {

namespace {

[[noreturn]] void Fail(const char* what, cudaError_t err) {
  throw std::runtime_error(std::string("host_memory: ") + what +
                           " failed: " + cudaGetErrorString(err));
}

}  // namespace

void* AllocatePinned(std::size_t alignment, std::uint64_t bytes) {
  if (bytes == 0) {
    return nullptr;
  }
  // cudaMallocHost returns >=256 B-aligned memory by construction. The
  // config alignment is a hugepage hint for pageable/registered pools;
  // pinned pools never need more than the transport's 8 B word
  // alignment, so requests coarser than 256 B are satisfied as-is.
  (void)alignment;
  void* base = nullptr;
  cudaError_t err = cudaMallocHost(&base, static_cast<std::size_t>(bytes));
  if (err != cudaSuccess) {
    Fail("cudaMallocHost", err);
  }
  return base;
}

void FreePinned(void* base) {
  cudaError_t err = cudaFreeHost(base);
  if (err != cudaSuccess) {
    std::fprintf(stderr, "host_memory: cudaFreeHost: %s\n",
                 cudaGetErrorString(err));
  }
}

void* AllocateRegistered(std::size_t alignment, std::uint64_t bytes) {
  void* base = AllocatePageable(alignment, bytes);
  if (base == nullptr) {
    return nullptr;
  }
  cudaError_t err =
      cudaHostRegister(base, static_cast<std::size_t>(bytes),
                       cudaHostRegisterDefault);
  if (err != cudaSuccess) {
    std::free(base);
    Fail("cudaHostRegister", err);
  }
  return base;
}

void FreeRegistered(void* base) {
  if (base == nullptr) {
    return;
  }
  cudaError_t err = cudaHostUnregister(base);
  if (err != cudaSuccess) {
    std::fprintf(stderr, "host_memory: cudaHostUnregister: %s\n",
                 cudaGetErrorString(err));
  }
  std::free(base);
}

}  // namespace stratakv::data::host_memory
