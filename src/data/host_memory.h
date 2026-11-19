// Internal allocation seam for CpuTier pool backings (Phase E2).
//
// CPU-only builds link only the pageable implementations; the
// STRATAKV_GPU build additionally compiles host_memory_cuda.cpp and
// defines STRATAKV_HAVE_CUDA for this target, enabling the page-locked
// backings. CpuTier selects by CpuTierConfig::backing and fails loudly
// on backings its build cannot honor.

#pragma once

#include <cstddef>
#include <cstdint>

#include "stratakv/data/cpu_tier.h"

namespace stratakv::data::host_memory {

/// \brief Pageable reservation (posix_memalign; any build).
/// \throws std::runtime_error on allocation failure.
void* AllocatePageable(std::size_t alignment, std::uint64_t bytes);
void FreePageable(void* base);

#if defined(STRATAKV_HAVE_CUDA)

/// \brief cudaMallocHost reservation (SM-accessible page-locked memory;
/// the only kernel-safe host backing on WSL2 — Phase 0 finding F4).
/// \throws std::runtime_error with the CUDA error string on failure.
void* AllocatePinned(std::size_t alignment, std::uint64_t bytes);
void FreePinned(void* base);

/// \brief Aligned heap reservation + cudaHostRegister page-lock.
/// Copy-engine accessible on all platforms; SM access is platform-
/// dependent (faults on WSL2 — benchmark F4), so tiers using this
/// backing are routed through copy-engine transfers by the transport.
/// \throws std::runtime_error if allocation or registration fails.
void* AllocateRegistered(std::size_t alignment, std::uint64_t bytes);
void FreeRegistered(void* base);

#endif  // STRATAKV_HAVE_CUDA

}  // namespace stratakv::data::host_memory
