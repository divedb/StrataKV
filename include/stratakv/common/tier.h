#pragma once

#include <cstdint>
#include <string>

namespace stratakv::common {

/// \brief Declarative description of one storage tier.
///
/// GPU tiers reference engine-owned memory (registered, never allocated by
/// StrataKV); CPU/SSD tiers are owned by StrataKV and backed by one flat
/// pool per group (see roadmap Phase E / ADR 0005 for ownership rules).
struct TierSpec {
  enum class Kind : std::uint8_t { kGpu, kCpu, kSsd };

  Kind kind = Kind::kCpu;
  /// \brief Endpoint or path: registration endpoint for GPU tiers, pool
  /// file path for SSD tiers, empty for in-process CPU tiers.
  std::string identifier;
  std::uint64_t capacity_blocks = 0;  ///< Blocks per group; > 0 to be valid.

  /// \brief Structural validity.
  bool IsValid() const {
    return capacity_blocks > 0 &&
           (kind != Kind::kSsd || !identifier.empty());
  }
};

}  // namespace stratakv::common
