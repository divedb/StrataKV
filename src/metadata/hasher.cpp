#include "stratakv/metadata/hasher.h"

#include <cstring>
#include <memory>
#include <stdexcept>

#include <xxhash.h>

namespace stratakv::metadata {

/// \brief Owning wrapper around XXH64_state_t (keeps xxhash out of the
/// public header).
struct HasherState {
  XXH64_state_t* xx = nullptr;

  HasherState() {
    xx = XXH64_createState();
    if (xx == nullptr) {
      throw std::runtime_error("Hasher: XXH64_createState failed");
    }
    XXH64_reset(xx, 0);
  }

  ~HasherState() {
    if (xx != nullptr) {
      XXH64_freeState(xx);
    }
  }

  HasherState(const HasherState&) = delete;
  HasherState& operator=(const HasherState&) = delete;
};

Hasher::Hasher() : state_(std::make_unique<HasherState>()) {}

Hasher::~Hasher() = default;

void Hasher::Reset() { XXH64_reset(state_->xx, 0); }

void Hasher::Update(const void* data, std::size_t size_bytes) {
  if (size_bytes == 0) {
    return;
  }
  XXH64_update(state_->xx, data, size_bytes);
}

void Hasher::UpdateTokens(std::span<const common::TokenId> tokens) {
  // Encode as little-endian 8-byte words regardless of host endianness.
  std::byte scratch[8];
  for (common::TokenId token : tokens) {
    const std::uint64_t word = static_cast<std::uint64_t>(token);
    for (int b = 0; b < 8; ++b) {
      scratch[b] = static_cast<std::byte>((word >> (8 * b)) & 0xFF);
    }
    XXH64_update(state_->xx, scratch, sizeof(scratch));
  }
}

BlockHash Hasher::Digest() { return XXH64_digest(state_->xx); }

std::vector<BlockHash> GenBlockHashes(
    std::span<const common::TokenId> tokens,
    std::uint32_t tokens_per_block) {
  if (tokens_per_block == 0) {
    throw std::invalid_argument("GenBlockHashes: tokens_per_block must be > 0");
  }
  const std::size_t num_blocks = tokens.size() / tokens_per_block;
  std::vector<BlockHash> hashes(num_blocks);
  if (num_blocks == 0) {
    return hashes;
  }

  Hasher hasher;
  std::vector<std::byte> scratch(
      static_cast<std::size_t>(tokens_per_block) * 8);
  for (std::size_t i = 0; i < num_blocks; ++i) {
    const std::size_t base =
        i * static_cast<std::size_t>(tokens_per_block);
    for (std::size_t t = 0; t < tokens_per_block; ++t) {
      const std::uint64_t word =
          static_cast<std::uint64_t>(tokens[base + t]);
      for (int b = 0; b < 8; ++b) {
        scratch[t * 8 + static_cast<std::size_t>(b)] =
            static_cast<std::byte>((word >> (8 * b)) & 0xFF);
      }
    }
    hasher.Update(scratch.data(), scratch.size());
    hashes[i] = hasher.Digest();
  }
  return hashes;
}

}  // namespace stratakv::metadata
