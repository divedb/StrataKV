#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "stratakv/common/types.h"

namespace stratakv::metadata {

/// \brief Chained block hash (XXH64; FlexKV `HashType` parity).
using BlockHash = std::uint64_t;

struct HasherState;

/// \brief Streaming XXH64 hasher with seed 0 (port of FlexKV `Hasher`).
///
/// Backed by the pinned xxHash v0.8.2 submodule (third_party/xxhash), the
/// same library FlexKV links, so digests are equal by construction. The
/// dependency is private to the implementation; this header exposes no
/// xxhash types. Not thread-safe: one Hasher per thread (cheap to
/// construct).
class Hasher {
 public:
  Hasher();
  ~Hasher();

  Hasher(const Hasher&) = delete;
  Hasher& operator=(const Hasher&) = delete;

  /// \brief Restart a fresh digest (seed 0).
  void Reset();

  /// \brief Absorb raw bytes.
  void Update(const void* data, std::size_t size_bytes);

  /// \brief Absorb tokens encoded as 8-byte little-endian words.
  ///
  /// The little-endian encoding is the FlexKV contract (its runtime hosts
  /// are little-endian) and is applied regardless of host endianness so
  /// digests are portable.
  void UpdateTokens(std::span<const common::TokenId> tokens);

  /// \brief Finish and return the digest (state may then be Reset).
  BlockHash Digest();

 private:
  std::unique_ptr<HasherState> state_;
};

/// \brief Chained block hashes for a token sequence (FlexKV `gen_hashes`
/// port).
///
/// Contract: block i is full only when the sequence covers (i+1) *
/// tokens_per_block tokens; hash(i) = XXH64 over the bytes of tokens
/// [0, (i+1) * tokens_per_block) — i.e. hash(i) covers ALL tokens up to
/// and including block i (the chain is never reset between blocks).
/// Trailing tokens that do not fill a block are ignored.
///
/// \throws std::invalid_argument if tokens_per_block == 0.
std::vector<BlockHash> GenBlockHashes(
    std::span<const common::TokenId> tokens,
    std::uint32_t tokens_per_block);

}  // namespace stratakv::metadata
