// Phase C (C1): chained xxHash contract tests. The vendored xxHash
// v0.8.2 is the same library FlexKV links, so digests are equal by
// construction; these tests pin the CHAINING semantics (hash(i) covers
// tokens <= block i, LE token encoding, never reset between blocks)
// against independently computed one-shot digests, plus the canonical
// XXH64 empty-input constant.

#ifdef STRATAKV_GTEST_SHIM
#include "gtest_shim.h"
#else
#include <gtest/gtest.h>
#endif

#include <cstdint>
#include <span>
#include <vector>

#include <xxhash.h>

#include "stratakv/common/types.h"
#include "stratakv/metadata/hasher.h"

namespace {

using stratakv::common::TokenId;
using stratakv::metadata::BlockHash;
using stratakv::metadata::GenBlockHashes;
using stratakv::metadata::Hasher;

std::vector<std::byte> LeBytes(std::span<const TokenId> tokens) {
  std::vector<std::byte> bytes;
  bytes.reserve(tokens.size() * 8);
  for (TokenId token : tokens) {
    const std::uint64_t word = static_cast<std::uint64_t>(token);
    for (int b = 0; b < 8; ++b) {
      bytes.push_back(static_cast<std::byte>((word >> (8 * b)) & 0xFF));
    }
  }
  return bytes;
}

BlockHash OneShot(std::span<const TokenId> tokens) {
  const std::vector<std::byte> bytes = LeBytes(tokens);
  return XXH64(bytes.data(), bytes.size(), /*seed=*/0);
}

TEST(HasherTest, CanonicalEmptyDigest) {
  // Canonical XXH64("", seed 0) — the reference constant.
  EXPECT_EQ(XXH64("", 0, 0), 0xEF46DB3751D8E999ULL);
  Hasher hasher;
  EXPECT_EQ(hasher.Digest(), 0xEF46DB3751D8E999ULL);
}

TEST(HasherTest, StreamingMatchesOneShot) {
  const std::vector<TokenId> tokens = {1, 2, 3, -4, 0, 151877, 7, 8};
  Hasher hasher;
  hasher.UpdateTokens(tokens);
  EXPECT_EQ(hasher.Digest(), OneShot(tokens));
}

TEST(HasherTest, ChunkedUpdatesEqualSingleUpdate) {
  const std::vector<TokenId> tokens(37, 42);
  Hasher whole;
  whole.UpdateTokens(tokens);
  Hasher chunked;
  chunked.UpdateTokens(std::vector<TokenId>(tokens.begin(), tokens.begin() + 11));
  chunked.UpdateTokens(std::vector<TokenId>(tokens.begin() + 11, tokens.end()));
  EXPECT_EQ(whole.Digest(), chunked.Digest());
}

TEST(HasherTest, ResetRestartsDigest) {
  Hasher hasher;
  hasher.UpdateTokens(std::vector<TokenId>{1, 2, 3});
  hasher.Reset();
  EXPECT_EQ(hasher.Digest(), 0xEF46DB3751D8E999ULL);
}

TEST(GenBlockHashesTest, ChainedPrefixContract) {
  // hash(i) must equal a one-shot XXH64 over tokens [0, (i+1)*tpb) —
  // the chain is never reset between blocks (FlexKV gen_hashes).
  const std::uint32_t tpb = 2;
  const std::vector<TokenId> tokens = {1, 2, 3, 4, 5, 6, 7, 8};
  const std::vector<BlockHash> hashes = GenBlockHashes(tokens, tpb);

  EXPECT_EQ(hashes.size(), static_cast<std::size_t>(4));
  for (std::size_t i = 0; i < hashes.size(); ++i) {
    const std::vector<TokenId> prefix(tokens.begin(),
                                      tokens.begin() + (i + 1) * tpb);
    EXPECT_EQ(hashes[i], OneShot(prefix));
  }
}

TEST(GenBlockHashesTest, TrailingPartialBlockIgnored) {
  const std::uint32_t tpb = 4;
  const std::vector<TokenId> tokens = {1, 2, 3, 4, 5, 6, 7};  // 1 full + 3.
  const std::vector<BlockHash> hashes = GenBlockHashes(tokens, tpb);
  EXPECT_EQ(hashes.size(), static_cast<std::size_t>(1));
  const std::vector<TokenId> full(tokens.begin(), tokens.begin() + 4);
  EXPECT_EQ(hashes[0], OneShot(full));
}

TEST(GenBlockHashesTest, DeterministicAcrossCalls) {
  const std::vector<TokenId> tokens = {9, 8, 7, 6};
  const std::vector<BlockHash> first = GenBlockHashes(tokens, 2);
  const std::vector<BlockHash> second = GenBlockHashes(tokens, 2);
  EXPECT_EQ(first, second);
}

TEST(GenBlockHashesTest, RejectsZeroTokensPerBlock) {
  const std::vector<TokenId> tokens = {1, 2};
  EXPECT_THROW(GenBlockHashes(tokens, 0), std::invalid_argument);
}

}  // namespace
