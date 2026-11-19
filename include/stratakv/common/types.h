#pragma once

#include <cstdint>
#include <string>
#include <variant>
#include <vector>

namespace stratakv::common {

/// \brief Token identifier as consumed by inference engines (token id).
using TokenId = std::int64_t;

/// \brief Identifier of one block slot inside a tier's pool.
using BlockId = std::uint32_t;

/// \brief Stable identifier of a storage tier instance.
using TierId = std::uint32_t;

/// \brief Kind of memory a BufferView points into.
enum class DeviceKind : std::uint8_t {
  kGpu,        ///< Device memory in a CUDA context.
  kCpuPinned,  ///< Page-locked host memory (cudaHostRegister'ed).
  kCpu,        ///< Ordinary pageable host memory.
  kSsd,        ///< Backed by a storage file.
};

/// \brief Human-readable name for a DeviceKind.
std::string ToString(DeviceKind kind);

/// \brief Non-owning view of one contiguous memory region.
///
/// Ownership: the caller retains ownership of the memory; StrataKV never
/// frees a BufferView. The view must outlive every operation that uses it
/// (for transfers: until completion is delivered). Thread safety:
/// immutable after construction; safe to copy and share across threads.
struct BufferView {
  void* base = nullptr;       ///< Raw pointer; never owned by StrataKV.
  DeviceKind kind = DeviceKind::kCpu;
  int device_id = -1;         ///< CUDA device (kGpu) or NUMA node hint.
  std::uint64_t size_bytes = 0;
  /// \brief Row-major byte strides, outermost first; empty for opaque blobs.
  std::vector<std::uint64_t> strides_bytes;

  /// \brief Structural validity: non-null base, non-zero size, strides
  /// monotonically non-increasing and consistent with size_bytes.
  bool IsValid() const;
};

// ---------------------------------------------------------------------------
// Layer specification (heterogeneous from day one; ADR 0003).
// ---------------------------------------------------------------------------

/// \brief Full-attention KV geometry (paged token blocks).
struct FullAttnSpec {
  int num_kv_heads = 0;  ///< KV heads after GQA folding (>= 1).
  int head_size = 0;     ///< Per-head dimension in elements (>= 1).
};

/// \brief Multi-head latent attention (absorbed latent, single tensor).
struct MlaSpec {
  int kv_lora_rank = 0;  ///< Latent rank (>= 1).
  int rope_dim = 0;      ///< RoPE dimension folded into the latent (>= 0).
};

/// \brief Sliding-window attention KV (window-bounded reuse).
struct SlidingWindowSpec {
  int num_kv_heads = 0;
  int head_size = 0;
  int window_blocks = 0;  ///< Reuse horizon in blocks (>= 1).
};

/// \brief Linear-attention / SSM recurrent state (not token-addressable).
struct LinearStateSpec {
  std::uint64_t state_bytes = 0;  ///< Fixed-size state tensor per layer.
};

/// \brief Variant over all supported per-group KV geometries.
using LayerSpec =
    std::variant<FullAttnSpec, MlaSpec, SlidingWindowSpec, LinearStateSpec>;

/// \brief Stable human-readable kind name: "full_attn" | "mla" |
/// "sliding_window" | "linear_state".
std::string SpecName(const LayerSpec& spec);

/// \brief True iff the spec stores per-token KV addressable by token block
/// (full attention, MLA, sliding window). LinearState is prefix-addressed
/// as a whole state object, not token-addressable.
bool IsTokenAddressable(const LayerSpec& spec);

/// \brief Bytes occupied by one token block of this spec.
///
/// FullAttn/SlidingWindow: 2 (K and V) * tokens_per_block * num_kv_heads *
/// head_size * dtype_bytes. Mla: tokens_per_block * (kv_lora_rank +
/// rope_dim) * dtype_bytes. LinearState: 0 (use StateBytes).
///
/// \param spec Layer geometry.
/// \param tokens_per_block Tokens per block (>= 1 for token-addressable).
/// \param dtype_bytes Bytes per element (>= 1).
/// \return Size in bytes; 0 for LinearStateSpec.
std::uint64_t BlockBytes(const LayerSpec& spec, int tokens_per_block,
                         int dtype_bytes);

/// \brief Bytes of one LinearState sidecar object (state_bytes), else 0.
std::uint64_t StateBytes(const LayerSpec& spec);

/// \brief Structural validation: positive heads/dims, non-zero state size,
/// positive window, and similar per-alternative invariants.
bool IsValid(const LayerSpec& spec);

}  // namespace stratakv::common
