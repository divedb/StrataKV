#include "stratakv/common/types.h"

#include <algorithm>

namespace stratakv::common {

std::string ToString(DeviceKind kind) {
  switch (kind) {
    case DeviceKind::kGpu:
      return "gpu";
    case DeviceKind::kCpuPinned:
      return "cpu_pinned";
    case DeviceKind::kCpu:
      return "cpu";
    case DeviceKind::kSsd:
      return "ssd";
  }
  return "unknown";
}

bool BufferView::IsValid() const {
  if (base == nullptr || size_bytes == 0) return false;
  if (device_id < -1) return false;
  // Strides, when present, must be row-major (non-increasing) and the
  // outermost stride must not exceed the region size for a 1-element walk.
  for (std::size_t i = 1; i < strides_bytes.size(); ++i) {
    if (strides_bytes[i - 1] < strides_bytes[i]) return false;
  }
  return strides_bytes.empty() || strides_bytes.front() <= size_bytes;
}

std::string SpecName(const LayerSpec& spec) {
  return std::visit(
      [](const auto& s) -> std::string {
        using T = std::decay_t<decltype(s)>;
        if constexpr (std::is_same_v<T, FullAttnSpec>) {
          return "full_attn";
        } else if constexpr (std::is_same_v<T, MlaSpec>) {
          return "mla";
        } else if constexpr (std::is_same_v<T, SlidingWindowSpec>) {
          return "sliding_window";
        } else {
          return "linear_state";
        }
      },
      spec);
}

bool IsTokenAddressable(const LayerSpec& spec) {
  return !std::holds_alternative<LinearStateSpec>(spec);
}

std::uint64_t BlockBytes(const LayerSpec& spec, int tokens_per_block,
                         int dtype_bytes) {
  if (dtype_bytes <= 0) return 0;
  return std::visit(
      [tokens_per_block, dtype_bytes](const auto& s) -> std::uint64_t {
        using T = std::decay_t<decltype(s)>;
        if constexpr (std::is_same_v<T, LinearStateSpec>) {
          return 0;  // State objects are not block-granular.
        } else {
          if (tokens_per_block <= 0) return 0;
          if constexpr (std::is_same_v<T, FullAttnSpec> ||
                        std::is_same_v<T, SlidingWindowSpec>) {
            return static_cast<std::uint64_t>(tokens_per_block) * 2u *
                   static_cast<std::uint32_t>(s.num_kv_heads) *
                   static_cast<std::uint32_t>(s.head_size) *
                   static_cast<std::uint32_t>(dtype_bytes);
          } else {  // MlaSpec: one latent (rank + rope) per token.
            return static_cast<std::uint64_t>(tokens_per_block) *
                   static_cast<std::uint32_t>(s.kv_lora_rank + s.rope_dim) *
                   static_cast<std::uint32_t>(dtype_bytes);
          }
        }
      },
      spec);
}

std::uint64_t StateBytes(const LayerSpec& spec) {
  if (const auto* linear = std::get_if<LinearStateSpec>(&spec)) {
    return linear->state_bytes;
  }
  return 0;
}

bool IsValid(const LayerSpec& spec) {
  return std::visit(
      [](const auto& s) -> bool {
        using T = std::decay_t<decltype(s)>;
        if constexpr (std::is_same_v<T, FullAttnSpec>) {
          return s.num_kv_heads > 0 && s.head_size > 0;
        } else if constexpr (std::is_same_v<T, MlaSpec>) {
          return s.kv_lora_rank > 0 && s.rope_dim >= 0;
        } else if constexpr (std::is_same_v<T, SlidingWindowSpec>) {
          return s.num_kv_heads > 0 && s.head_size > 0 &&
                 s.window_blocks > 0;
        } else {
          return s.state_bytes > 0;
        }
      },
      spec);
}

}  // namespace stratakv::common
