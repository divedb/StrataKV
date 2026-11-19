#include "stratakv/common/handles.h"

#include <vector>

namespace stratakv::common {

bool IsTerminal(TaskState state) {
  return state == TaskState::kSucceeded || state == TaskState::kFailed ||
         state == TaskState::kCancelled;
}

bool CacheEntry::IsCompleteFor(const ModelLayout& layout) const {
  if (!is_ready) return false;
  const std::size_t num_groups = layout.num_groups();
  std::vector<bool> has_block(num_groups, false);
  std::vector<bool> has_state(num_groups, false);

  for (const BlockRef& ref : blocks) {
    if (ref.group >= num_groups) return false;
    has_block[ref.group] = true;
  }
  for (const StateRef& ref : states) {
    if (ref.group >= num_groups) return false;
    has_state[ref.group] = true;
  }

  for (std::size_t g = 0; g < num_groups; ++g) {
    const LayerGroup& group = layout.Group(g);
    if (IsTokenAddressable(group.spec)) {
      if (!has_block[g]) return false;  // Missing KV for a token-addressed
                                        // group: partial caching (ADR 0003).
    } else {
      if (!has_state[g]) return false;  // Missing sidecar state.
    }
  }
  return true;
}

}  // namespace stratakv::common
