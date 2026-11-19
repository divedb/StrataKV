#include "stratakv/metadata/index.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <deque>
#include <list>
#include <map>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#include "stratakv/common/exceptions.h"

namespace stratakv::metadata {

namespace {

std::uint64_t DefaultNow() {
  const auto now = std::chrono::steady_clock::now();
  return static_cast<std::uint64_t>(
      now.time_since_epoch().count());
}

}  // namespace

/// \brief Node payload for one block position: one BlockRef per
/// token-addressable group, in ModelLayout group order.
struct RadixSlot {
  std::vector<common::BlockRef> refs;
};

/// \brief Radix node (port of FlexKV CRadixNode; hashes generalize
/// physical blocks to per-group ref bundles; boundary states live at the
/// node END).
struct RadixNode {
  std::deque<BlockHash> hashes;
  std::deque<RadixSlot> slots;
  std::map<BlockHash, RadixNode*> children;
  RadixNode* parent = nullptr;
  bool ready = false;
  bool in_leaf_list = false;
  std::uint64_t last_access_ns = 0;
  std::vector<common::StateRef> states;        ///< Boundary sidecars.
  std::unique_ptr<common::CacheEntry> entry;   ///< Boundary entry handle.

  std::size_t size() const { return hashes.size(); }
  BlockHash head_hash() const {
    return hashes.empty() ? BlockHash{0} : hashes.front();
  }
  bool is_leaf() const { return children.empty(); }
  bool has_boundary() const { return entry != nullptr; }
  bool any_pinned() const {
    for (const RadixSlot& slot : slots) {
      for (const common::BlockRef& ref : slot.refs) {
        if (ref.refcount > 0) return true;
      }
    }
    for (const common::StateRef& ref : states) {
      if (ref.refcount > 0) return true;
    }
    return false;
  }
};

/// \brief One matched segment of the walk: `len` leading blocks of
/// `node` matched the query.
struct WalkSegment {
  RadixNode* node;
  std::uint32_t len;
};

/// \brief Internal match walk state (FlexKV match_prefix port).
struct WalkResult {
  std::uint32_t matched = 0;   ///< Raw hash-matched blocks.
  std::uint32_t ready = 0;     ///< Ready-prefix blocks (FlexKV
                               ///< num_ready_matched_blocks).
  std::vector<WalkSegment> segs;  ///< Matched segments, prefix order.
  RadixNode* final_node = nullptr;
  std::uint32_t final_node_matched = 0;
  /// \brief Hybrid (ADR 0003): cumulative blocks at the latest node-end
  /// boundary carrying a complete, ready state bundle.
  std::uint32_t bundle_boundary = 0;
  RadixNode* bundle_node = nullptr;
};

struct RadixIndex::Impl {
  const common::ModelLayout& layout;
  std::vector<std::uint32_t> token_groups;  ///< Token-addressable groups.
  std::vector<std::uint32_t> state_groups;  ///< LinearState groups.
  bool hybrid = false;
  std::uint32_t tokens_per_block = 0;
  TimeSource now;
  std::uint64_t grace_ns = 0;

  std::unique_ptr<RadixNode> root;
  std::unordered_map<RadixNode*, std::unique_ptr<RadixNode>> nodes;
  std::list<RadixNode*> leaf_list;
  std::unordered_map<const common::CacheEntry*, RadixNode*> entry_nodes;
  mutable std::mutex mu;

  explicit Impl(const common::ModelLayout& lay, TimeSource clock,
                std::chrono::milliseconds grace)
      : layout(lay), now(clock ? std::move(clock) : TimeSource(&DefaultNow)) {
    for (std::size_t g = 0; g < lay.num_groups(); ++g) {
      if (IsTokenAddressable(lay.Group(static_cast<int>(g)).spec)) {
        token_groups.push_back(static_cast<std::uint32_t>(g));
      } else {
        state_groups.push_back(static_cast<std::uint32_t>(g));
      }
    }
    if (token_groups.empty()) {
      throw std::invalid_argument(
          "RadixIndex: layout has no token-addressable group");
    }
    const std::uint32_t tpb =
        static_cast<std::uint32_t>(lay.Group(token_groups.front()).tokens_per_block);
    for (std::uint32_t g : token_groups) {
      if (static_cast<std::uint32_t>(lay.Group(g).tokens_per_block) != tpb) {
        throw std::invalid_argument(
            "RadixIndex: token-addressable groups must share one "
            "tokens_per_block");
      }
    }
    hybrid = !state_groups.empty();
    tokens_per_block = tpb;
    grace_ns = static_cast<std::uint64_t>(grace.count()) * 1000000ULL;
    ResetLocked();
  }

  void ResetLocked() {
    leaf_list.clear();
    nodes.clear();
    entry_nodes.clear();
    root = std::make_unique<RadixNode>();
    root->ready = true;
    root->last_access_ns = now();
  }

  void AddLeaf(RadixNode* node) {
    if (node->in_leaf_list) return;
    node->in_leaf_list = true;
    leaf_list.push_back(node);
  }

  void RemoveLeaf(RadixNode* node) {
    if (!node->in_leaf_list) return;
    leaf_list.remove(node);
    node->in_leaf_list = false;
  }

  bool Evictable(const RadixNode* node) const {
    if (node == root.get() || !node->is_leaf() || !node->ready) {
      return false;
    }
    if (node->any_pinned()) return false;
    return (now() - node->last_access_ns) >= grace_ns;
  }

  /// \brief Split `node` after its first prefix_length blocks (FlexKV
  /// CRadixNode::split). Boundary states stay with the deeper node.
  RadixNode* Split(RadixNode* node, std::uint32_t prefix_length) {
    auto fresh = std::make_unique<RadixNode>();
    RadixNode* split_parent = fresh.get();
    split_parent->ready = node->ready;
    split_parent->last_access_ns = node->last_access_ns;
    split_parent->parent = node->parent;
    nodes.emplace(split_parent, std::move(fresh));

    for (std::uint32_t i = 0; i < prefix_length; ++i) {
      split_parent->hashes.push_back(node->hashes.front());
      split_parent->slots.push_back(node->slots.front());
      node->hashes.pop_front();
      node->slots.pop_front();
    }
    // Replace `node` under its parent (same head hash), then re-parent
    // `node` under the new split parent (its head hash changed).
    node->parent->children[split_parent->head_hash()] = split_parent;
    split_parent->children[node->head_hash()] = node;
    node->parent = split_parent;
    return split_parent;
  }

  /// \brief FlexKV match_prefix port over precomputed block hashes.
  ///
  /// Two deliberate divergences from the FlexKV port (both documented
  /// in the RadixIndex class comment): the child-hash lookup is skipped
  /// when the node would consume the rest of the query (FlexKV reads
  /// one element past the tensor there), and ready accounting is a
  /// contiguous prefix (FlexKV sums ready segments even across unready
  /// gaps).
  WalkResult Walk(const std::vector<BlockHash>& query, bool update_time) {
    WalkResult w;
    RadixNode* cur = root.get();
    bool ready_blocked = false;
    while (w.matched < query.size()) {
      if (update_time) {
        cur->last_access_ns = now();
      }
      const std::uint32_t cur_size =
          static_cast<std::uint32_t>(cur->size());
      const bool child_may_exist =
          w.matched + cur_size < static_cast<std::uint32_t>(query.size());
      auto child = child_may_exist
          ? cur->children.find(
                query[w.matched + cur_size])
          : cur->children.end();
      if (child != cur->children.end()) {
        if (cur->ready) {
          w.ready += cur_size;
        } else {
          ready_blocked = true;  // Never serve through unready data.
        }
        w.matched += cur_size;
        if (!cur->hashes.empty()) {
          w.segs.push_back(WalkSegment{cur, cur_size});
          RecordBundleBoundary(cur, &w);
        }
        cur = child->second;
      } else {
        std::uint32_t matched_length = 0;
        if (cur != root.get()) {
          const std::uint32_t cmp = std::min(
              cur_size,
              static_cast<std::uint32_t>(query.size()) - w.matched);
          std::uint32_t lo = 0;
          std::uint32_t hi = cmp;
          while (lo < hi) {
            const std::uint32_t mid = (lo + hi) / 2;
            if (cur->hashes[mid] ==
                query[w.matched + mid]) {
              lo = mid + 1;
            } else {
              hi = mid;
            }
          }
          matched_length = lo;
        }
        if (cur->ready && !ready_blocked) {
          w.ready += matched_length;
        }
        w.matched += matched_length;
        if (matched_length > 0) {
          w.segs.push_back(WalkSegment{cur, matched_length});
          if (matched_length == cur->size()) {
            RecordBundleBoundary(cur, &w);
          }
        }
        w.final_node = cur;
        w.final_node_matched = matched_length;
        return w;
      }
    }
    // Query exhausted exactly at a node boundary; FlexKV leaves
    // last_node = the freshly entered child with matched_length 0
    // (insert() never uses it in this case — it returns early).
    w.final_node = cur;
    w.final_node_matched = 0;
    return w;
  }

  void RecordBundleBoundary(RadixNode* node, WalkResult* w) const {
    if (!hybrid || !node->ready || !node->has_boundary()) {
      return;
    }
    w->bundle_boundary = w->matched;
    w->bundle_node = node;
  }

  /// \brief Increment (or decrement) refcounts along the first
  /// matched_blocks of the walk; boundary states at its end (hybrid).
  void ApplyPins(const WalkResult& w, std::uint32_t matched_blocks,
                 bool pin_states, int delta) {
    std::uint32_t consumed = 0;
    for (const WalkSegment& seg : w.segs) {
      if (consumed >= matched_blocks) break;
      const std::uint32_t take =
          std::min(seg.len, matched_blocks - consumed);
      for (std::uint32_t i = 0; i < take; ++i) {
        for (common::BlockRef& ref : seg.node->slots[i].refs) {
          if (delta > 0) {
            ++ref.refcount;
          } else {
            if (ref.refcount == 0) {
              throw LogicError("RadixIndex: refcount underflow on Release");
            }
            --ref.refcount;
          }
        }
      }
      consumed += take;
    }
    if (pin_states && consumed == matched_blocks && matched_blocks > 0) {
      RadixNode* boundary = w.bundle_node;
      if (boundary != nullptr && boundary->has_boundary() &&
          matched_blocks == w.bundle_boundary) {
        for (common::StateRef& ref : boundary->states) {
          if (delta > 0) {
            ++ref.refcount;
          } else {
            if (ref.refcount == 0) {
              throw LogicError("RadixIndex: state refcount underflow");
            }
            --ref.refcount;
          }
        }
      }
    }
  }

  void CollectSegRefs(const WalkResult& w, std::uint32_t matched_blocks,
                      std::vector<common::BlockRef>* out) const {
    std::uint32_t consumed = 0;
    for (const WalkSegment& seg : w.segs) {
      if (consumed >= matched_blocks) break;
      const std::uint32_t take =
          std::min(seg.len, matched_blocks - consumed);
      for (std::uint32_t i = 0; i < take; ++i) {
        for (const common::BlockRef& ref : seg.node->slots[i].refs) {
          out->push_back(ref);
        }
      }
      consumed += take;
    }
  }
};

RadixIndex::RadixIndex(const common::ModelLayout& layout, TimeSource now,
                       std::chrono::milliseconds eviction_grace)
    : impl_(std::make_unique<Impl>(layout, std::move(now), eviction_grace)) {}

RadixIndex::~RadixIndex() = default;

MatchResult RadixIndex::Match(std::span<const common::TokenId> tokens,
                              std::uint32_t tokens_per_block,
                              const MatchOptions& opts) {
  const std::vector<BlockHash> hashes = GenBlockHashes(tokens, tokens_per_block);
  std::lock_guard<std::mutex> lock(impl_->mu);

  WalkResult w = impl_->Walk(hashes, /*update_time=*/true);
  std::uint32_t servable;
  if (opts.require_ready) {
    servable = w.ready;
    if (impl_->hybrid) {
      servable = std::min(servable, w.bundle_boundary);
    }
  } else {
    servable = w.matched;
  }

  MatchResult result;
  result.matched_blocks = servable;
  result.matched_blocks_reserved = w.matched - servable;

  std::shared_ptr<MatchPin> pin;
  if (opts.lock_matched && servable > 0) {
    pin = std::make_shared<MatchPin>();
    pin->hashes = hashes;
    pin->tokens_per_block = tokens_per_block;
    pin->matched_blocks = servable;
    pin->pins_states = impl_->hybrid;
    impl_->ApplyPins(w, servable, pin->pins_states, +1);
    result.pin = pin;
  }

  impl_->CollectSegRefs(w, servable, &result.block_refs);
  if (impl_->hybrid && servable > 0 && w.bundle_node != nullptr &&
      servable == w.bundle_boundary) {
    result.state_refs = w.bundle_node->states;
  }

  result.token_mask.assign(tokens.size(), 0);
  const std::size_t servable_tokens = std::min(
      tokens.size(),
      static_cast<std::size_t>(servable) * tokens_per_block);
  std::fill(result.token_mask.begin(),
            result.token_mask.begin() +
                static_cast<std::ptrdiff_t>(servable_tokens),
            1);
  return result;
}

common::CacheEntry* RadixIndex::Insert(
    std::span<const common::TokenId> tokens,
    std::vector<common::BlockRef> blocks,
    std::vector<common::StateRef> states) {
  const std::vector<BlockHash> hashes =
      GenBlockHashes(tokens, impl_->tokens_per_block);
  std::lock_guard<std::mutex> lock(impl_->mu);

  WalkResult w = impl_->Walk(hashes, /*update_time=*/true);
  if (w.matched >= hashes.size()) {
    return nullptr;  // Fully present already (FlexKV parity).
  }
  const std::uint32_t new_blocks =
      static_cast<std::uint32_t>(hashes.size()) - w.matched;
  const std::uint32_t num_token_groups =
      static_cast<std::uint32_t>(impl_->token_groups.size());
  if (blocks.size() !=
      static_cast<std::size_t>(new_blocks) * num_token_groups) {
    throw std::invalid_argument(
        "RadixIndex::Insert: expected new_blocks * num_token_groups refs");
  }
  const std::size_t want_states =
      impl_->hybrid ? impl_->state_groups.size() : 0;
  if (states.size() != want_states) {
    throw std::invalid_argument(
        "RadixIndex::Insert: state ref count must match LinearState "
        "groups");
  }
  for (std::size_t i = 0; i < blocks.size(); ++i) {
    const std::uint32_t expect_group =
        impl_->token_groups[i % impl_->token_groups.size()];
    if (blocks[i].group != expect_group) {
      throw std::invalid_argument(
          "RadixIndex::Insert: block ref group mismatch at flat position " +
          std::to_string(i));
    }
  }
  for (std::size_t s = 0; s < states.size(); ++s) {
    if (states[s].group != impl_->state_groups[s]) {
      throw std::invalid_argument(
          "RadixIndex::Insert: state ref group mismatch at position " +
          std::to_string(s));
    }
  }

  auto fresh = std::make_unique<RadixNode>();
  RadixNode* node = fresh.get();
  node->ready = false;
  node->last_access_ns = impl_->now();
  for (std::uint32_t i = 0; i < new_blocks; ++i) {
    node->hashes.push_back(hashes[w.matched + i]);
    RadixSlot slot;
    slot.refs.reserve(num_token_groups);
    for (std::uint32_t g = 0; g < num_token_groups; ++g) {
      common::BlockRef ref =
          blocks[static_cast<std::size_t>(i) * num_token_groups + g];
      ref.refcount = 0;
      slot.refs.push_back(ref);
    }
    node->slots.push_back(std::move(slot));
  }
  for (common::StateRef& ref : states) {
    ref.refcount = 0;
  }
  node->states = std::move(states);

  // FlexKV insert(): split the final node when the match ends mid-node,
  // then hang the new leaf under the (possibly new) parent.
  RadixNode* parent = w.final_node;
  if (w.final_node_matched > 0 &&
      w.final_node_matched < parent->size()) {
    impl_->Split(parent, w.final_node_matched);
    parent = parent->parent;
  }
  if (parent->is_leaf()) {
    impl_->RemoveLeaf(parent);
  }
  node->parent = parent;
  parent->children[node->head_hash()] = node;
  impl_->nodes.emplace(node, std::move(fresh));
  impl_->AddLeaf(node);

  auto entry = std::make_unique<common::CacheEntry>();
  entry->blocks.reserve(blocks.size());
  for (std::uint32_t i = 0; i < new_blocks; ++i) {
    for (std::uint32_t g = 0; g < num_token_groups; ++g) {
      common::BlockRef ref =
          blocks[static_cast<std::size_t>(i) * num_token_groups + g];
      ref.refcount = 0;
      entry->blocks.push_back(ref);
    }
  }
  entry->states = node->states;
  entry->is_ready = false;
  node->entry = std::move(entry);
  impl_->entry_nodes[node->entry.get()] = node;
  return node->entry.get();
}

void RadixIndex::CommitReady(common::CacheEntry* entry) {
  std::lock_guard<std::mutex> lock(impl_->mu);
  auto found = impl_->entry_nodes.find(entry);
  if (found == impl_->entry_nodes.end()) {
    throw LogicError("RadixIndex::CommitReady: unknown or evicted entry");
  }
  RadixNode* node = found->second;
  node->ready = true;
  node->entry->is_ready = true;

  // Span-bounded ancestor marking (FlexKV's set_ready ready_length walk,
  // restricted to the blocks THIS put wrote): the entry's staged span is
  // [matched, len) with len = path depth of its node and matched =
  // len - new_blocks. Split-created ancestors inside that span carry
  // this put's blocks and become ready with it; ancestors belonging to
  // earlier puts keep their own readiness (no over-marking).
  const std::uint32_t num_token_groups =
      static_cast<std::uint32_t>(impl_->token_groups.size());
  const std::uint32_t new_blocks = static_cast<std::uint32_t>(
      node->entry->blocks.size() / num_token_groups);
  std::uint32_t len = 0;
  for (const RadixNode* p = node; p != nullptr; p = p->parent) {
    len += static_cast<std::uint32_t>(p->size());
  }
  const std::uint32_t matched = len - new_blocks;

  std::uint32_t child_start = len - static_cast<std::uint32_t>(node->size());
  for (RadixNode* p = node->parent; p != nullptr; p = p->parent) {
    if (child_start < p->size()) break;  // Parent starts before `matched`.
    child_start -= static_cast<std::uint32_t>(p->size());
    if (child_start < matched) break;
    p->ready = true;
  }
}

void RadixIndex::Release(const MatchResult& match) {
  if (match.pin == nullptr) {
    return;
  }
  std::lock_guard<std::mutex> lock(impl_->mu);
  bool expected = false;
  if (!match.pin->applied.compare_exchange_strong(expected, true)) {
    return;  // Idempotent per MatchResult.
  }
  WalkResult w = impl_->Walk(match.pin->hashes, /*update_time=*/false);
  if (w.matched < match.pin->matched_blocks) {
    throw LogicError(
        "RadixIndex::Release: pinned prefix shrank — pin invariant "
        "violated");
  }
  impl_->ApplyPins(w, match.pin->matched_blocks, match.pin->pins_states,
                   -1);
}

Evicted RadixIndex::Evict(std::uint32_t max_blocks) {
  std::lock_guard<std::mutex> lock(impl_->mu);
  Evicted evicted;
  if (max_blocks == 0) {
    return evicted;
  }

  struct OldestFirst {
    bool operator()(const RadixNode* a, const RadixNode* b) const {
      return a->last_access_ns > b->last_access_ns;  // min-heap on time.
    }
  };
  std::priority_queue<RadixNode*, std::vector<RadixNode*>, OldestFirst>
      candidates;
  for (RadixNode* leaf : impl_->leaf_list) {
    if (impl_->Evictable(leaf)) {
      candidates.push(leaf);
    }
  }

  std::uint32_t evicted_blocks = 0;
  while (evicted_blocks < max_blocks && !candidates.empty()) {
    RadixNode* node = candidates.top();
    candidates.pop();
    const std::uint32_t need = max_blocks - evicted_blocks;

    if (node->size() > need && node->states.empty()) {
      // Partial tail shrink (FlexKV shrink): drop the last `need` block
      // positions. Nodes carrying boundary STATES never shrink — the
      // sidecars annotate the node END, and moving it would re-bind the
      // snapshot to a shorter prefix (ADR 0003); they evict whole below.
      const std::uint32_t keep =
          static_cast<std::uint32_t>(node->size()) - need;
      for (std::uint32_t i = keep; i < node->size(); ++i) {
        for (const common::BlockRef& ref : node->slots[i].refs) {
          evicted.blocks.push_back(ref);
        }
      }
      node->hashes.erase(
          node->hashes.begin() + static_cast<std::ptrdiff_t>(keep),
          node->hashes.end());
      node->slots.erase(
          node->slots.begin() + static_cast<std::ptrdiff_t>(keep),
          node->slots.end());
      evicted_blocks += need;
      continue;
    }

    // Whole-node eviction.
    RadixNode* parent = node->parent;
    if (parent != nullptr) {
      parent->children.erase(node->head_hash());
    }
    for (const RadixSlot& slot : node->slots) {
      for (const common::BlockRef& ref : slot.refs) {
        evicted.blocks.push_back(ref);
      }
      ++evicted_blocks;
    }
    for (const common::StateRef& ref : node->states) {
      evicted.states.push_back(ref);
    }
    if (node->entry != nullptr) {
      impl_->entry_nodes.erase(node->entry.get());
      evicted.entries.push_back(node->entry.get());
    }
    if (parent != nullptr && parent->is_leaf() &&
        parent != impl_->root.get()) {
      impl_->AddLeaf(parent);
      if (impl_->Evictable(parent)) {
        candidates.push(parent);
      }
    }
    impl_->RemoveLeaf(node);
    impl_->nodes.erase(node);
  }
  return evicted;
}

void RadixIndex::Reset() {
  std::lock_guard<std::mutex> lock(impl_->mu);
  impl_->ResetLocked();
}

std::uint32_t RadixIndex::TotalCachedBlocks() const {
  std::lock_guard<std::mutex> lock(impl_->mu);
  std::uint32_t total = 0;
  for (const auto& [node, owned] : impl_->nodes) {
    (void)owned;
    total += static_cast<std::uint32_t>(node->size());
  }
  return total;
}

std::uint32_t RadixIndex::TotalReadyBlocks() const {
  std::lock_guard<std::mutex> lock(impl_->mu);
  std::uint32_t total = 0;
  for (const auto& [node, owned] : impl_->nodes) {
    (void)owned;
    if (node->ready) {
      total += static_cast<std::uint32_t>(node->size());
    }
  }
  return total;
}

std::size_t RadixIndex::TotalNodes() const { return impl_->nodes.size(); }

}  // namespace stratakv::metadata
