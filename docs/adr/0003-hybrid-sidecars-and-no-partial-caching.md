# ADR 0003 — Hybrid attention support and the no-partial-caching invariant

- Status: Accepted
- Date: 2026-11-19

## Context

Linear-attention / SSM layers (Qwen3-Next-class hybrids) carry a
fixed-size recurrent **state**, not token-addressable KV; sliding-window
layers only reuse a bounded horizon. FlexKV's single-geometry model
(`num_layers × num_kv_heads × head_size`) cannot express these, and
skipping such layers while caching the rest produces **silently wrong
attention** — the skipped state depends on the entire prefix and cannot be
reconstructed from tokens plus partial KV.

## Decision

1. The model description is a per-layer-group spec table
   (`ModelLayout` of `FullAttn | Mla | SlidingWindow | LinearState`
   groups). No global single-tuple coupling anywhere in the core.
2. A cache entry is a **bundle**: token blocks for token-addressable
   groups **plus** state sidecar snapshots for `LinearState` groups, taken
   at the prefix boundary. States are prefix-addressable even though they
   are not token-addressable.
3. **No-partial-caching invariant**: a prefix match counts only when the
   entire bundle at that boundary is loadable. Enforced structurally by
   `CacheEntry::IsCompleteFor(const ModelLayout&)` (Phase B) and by
   readiness filtering in the metadata plane's match path (Phase C).
4. Sliding-window v1 policy: store SWA-layer KV fully (correct, wasteful)
   and move to window-bounded storage only after correctness parity is
   proven — the same progression upstream FlexKV took with "FullKV + SWA
   dual caches".

## Consequences

- Radix matching stays token-hash based and model-agnostic; only node
  payloads and pool types generalize (block pools + state pools).
- Hybrid support becomes an object-model property, not a per-model
  rewrite (the FlexKV v1.0.0 failure mode this design exists to avoid).
