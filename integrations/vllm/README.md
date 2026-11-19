# vLLM integration (Phase F)

`v0_17/stratakv_connector.py` — the thin, version-isolated connector
for mainline-era vLLM (>= 0.17.2 connector API; roadmap F1, design.md
Phase F). 0.10.1.1 support is deliberately absent until E0 parity is
actually required.

## What is implemented (2026-11-19)

- **Worker role**: `register_kv_caches` parses the per-group GPU
  tensors (raw descriptors — torch stays at this boundary; anything
  with `data_ptr()`/`shape`/`element_size()` works), builds the native
  engine, registers every layout group, starts. All per-forward hooks
  (`start_load_kv`, `wait_for_layer_load`, `save_kv_layer`,
  `wait_for_save`) are no-ops by design — the native data plane moves
  KV without vLLM involvement.
- **Scheduler role**: `get_num_new_matched_tokens` (native
  readiness-filtered match; ADR 0003 no-partial-caching truncation
  surfaces here), `update_state_after_alloc` (submit_get with
  position-major/group-minor slot interleaving), `build_connector_meta`
  (returns `None` — the engine flushes on the next poll),
  `update_connector_output` (the single per-step native poll and
  launch point), `request_finished` (stage+submit put, sidecar states
  via `gpu_state_slots`), `get_finished` (finished_sending/recving).
- **Hybrid (F3)**: multi-group layouts (FullAttn/Mla/SWA/LinearState),
  `HybridSidecarMixin` slot bookkeeping, SWA groups register
  token-addressable (store-all v1; window-trim deferred per roadmap).
- **Topology (v1)**: single-process vLLM
  (`VLLM_ENABLE_V1_MULTIPROCESSING=0`) — both roles share one native
  engine via a process-wide singleton. The cross-process
  registration-endpoint topology (ADR 0005/0006) is future work with
  the real E2 environment.

## Testing without vLLM (and with real torch)

`tests/` holds a fake-vLLM harness (duck-typed tensors over the
facade's debug device memory; scheduler-step driver) plus the suite:
role ordering, cold-put -> warm-get **byte parity** (F2's
block-checksum leg: KV bytes identical across offload+reload, block by
block), hybrid truncation (no-partial-caching at the connector
boundary), SWA store-all, shutdown-with-tasks-in-flight torture, and —
where torch is installed (the E0 remote host) — a round trip with REAL
`torch` CUDA tensors verified via `torch.equal`. CPU-only module
builds run the suite in simulation mode (state-machine coverage; byte
assertions skipped, loudly warned).

ctests as `stratakv_connector_tests` (needs `-DSTRATAKV_PYTHON=ON`).

## Not yet (blocked on the E2 vLLM environment)

- Real vLLM integration test (F1's ctest-in-vLLM leg) — needs vLLM >=
  0.17.2 installed (the E0 host has torch 2.8 + CUDA 12.8 already).
- F2's model-output parity and F4's real hybrid-model validation.
- Cross-process scheduler/worker topology (registration endpoint).

Status: **F1 done (adapter code + fake-harness integration), F2
partial (byte parity done; model-output parity blocked), F3 done
(adapter level), F4 blocked** — see `roadmap.md` Phase F status.
