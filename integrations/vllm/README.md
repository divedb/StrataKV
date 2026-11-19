# vLLM integration (Phase F)

`v0_17/stratakv_connector.py` — the thin, version-isolated connector
for mainline-era vLLM (>= 0.17.2 connector API; roadmap F1, design.md
Phase F). 0.10.1.1 support is deliberately absent until E0 parity is
actually required.

## What is implemented (2026-11-20)

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
  (returns empty vLLM metadata),
  `update_connector_output` and worker `get_finished` (native polling
  and launch, including empty async-load steps), `request_finished` (stage+submit put, sidecar states
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

## Real vLLM validation

Validated on vLLM 0.34.0, torch 2.14.0, RTX 3080 Ti with
Qwen2.5-0.5B-Instruct. The connector derives geometry and layer names
from `kv_cache_config` (64-token blocks on this backend). Two rounds
produce identical completions to the baseline with vLLM prefix caching
disabled; puts, gets, and StrataKV match hits all advance.

The round-2 hang required polling in worker `get_finished`, including
empty steps. Finished request IDs announce generation completion and
must still be reported when their transfers finish. Offloads select
complete new blocks after the matched prefix, excluding partial tails.
Native transfer failures raise `ConnectorError` with the native error;
automatic recomputation after transfer failure is not implemented.

Run on the GPU host:

```bash
STRATAKV_TEST_MODEL=/path/to/qwen2.5-0.5b-instruct \
PYTHONPATH=python:integrations/vllm/v0_17:integrations/vllm/tests \
VLLM_ENABLE_V1_MULTIPROCESSING=0 timeout 180 python3 -m pytest \
  integrations/vllm/tests -q
```

Remaining: F4 real hybrid-model validation and cross-process
scheduler/worker registration (ADR 0005/0006). Hybrid support is tested
at the adapter level; real vLLM HMA support remains unvalidated.

Status: **F1 done, F2 done, F3 done (adapter level), F4 pending**.
