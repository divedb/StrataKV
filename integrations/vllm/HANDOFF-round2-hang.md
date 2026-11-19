# HANDOFF: real-vLLM integration test — round-2 generate hangs

Written 2026-11-20 for the next session. Everything you need is in
this file; read it fully before touching code. **Local repo is the
single source of truth** — never edit code on the remote; edit locally,
`~/.ssh-helper/rrsync.sh`, rebuild/rerun remotely.

## 1. What this is

Phase F (roadmap): the StrataKV↔vLLM connector
(`integrations/vllm/v0_17/stratakv_connector.py`) is being validated
against REAL vLLM (0.34.0) on the remote GPU host. The fake-harness
suite (`tests/test_connector.py`, 8 tests) is green locally and
remotely, including real-torch byte parity. The remaining work is ONE
test: `tests/test_vllm_integration.py::
test_offline_serving_parity_with_connector`. It currently **hangs**
(no crash, no timeout) in its second `llm.generate()` round.

Everything up to the hang WORKS: vLLM boots with the connector via
`KVTransferConfig`, the connector subclasses the real
`KVConnectorBase_V1`, derives its layout from vLLM's own
`kv_cache_config`, registers real per-layer CUDA tensors, round-1
generation completes, requests finish, puts are submitted and
committed to the CPU tier, and (in the earlier single-round version of
the test) **outputs were byte-identical with vs without the connector**.
Only prefix-REUSE across a second round hangs.

## 2. Environment (remote E0-class host)

- SSH: `~/.ssh-helper/rssh.sh '<command>'` (SSH_ASKPASS wrapper;
  password auth; port 10304, root@connect.westb.seetacloud.com).
  Sync: `~/.ssh-helper/rrsync.sh` (incremental rsync →
  `/root/autodl-tmp/StrataKV`, excludes builds/.git/*.so).
- RTX 3080 Ti 12 GB (sm_86), driver 580.76.05, CUDA 12.8.
- venv: `/root/autodl-tmp/venv-vllm` — **vLLM 0.34.0, torch 2.14.0,
  pytest** installed. Model:
  `/root/autodl-tmp/models/qwen2.5-0.5b-instruct` (already downloaded
  via **modelscope** — use modelscope for models; HF is slow unless
  `source /etc/network_turbo`; pip is fastest via the aliyun mirror
  WITHOUT turbo).
- Native module: built by the remote tree
  `/root/autodl-tmp/StrataKV/build-py` (STRATAKV_PYTHON=ON
  STRATAKV_GPU=ON, sm_86). It is already built and current.

Run command (use `timeout` — the hang gives no failure otherwise):

```bash
~/.ssh-helper/rrsync.sh   # after every local edit
~/.ssh-helper/rssh.sh 'cd /root/autodl-tmp/StrataKV && \
  STRATAKV_TEST_MODEL=/root/autodl-tmp/models/qwen2.5-0.5b-instruct \
  PYTHONPATH=python:integrations/vllm/v0_17:integrations/vllm/tests \
  VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_LOGGING_LEVEL=INFO \
  timeout 240 /root/autodl-tmp/venv-vllm/bin/python3 -m pytest \
  integrations/vllm/tests/test_vllm_integration.py -q -x --tb=short'
```

Each run costs ~40–80 s (two LLM boots). Debug env vars already wired
in the connector: `STRATAKV_DEBUG_REG` (dumps kv_caches shapes +
kv_cache_config), `STRATAKV_DEBUG_PUT` (dumps every request_finished +
submit). Use them; add more if needed.

## 3. vLLM 0.34 facts already established (do not re-derive)

- Connector base: `vllm.distributed.kv_transfer.kv_connector.v1.base.
  KVConnectorBase_V1`. Ctor: `(vllm_config, role: KVConnectorRole,
  kv_cache_config)`. vLLM constructs TWO instances (SCHEDULER + WORKER
  roles) from the same config → the connector's shared-engine
  singleton must ATTACH on the second construction (already fixed).
- Config: `KVTransferConfig(kv_connector="StrataKVConnectorV1",
  kv_connector_module_path="stratakv_connector", kv_role="kv_both")`.
  Extra config field is `kv_connector_extra_config` (NOT
  `kv_extra_config`).
- Abstract hooks we implement: `build_connector_meta`,
  `get_num_new_matched_tokens`, `save_kv_layer`, `start_load_kv`,
  `update_state_after_alloc`, `wait_for_layer_load`, `wait_for_save`.
  `build_connector_meta` must return a real `KVConnectorMetadata()`
  (empty is fine).
- `update_state_after_alloc(request, blocks: KVCacheBlocks,
  num_external_tokens)` is called for EVERY scheduled request —
  including `num_external_tokens == 0` (must no-op, already fixed).
  `KVCacheBlocks.blocks` = tuple per kv-group of block objects
  (`.block_id`).
- `request_finished(request, block_ids)` — `block_ids` are only the
  blocks STILL HELD (may be fewer than the request's total); stage
  only `tokens[: len(all_ids) * tpb]` and use the NEWEST
  `plan.new_blocks` ids (already fixed).
- `get_finished(...) -> tuple[set|None, set|None]` — ALWAYS a tuple,
  None for empty sides (0.34 shape; already fixed).
- Geometry comes from `kv_cache_config`: groups → specs
  (`FullAttentionSpec.block_size` etc.), `num_blocks`, exact
  `layer_names`. **On this setup vLLM chose block_size=64 with the
  LBNHC layout**: per-layer tensor shape [num_blocks=5120, kv_heads=2,
  64, 128] bf16 → per-layer per-block 32768 B, model-wide 786432 B
  (24 layers). NO padding mystery — it was my test's explicit
  `tokens_per_block=16` override that mismatched; the fix was to pass
  NO `kv_connector_extra_config` and let `kv_cache_config` drive
  (already fixed).
- `VLLM_ENABLE_V1_MULTIPROCESSING=0` is set — but NOTE: hung-process
  evidence (`pytest` main in `do_wait` waiting on a child) suggests
  0.34 may STILL spawn an engine-core child. VERIFY whether the
  single-process env var still exists/works in 0.34 (it may have been
  renamed, e.g. V1 multiplayer split always-on). If the engine core is
  a separate PROCESS, the shared-engine singleton assumption breaks
  (scheduler process and worker process each get their own connector
  objects and CANNOT share memory) — see hypothesis D below.
- Other base hooks that may matter (defaults currently inherited):
  `may_load_async`, `get_finished_count`, `has_pending_push_work`,
  `get_block_ids_with_load_errors`, `handle_preemptions`,
  `bind_connector_metadata`/`_get_connector_metadata` (the metadata
  bind/clear pair around each step).

## 4. The failure, precisely

Test flow (see `tests/test_vllm_integration.py`): baseline LLM (no
connector) → outputs; connector LLM → round 1 (two long prompts
sharing a ~200-token preamble; block_size=64 so multi-block) →
completes; `sleep(1.0)` for puts to commit; round 2 (exact replay of
round-1 prompt 0) → **`llm.generate()` never returns**. Earlier
single-round version of this test (all prompts in one batch)
COMPLETED with byte-identical outputs but had zero match hits
(requests scheduled before the first put committed — hence the split
into rounds). Remote evidence: pytest main PID in `do_wait`; no
assertion, no exception, GPU idle.

## 5. Ranked hypotheses (investigate in this order)

- **A. Round-2 GET fails and the failure never surfaces (MOST
  LIKELY; real connector bug).** `update_connector_output` only moves
  `"succeeded"` gets into `finished_recving`. If the round-2
  submit_get → transport op fails at flush (validation, geometry) the
  task goes `kFailed`, vLLM waits for `finished_recving` forever, and
  we never look at `item.error`. FIX: in the poll loop, treat
  `"failed"` gets as terminal too — report them via `finished_recving`
  (vLLM has `kv_load_failure_policy` to recompute) and LOG
  `item.error`. Also add the error string to the debug output. Check
  `KVTransferConfig.kv_load_failure_policy` values in 0.34.
- **B. The async-load handshake needs hooks we left at defaults.**
  `may_load_async()` (we return `True` in get_num's second tuple
  element — but ALSO check the base default of `may_load_async`; if it
  must agree), and especially `get_finished_count()` — if vLLM uses it
  to decide whether to call `get_finished`, a default `None` may mean
  "never poll". Inspect the vLLM scheduler source
  (`vllm/v1/core/sched/scheduler.py`, around where `get_finished` /
  `get_finished_count` are called) to see the exact contract.
- **C. Round-2 get is never SUBMITTED.** `get_num_new_matched_tokens`
  must return `(matched_tokens>0, True)` for the replay; if vLLM
  caches the negative answer from step 0 or calls it with
  `num_computed_tokens` semantics we mishandle, the request may wait
  without any submit. Debug-print the return value for round-2
  requests.
- **D. Process topology assumption broken.** If 0.34 spawns an engine
  core child regardless of the env var, the WORKER-role connector
  lives in a different process from the SCHEDULER-role one; the
  scheduler process's engine then has no GPU groups registered (they
  were registered in the worker process) — round-1 puts may have run
  on the scheduler side only because `from_vllm` auto-constructed an
  engine there; round-2 gets then fail/hang differently. VERIFY:
  print `os.getpid()` in both role constructors. If PIDs differ → the
  v1 single-process contract is unmet; either find the correct 0.34
  single-process switch or use the cross-process path (the engine's
  `IpcTransportClient` isolated mode exists for exactly this — but
  that is ADR 0005/0006 future work; timeboxed, prefer A/B/C first).

## 6. Suggested first debugging session (30 min)

1. Add to `update_connector_output` (locally, then rsync): print every
   non-`launched` try_wait item (id, state, error); treat `failed`
   gets/puts as terminal + reported.
2. Add `os.getpid()` + role prints to the constructor (hypothesis D).
3. Rerun with `timeout 240` and `VLLM_LOGGING_LEVEL=INFO`. The vLLM
   scheduler logs external-KV waits; our prints show the task states.
4. If a get FAILED: its `item.error` names the transport validation —
   follow it (likely geometry/ids in round-2's `update_state_after_
   alloc`; print `match.blocks` and the resolved `gpu_slots`).
5. If nothing was submitted: print `get_num_new_matched_tokens`
   returns for round-2 (hypothesis C) and read the 0.34 scheduler
   source for the waiting-request path (hypothesis B).

## 7. Acceptance criteria for this case

- `test_offline_serving_parity_with_connector` passes on the remote:
  identical outputs vs baseline (both rounds), `connector_puts > 0`,
  `connector_match_hits > 0` (the replay served from StrataKV, not
  vLLM's prefix cache — it is disabled in the test), and the test
  finishes under the timeout.
- Fake suite (`test_connector.py`, 8 tests) still green in the same
  venv AND locally on WSL2 (no regressions from your changes).
- Commit with the story; update `integrations/vllm/README.md`
  ("blocked on" section) and `roadmap.md` Phase F status.

## 8. Repo state at handoff

- All Phase-F connector code committed through `57a6184` PLUS the
  uncommitted-but-rsynced vLLM-0.34 adaptations described above
  (subclassing, kv_cache_config layout derivation, capacities, the 8
  fixes). The first action of the next session should be committing
  the current working tree as WIP so diffs stay clean.
- Full test matrix green before this test: local 4 trees + remote
  4/4 ctest (unit/GPU/isolated/fault-injection/python/connector).
- The one-off WSL2 segfault noted in commit `198437b` is unrelated to
  this hang (remote run, native Linux).
