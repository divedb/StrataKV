"""Phase F: REAL vLLM integration tests (run on the E2-class host).

Requires the venv with vllm + the GPU native module (see
integrations/vllm/README.md). Two legs:

1. API shape: our connector plugs into the installed vLLM's connector
   base without signature drift.
2. Offline serving parity (F2's model-output leg): the same prompts
   through vLLM with and without the StrataKV connector produce
   IDENTICAL outputs, while the connector actually moved KV (its
   put/get counters advance; prefix reuse goes through the native
   match/get path, not vLLM's own prefix cache — we disable that for
   the connector run).

Single-process topology (VLLM_ENABLE_V1_MULTIPROCESSING=0): both
connector roles share the native engine, per the v1 adapter contract.
"""

from __future__ import annotations

import os

# Must precede vllm import.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

import pytest  # noqa: E402

vllm = pytest.importorskip("vllm")
from vllm.distributed.kv_transfer.kv_connector.v1.base import (  # noqa: E402
    KVConnectorBase_V1)

from stratakv_connector import StrataKVConnectorV1  # noqa: E402

MODEL = os.environ.get("STRATAKV_TEST_MODEL",
                       "Qwen/Qwen2.5-0.5B-Instruct")


def test_connector_subclasses_real_base():
    assert issubclass(StrataKVConnectorV1, KVConnectorBase_V1)
    for hook in ("register_kv_caches", "get_num_new_matched_tokens",
                 "update_state_after_alloc", "build_connector_meta",
                 "update_connector_output", "request_finished",
                 "get_finished", "shutdown"):
        assert callable(getattr(StrataKVConnectorV1, hook, None)), hook


def test_offline_serving_parity_with_connector():
    """F2: identical outputs with vs without the connector, with KV
    actually offloaded + reloaded through StrataKV."""
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig

    # vLLM 0.34 picked block_size=64 (LBNHC layout) on this backend —
    # prompts must span multiple blocks with a shared multi-block
    # prefix for offload/reload to engage.
    preamble = ("You are a meticulous assistant. " +
                "Explain your reasoning step by step. " * 24)
    prompts = [
        preamble + "Now, the capital of France is",
        preamble + "Now, write a haiku about databases.",
        preamble + "Now, the capital of France is",  # Full-prefix reuse.
    ]
    params = SamplingParams(temperature=0.0, max_tokens=24,
                            ignore_eos=False)

    # Round 1 populates; round 2 replays prompt 0 — its prefix must
    # come from StrataKV (vLLM's own prefix cache is disabled), which
    # only engages if the earlier put committed (hence the split).
    rounds = [prompts[:2], [prompts[2]]]

    # ---- baseline: no connector, no vLLM prefix cache ----
    llm_plain = LLM(model=MODEL, gpu_memory_utilization=0.45,
                    max_model_len=512, enforce_eager=True,
                    enable_prefix_caching=False)
    baseline = [out.outputs[0].text
                for out in llm_plain.generate(rounds[0] + rounds[1],
                                              params)]
    del llm_plain
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()

    # ---- connector run: StrataKV carries the KV ----
    import stratakv_connector as sc
    sc.reset_shared_engine()
    # No kv_connector_extra_config: the connector derives its layout
    # from vLLM's own kv_cache_config (block_size 64 / LBNHC on this
    # backend) — the authoritative geometry.
    kv_config = KVTransferConfig(
        kv_connector="StrataKVConnectorV1",
        kv_connector_module_path="stratakv_connector",
        kv_role="kv_both",
    )
    llm_kv = LLM(model=MODEL, gpu_memory_utilization=0.45,
                 max_model_len=512, enforce_eager=True,
                 enable_prefix_caching=False,
                 kv_transfer_config=kv_config)
    shared = sc._SHARED
    outs = [out.outputs[0].text
            for out in llm_kv.generate(rounds[0], params)]
    import time as _time
    _time.sleep(1.0)  # Let round-1 puts commit in the engine.
    outs += [out.outputs[0].text
             for out in llm_kv.generate(rounds[1], params)]
    stats = (shared.connector_puts, shared.connector_gets,
             shared.connector_match_hits)
    sc.reset_shared_engine()

    # Parity: identical completions with the connector active.
    assert outs == baseline, (
        f"outputs diverged with connector:\n{outs!r}\nvs\n{baseline!r}")
    # And the connector actually did work (offloads at request finish;
    # the duplicate prompt's prefix is served by the native match).
    assert stats[0] > 0, "no puts reached the engine"
    assert stats[1] > 0, "no gets reached the engine"
    assert stats[2] > 0, "no prefix matches served from StrataKV"
