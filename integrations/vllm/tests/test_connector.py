"""Phase F connector tests (F1 flow + F3 hybrid + F2 byte parity).

GPU builds move REAL bytes end to end through the connector hooks and
verify them (F2's block-checksum parity at the adapter boundary: KV is
byte-identical after offload + reload). CPU builds run the same logic
through the InlineExecutor simulation and skip byte assertions (the
suite is then the connector state-machine check).

Run via ctest (stratakv_connector_tests) with STRATAKV_PYTHON=ON, or:
    PYTHONPATH=python:integrations/vllm/v0_17:integrations/vllm/tests \
    python3 -m pytest integrations/vllm/tests -q
"""

from __future__ import annotations

import os

import pytest

from fake_vllm import (GPU, FakeRequest, FakeSchedulerStep,
                       FakeTensor, pump)
from stratakv import FullAttn, LinearState, SlidingWindow
from stratakv_connector import (ConnectorConfig, ConnectorError,
                                StrataKVConnectorV1, reset_shared_engine)


def teardown_function(_function) -> None:
    reset_shared_engine()


# Qwen2-0.5B-shaped single group: 4 layers x (2 heads x 64) x 16 tpb.
LAYERS = 4
TPB = 16
KV_DIM = 2 * 2 * 64  # 2(K+V) x 2 heads x 64 = 512 elements/layer/token.
BLOCKS = 64

LAYOUT = [FullAttn(name="full_attn", layer_indices=list(range(LAYERS)),
                   num_kv_heads=2, head_size=64, tokens_per_block=TPB)]


def make_connector(layout=None, kv_cache_keys=None,
                   state_objects=64) -> StrataKVConnectorV1:
    layout = layout if layout is not None else LAYOUT
    worker = StrataKVConnectorV1(
        role="worker",
        config=ConnectorConfig(layout=layout,
                               kv_cache_keys=kv_cache_keys or {},
                               blocks_per_group=BLOCKS,
                               state_objects_per_group=state_objects))
    return worker


def make_kv_caches(capacity: int = BLOCKS) -> dict[str, FakeTensor]:
    return {f"full_attn_layer_{i}": FakeTensor((capacity, TPB, KV_DIM))
            for i in range(LAYERS)}


def golden_row(seed: int, size: int) -> bytes:
    out = bytearray()
    state = seed
    while len(out) < size:
        state = (state * 6364136223846793005
                 + 1442695040888963407) & (2**64 - 1)
        out += state.to_bytes(8, "little")
    return bytes(out[:size])


def test_worker_registers_and_starts():
    connector = make_connector()
    caches = make_kv_caches()
    connector.register_kv_caches(caches)
    assert connector._shared.started
    # Layer-name heuristic mapped all four per-layer tensors.
    assert len(connector._shared.config.kv_cache_keys) == 0  # heuristic
    connector.shutdown()
    for tensor in caches.values():
        tensor.free()


def test_register_rejects_missing_group():
    connector = make_connector()
    caches = make_kv_caches()
    del caches["full_attn_layer_3"]
    with pytest.raises(ConnectorError):
        connector.register_kv_caches(caches)
    connector.shutdown()
    for tensor in caches.values():
        tensor.free()


def test_scheduler_role_requires_worker_first():
    with pytest.raises(ConnectorError):
        StrataKVConnectorV1(role="scheduler")


def test_cold_put_then_warm_get_byte_parity():
    """F2 (byte-level): KV bytes survive offload -> reload identically,
    driven entirely through vLLM's hook order."""
    worker = make_connector()
    scheduler = StrataKVConnectorV1(role="scheduler")
    caches = make_kv_caches()
    worker.register_kv_caches(caches)

    tokens = list(range(4 * TPB))  # 4 blocks
    put_blocks = [0, 1, 2, 3]
    get_blocks = [20, 21, 22, 23]
    # Per-layer, per-block bytes: [blocks, tpb, kv_dim] fp16.
    layer_block_bytes = TPB * KV_DIM * 2
    # Deterministic golden KV for blocks 0..3, assembled in the same
    # (layer-major, block-minor) order the verification reads back.
    golden = b"".join(
        golden_row(0xB10C + 1000 * layer + block, layer_block_bytes)
        for layer in range(LAYERS) for block in range(4))
    if GPU:
        for layer in range(LAYERS):
            caches[f"full_attn_layer_{layer}"].write_at(0, b"".join(
                golden_row(0xB10C + 1000 * layer + block,
                           layer_block_bytes)
                for block in range(4)))

    # ---- cold request: runs to completion, offloads blocks 0..3 ----
    cold = FakeRequest("cold", tokens)
    matched, async_load = scheduler.get_num_new_matched_tokens(
        cold, num_computed_tokens=0)
    assert (matched, async_load) == (0, False)
    delays_free, _meta = scheduler.request_finished(cold, put_blocks)
    assert delays_free
    assert pump(scheduler, worker, "cold", sending=True)
    if not GPU:
        scheduler.shutdown()
        for tensor in caches.values():
            tensor.free()
        return  # Simulation: no bytes to verify.

    # ---- warm request: same prefix loads into blocks 20..23 ----
    warm = FakeRequest("warm", tokens)
    matched, async_load = scheduler.get_num_new_matched_tokens(
        warm, num_computed_tokens=0)
    assert matched == 4 * TPB and async_load
    FakeSchedulerStep.run(scheduler, [warm], {"warm": get_blocks})
    assert pump(scheduler, worker, "warm", sending=False)

    # ---- byte parity: blocks 20..23 == the original golden ----
    reloaded = b""
    for layer in range(LAYERS):
        tensor = caches[f"full_attn_layer_{layer}"]
        for block in range(4):
            at = get_blocks[block] * layer_block_bytes
            reloaded += tensor.read_at(at, layer_block_bytes)
    assert reloaded == golden, "KV bytes changed across offload+reload"
    scheduler.shutdown()
    for tensor in caches.values():
        tensor.free()


def test_hybrid_truncation_enforces_no_partial_caching():
    """F3/ADR 0003 through the connector: a prefix whose block-level
    match lands past the last complete sidecar boundary truncates."""
    layout = [
        FullAttn(name="full_attn", layer_indices=[0, 2],
                 num_kv_heads=2, head_size=64, tokens_per_block=2),
        LinearState(name="linear_state", layer_indices=[1, 3],
                    state_bytes=4096),
    ]
    worker = make_connector(layout=layout)
    scheduler = StrataKVConnectorV1(role="scheduler")
    caches = {
        "full_attn_0": FakeTensor((32, 2, 2 * 2 * 64)),
        "full_attn_1": FakeTensor((32, 2, 2 * 2 * 64)),
        "linear_state_0": FakeTensor((32, 4096 // 2)),
        "linear_state_1": FakeTensor((32, 4096 // 2)),
    }
    worker.register_kv_caches(caches)

    # Two staged puts: boundary at block 2 (state snapshotted), then
    # extended to block 4 (state again) — mirrors the core hybrid test.
    put1 = FakeRequest("h1", [1, 2, 3, 4])
    assert scheduler.get_num_new_matched_tokens(put1, 0) == (0, False)
    scheduler.request_finished(put1, {"full_attn": [0, 1]})
    assert pump(scheduler, worker, "h1", sending=True)

    put2 = FakeRequest("h2", [1, 2, 3, 4, 5, 6, 7, 8])
    scheduler.get_num_new_matched_tokens(put2, 0)  # Extends (2+2).
    scheduler.request_finished(put2, {"full_attn": [2, 3]})
    assert pump(scheduler, worker, "h2", sending=True)

    # Query sharing 3 of 4 blocks: must truncate to block 2 — the last
    # COMPLETE bundle boundary (block 3's sidecar is stale/absent).
    query = FakeRequest("h3", [1, 2, 3, 4, 5, 6])
    matched, async_load = scheduler.get_num_new_matched_tokens(query, 0)
    assert matched == 2 * 2  # 2 blocks x 2 tpb, NOT 3 blocks.
    assert async_load
    scheduler.shutdown()
    for tensor in caches.values():
        tensor.free()


def test_swa_group_registers_store_all():
    """F3: sliding-window groups register token-addressable (store-all
    v1; window-trim is a later optimization per the roadmap)."""
    layout = [
        FullAttn(name="full_attn", layer_indices=[0, 1],
                 num_kv_heads=2, head_size=64, tokens_per_block=16),
        SlidingWindow(name="swa", layer_indices=[2, 3],
                      num_kv_heads=2, head_size=64, window_blocks=8,
                      tokens_per_block=16),
    ]
    worker = make_connector(layout=layout)
    caches = {name: FakeTensor((32, 16, 2 * 2 * 64))
              for name in ("full_attn_0", "full_attn_1",
                           "swa_2", "swa_3")}
    worker.register_kv_caches(caches)
    assert worker._shared.started
    scheduler = StrataKVConnectorV1(role="scheduler")
    tokens = list(range(3 * 16))
    request = FakeRequest("swa-req", tokens)
    assert scheduler.get_num_new_matched_tokens(request, 0) == (0, False)
    delays, _ = scheduler.request_finished(
        request, {"full_attn": [0, 1, 2], "swa": [10, 11, 12]})
    assert delays
    assert pump(scheduler, worker, "swa-req", sending=True)
    scheduler.shutdown()
    for tensor in caches.values():
        tensor.free()


def test_shutdown_with_tasks_in_flight_survives():
    """Torture: shutdown while puts are still executing must be clean
    (engine cancels, transport drains, process healthy)."""
    worker = make_connector()
    scheduler = StrataKVConnectorV1(role="scheduler")
    caches = make_kv_caches()
    worker.register_kv_caches(caches)
    for round_index in range(20):
        request = FakeRequest(f"torture-{round_index}",
                              list(range(4 * TPB)))
        scheduler.get_num_new_matched_tokens(request, 0)
        scheduler.request_finished(request, [0, 1, 2, 3])
        if round_index % 3 == 0:
            scheduler.update_connector_output(None)  # Some mid-flight.
    scheduler.shutdown()
    for tensor in caches.values():
        tensor.free()
    # A fresh engine works afterwards (process healthy).
    worker2 = make_connector()
    caches2 = make_kv_caches()
    worker2.register_kv_caches(caches2)
    assert worker2._shared.started
    worker2.shutdown()
    for tensor in caches2.values():
        tensor.free()


def test_real_torch_tensors_round_trip():
    """The duck-typed registration path against REAL torch CUDA
    tensors (runs where torch + the GPU module are both present — the
    remote E0 host; skipped on torch-less dev boxes)."""
    torch = pytest.importorskip("torch")
    if not GPU:
        pytest.skip("native module without GPU transport")
    worker = make_connector()
    scheduler = StrataKVConnectorV1(role="scheduler")
    caches = {f"full_attn_layer_{i}":
              torch.zeros((BLOCKS, TPB, KV_DIM), dtype=torch.float16,
                          device="cuda")
              for i in range(LAYERS)}
    worker.register_kv_caches(caches)
    assert worker._shared.started

    tokens = list(range(4 * TPB))
    golden = {i: torch.randint(
                  0, 255, (4, TPB, KV_DIM), dtype=torch.int16,
                  device="cpu").to(torch.float16)
              for i in range(LAYERS)}
    for i in range(LAYERS):
        caches[f"full_attn_layer_{i}"][:4].copy_(golden[i])

    cold = FakeRequest("torch-cold", tokens)
    assert scheduler.get_num_new_matched_tokens(cold, 0) == (0, False)
    scheduler.request_finished(cold, [0, 1, 2, 3])
    assert pump(scheduler, worker, "torch-cold", sending=True)

    warm = FakeRequest("torch-warm", tokens)
    matched, async_load = scheduler.get_num_new_matched_tokens(warm, 0)
    assert matched == 4 * TPB and async_load
    FakeSchedulerStep.run(scheduler, [warm], {"torch-warm": [20, 21, 22,
                                                             23]})
    assert pump(scheduler, worker, "torch-warm", sending=False)
    for i in range(LAYERS):
        reloaded = caches[f"full_attn_layer_{i}"][20:24]
        assert torch.equal(reloaded.cpu(), golden[i]), \
            f"layer {i}: KV bytes changed across offload+reload"
    scheduler.shutdown()
