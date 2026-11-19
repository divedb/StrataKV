"""Phase D acceptance (roadmap.md): the pytest suite drives the native
engine from Python, and the GIL is verifiably released during waits."""

from __future__ import annotations

import threading
import time

import pytest

import stratakv
from stratakv import Engine, EngineConfig, FullAttn, LinearState

TOKENS = [1, 2, 3, 4, 5, 6, 7, 8]  # 4 blocks at tokens_per_block=2.


def homogeneous_engine(**config_overrides) -> Engine:
    config = EngineConfig(blocks_per_group=64,
                          state_objects_per_group=8)
    for key, value in config_overrides.items():
        setattr(config, key, value)
    return Engine(
        [FullAttn(name="full_attn", layer_indices=[0, 1, 2, 3],
                  num_kv_heads=2, head_size=64, tokens_per_block=2)],
        config)


def state_of(engine: Engine, handle) -> str:
    return engine.try_wait([handle])[0][1]


def test_native_surface_intact():
    assert stratakv.NATIVE_AVAILABLE
    assert stratakv.version() == "0.1.0"
    assert "full_attn" in stratakv.spec_names()


def test_layout_validation_maps_to_value_error():
    with pytest.raises(ValueError):
        Engine([FullAttn(name="", layer_indices=[0], num_kv_heads=2,
                         head_size=64)])  # Empty name is invalid.


def test_put_get_lifecycle():
    engine = homogeneous_engine()
    plan = engine.stage_put(TOKENS)
    assert plan.new_blocks == 4
    assert plan.feasible

    put = engine.submit_put(plan, gpu_blocks=[0, 1, 2, 3])
    assert state_of(engine, put) == "succeeded"  # Inline executor.

    match = engine.match(TOKENS)
    assert match.matched_blocks == 4
    assert match.matched_blocks_reserved == 0
    assert [b for _, b in match.blocks] == [0, 1, 2, 3]
    assert match.token_mask == [1] * 8

    get = engine.submit_get(match, gpu_slots=[10, 11, 12, 13])
    result = engine.wait([get], timeout_ms=1000)
    assert result[0][1] == "succeeded"
    assert engine.num_in_flight() == 0

    engine.shutdown()
    engine.shutdown()  # Idempotent.


def test_cold_match_is_empty_and_release_is_idempotent():
    engine = homogeneous_engine()
    match = engine.match([5, 6])
    assert match.matched_blocks == 0
    engine.release_match(match)
    engine.release_match(match)  # Second release is a no-op per id.


def test_hybrid_no_partial_caching():
    engine = Engine(
        [FullAttn(name="full_attn", layer_indices=[0, 3], num_kv_heads=4,
                  head_size=128, tokens_per_block=2),
         LinearState(name="linear_state", layer_indices=[1, 2],
                     state_bytes=4096)],
        EngineConfig(blocks_per_group=64, state_objects_per_group=8))

    # Two boundaries: block 2 (first put) and block 4 (its extension).
    plan1 = engine.stage_put([1, 2, 3, 4])
    assert engine.try_wait([engine.submit_put(plan1, gpu_blocks=[0, 1])])[0][1] == "succeeded"
    plan2 = engine.stage_put(TOKENS)  # Extends: 2 matched + 2 new.
    assert plan2.matched_blocks == 2
    put2 = engine.submit_put(plan2, gpu_blocks=[2, 3])
    assert state_of(engine, put2) == "succeeded"

    # Query sharing 3 of 4 blocks: serving truncates to the last
    # COMPLETE bundle boundary (block 2) — the third block has no state
    # snapshot, so resuming there would use stale linear states: the
    # exact wrong-attention failure ADR 0003 forbids.
    match = engine.match([1, 2, 3, 4, 5, 6])
    assert match.matched_blocks == 2
    assert match.matched_blocks_reserved == 1
    assert len(match.states) == 1  # Sidecar at the servable boundary.
    engine.release_match(match)

    # A query landing exactly on a boundary serves fully.
    match = engine.match([1, 2, 3, 4])
    assert match.matched_blocks == 2
    engine.release_match(match)


def test_wait_timeout_then_success_with_simulated_latency():
    engine = homogeneous_engine(inline_delay_ms=30)
    plan = engine.stage_put(TOKENS)

    # The inline executor runs inside submit_put (GIL released), so the
    # in-flight window is only observable from another thread.
    submitted = []
    submitter = threading.Thread(
        target=lambda: submitted.append(
            engine.submit_put(plan, gpu_blocks=[0, 1, 2, 3])))
    start = time.monotonic()
    submitter.start()

    while submitter.is_alive():
        if submitted:
            # Handle already returned: poll its live state.
            states = [s for _, s, _ in engine.try_wait(submitted)]
        else:
            # No handle yet: a wait on an unknown id must time out
            # cleanly rather than block (partial-result contract).
            pending = engine.wait([stratakv.TaskHandle(999)], timeout_ms=5)
            assert pending[0][1] == "staged"  # Unknown-handle marker.
        time.sleep(0.002)
    submitter.join()

    done = engine.wait(submitted, timeout_ms=1000)
    assert done[0][1] == "succeeded"
    # One D2H op (per group) x 30ms simulated latency.
    assert time.monotonic() - start >= 0.030


def test_blocking_native_calls_release_gil_heartbeat():
    """ADR 0004 acceptance: a Python heartbeat thread must keep running
    while another thread blocks in native engine code (submit with
    simulated latency + wait). Impossible with the GIL held."""
    engine = homogeneous_engine(inline_delay_ms=100)
    plan = engine.stage_put(TOKENS)

    submitted = []
    submitter = threading.Thread(
        target=lambda: submitted.append(
            engine.submit_put(plan, gpu_blocks=[0, 1, 2, 3])))

    beats = [0]
    stop = threading.Event()

    def heartbeat() -> None:
        while not stop.is_set():
            beats[0] += 1
            time.sleep(0.002)

    heartbeat_thread = threading.Thread(target=heartbeat)
    heartbeat_thread.start()
    try:
        submitter.start()
        submitter.join()  # ~100ms blocked in native submit (GIL out).
        result = engine.wait(submitted, timeout_ms=5000)
        assert result[0][1] == "succeeded"
    finally:
        stop.set()
        heartbeat_thread.join()

    # ~100ms of blocked native calls: the heartbeat (2ms period) must
    # have ticked dozens of times throughout — zero with the GIL held.
    assert beats[0] > 20, f"heartbeat starved: {beats[0]} beats"


def test_cancel_terminal_is_noop_and_shutdown_rejects():
    engine = homogeneous_engine()
    plan = engine.stage_put(TOKENS)
    put = engine.submit_put(plan, gpu_blocks=[0, 1, 2, 3])
    assert state_of(engine, put) == "succeeded"
    assert engine.cancel(put) is False

    engine.shutdown()
    plan2 = engine.stage_put([42, 43])
    with pytest.raises(stratakv.LogicError):
        engine.submit_put(plan2, gpu_blocks=[0])


def test_abort_put_recycles_reservation():
    engine = homogeneous_engine()
    free_before = engine.num_free_blocks(group=0)
    plan = engine.stage_put(TOKENS)
    assert engine.num_free_blocks(group=0) == free_before - 4

    engine.abort_put(plan)
    assert engine.num_free_blocks(group=0) == free_before
    assert engine.match(TOKENS).matched_blocks == 0


def test_pools_exhaustion_raises():
    engine = homogeneous_engine(blocks_per_group=4)
    engine.stage_put(TOKENS)  # 4 blocks — pools now empty.
    with pytest.raises(stratakv.NotEnoughSpaceError):
        engine.stage_put([9, 10, 11, 12])
