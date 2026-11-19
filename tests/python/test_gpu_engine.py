"""Phase E (GPU portion), Python side: the real data plane through the
typed facade — register_gpu_group/start lifecycle, byte-verified put/get
round trips via the debug device helpers, loud pre-start failures, and
the ADR 0004 GIL check against REAL asynchronous completion (the
InlineExecutor suite cannot exercise that: nothing ever completes from
another thread there).

Runs only when the native module was built with STRATAKV_GPU
(GPU_TRANSPORT_AVAILABLE marker) and a usable CUDA device exists;
collects as skip otherwise (CI shells keep the simulation suite).
"""

from __future__ import annotations

import threading
import time

import pytest

import stratakv
from stratakv import Engine, EngineConfig, FullAttn

pytestmark = pytest.mark.skipif(
    not stratakv.GPU_TRANSPORT_AVAILABLE,
    reason="native module built without STRATAKV_GPU (simulation only)")


@pytest.fixture(scope="module")
def device_ok() -> bool:
    try:
        addr = stratakv.debug_alloc_device(8)
        stratakv.debug_free_device(addr)
        return True
    except RuntimeError:
        return False


def require_device(device_ok: bool) -> None:
    if not device_ok:
        pytest.skip("no usable CUDA device")


# Qwen2-0.5B-shaped geometry, scaled down: 4 layers x 8 KiB per-layer
# slices = 32 KiB model-wide blocks, 16 tokens per block.
LAYERS = 4
PER_LAYER = 8192
SLOT_BYTES = LAYERS * PER_LAYER
TPB = 16
CAPACITY = 64


def layout_groups():
    return [FullAttn(name="full_attn", layer_indices=list(range(LAYERS)),
                     num_kv_heads=2, head_size=64, tokens_per_block=TPB)]


def make_engine(backing: str = "pinned") -> Engine:
    return Engine(layout_groups(),
                  EngineConfig(blocks_per_group=CAPACITY,
                               state_objects_per_group=4,
                               use_gpu_transport=True,
                               cpu_backing=backing))


class GpuGroup:
    """Stand-in for engine-owned GPU memory (debug helpers)."""

    def __init__(self, capacity: int, per_layer: int = PER_LAYER):
        self.addrs = [stratakv.debug_alloc_device(capacity * per_layer)
                      for _ in range(LAYERS)]
        self.capacity = capacity
        self.per_layer = per_layer

    def free(self) -> None:
        for addr in self.addrs:
            stratakv.debug_free_device(addr)

    def write_block(self, slot: int, data: bytes) -> None:
        assert len(data) == SLOT_BYTES
        for layer in range(LAYERS):
            stratakv.debug_write_device(
                self.addrs[layer] + slot * self.per_layer,
                data[layer * self.per_layer:(layer + 1) * self.per_layer])

    def read_block(self, slot: int) -> bytes:
        out = b""
        for layer in range(LAYERS):
            out += stratakv.debug_read_device(
                self.addrs[layer] + slot * self.per_layer, self.per_layer)
        return out


def pattern(seed: int) -> bytes:
    state = seed
    words = []
    for _ in range(SLOT_BYTES // 8):
        state = (state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        words.append(z ^ (z >> 31))
    return b"".join(w.to_bytes(8, "little") for w in words)


def tokens_for(blocks: int, salt: int) -> list[int]:
    state = salt
    out = []
    for _ in range(blocks * TPB):
        state = (state * 6364136223846793005 + 1442695040888963407) \
            & 0xFFFFFFFFFFFFFFFF
        out.append((state >> 33) % 100000)
    return out


def state_of(engine: Engine, handle) -> str:
    return engine.try_wait([handle])[0][1]


def test_round_trip_put_then_get(device_ok):
    require_device(device_ok)
    gpu = GpuGroup(CAPACITY)
    try:
        engine = make_engine("pinned")
        engine.register_gpu_group(
            "full_attn", 0,
            [(addr, gpu.capacity * gpu.per_layer) for addr in gpu.addrs],
            gpu.capacity, layout="layerwise")
        engine.start()

        blocks = 4
        golden = [pattern(0xC0FFEE + b) for b in range(blocks)]
        for b, data in enumerate(golden):
            gpu.write_block(b, data)

        # Put: GPU slots 0..3 -> CPU tier.
        plan = engine.stage_put(tokens_for(blocks, 1))
        put = engine.submit_put(plan, gpu_blocks=list(range(blocks)))
        assert engine.wait([put], timeout_ms=5000)[0][1] == "succeeded"

        # Get: CPU tier -> GPU slots 32..35; bytes must round-trip.
        match = engine.match(tokens_for(blocks, 1))
        assert match.matched_blocks == blocks
        got = engine.submit_get(match, gpu_slots=[32 + b for b in
                                                  range(blocks)])
        assert engine.wait([got], timeout_ms=5000)[0][1] == "succeeded"
        for b, data in enumerate(golden):
            assert gpu.read_block(32 + b) == data
        engine.shutdown(2000.0)
    finally:
        gpu.free()


def test_pageable_backing_round_trip(device_ok):
    """The copy-engine fallback path (non-SM-safe host memory)."""
    require_device(device_ok)
    gpu = GpuGroup(CAPACITY)
    try:
        engine = make_engine("pageable")
        engine.register_gpu_group(
            "full_attn", 0,
            [(addr, gpu.capacity * gpu.per_layer) for addr in gpu.addrs],
            gpu.capacity, layout="layerwise")
        engine.start()
        golden = pattern(7)
        gpu.write_block(0, golden)
        plan = engine.stage_put(tokens_for(1, 42))
        put = engine.submit_put(plan, gpu_blocks=[0])
        assert engine.wait([put], timeout_ms=5000)[0][1] == "succeeded"
        match = engine.match(tokens_for(1, 42))
        got = engine.submit_get(match, gpu_slots=[1])
        assert engine.wait([got], timeout_ms=5000)[0][1] == "succeeded"
        assert gpu.read_block(1) == golden
        engine.shutdown(2000.0)
    finally:
        gpu.free()


def test_pre_start_contract(device_ok):
    require_device(device_ok)
    engine = make_engine()
    gpu = GpuGroup(CAPACITY)
    try:
        # Metadata-side calls work pre-start (they only touch the
        # plane): staging, matching (reserved here — the entry is not
        # ready), release, abort.
        plan = engine.stage_put(tokens_for(1, 3))
        assert plan.new_blocks == 1
        match = engine.match(tokens_for(1, 3))
        assert match.matched_blocks == 0
        assert match.matched_blocks_reserved == 1
        engine.release_match(match)
        engine.abort_put(plan)
        # ...submissions do not (loud, not silent).
        plan = engine.stage_put(tokens_for(1, 3))
        with pytest.raises(stratakv.LogicError):
            engine.submit_put(plan, gpu_blocks=[0])
        engine.abort_put(plan)
        # start() without registrations fails loudly.
        with pytest.raises(stratakv.LogicError):
            engine.start()
        engine.register_gpu_group(
            "full_attn", 0,
            [(addr, gpu.capacity * gpu.per_layer) for addr in gpu.addrs],
            gpu.capacity, layout="layerwise")
        engine.start()
        with pytest.raises(stratakv.LogicError):
            engine.start()  # Double start is a bug, not a no-op.
        engine.shutdown(1000.0)
    finally:
        gpu.free()


def test_gil_heartbeat_under_real_completion(device_ok):
    """ADR 0004: waits release the GIL while the transport worker
    completes ops from its own thread — a Python heartbeat thread must
    not starve (the InlineExecutor cannot test this: nothing completes
    asynchronously there). Real transfers are fast, so the assertion
    self-scales: many wait rounds span a measured wall-time window and
    the 1 kHz heartbeat must tick in proportion (generous slack)."""
    require_device(device_ok)
    gpu = GpuGroup(CAPACITY)
    try:
        engine = make_engine("pinned")
        engine.register_gpu_group(
            "full_attn", 0,
            [(addr, gpu.capacity * gpu.per_layer) for addr in gpu.addrs],
            gpu.capacity, layout="layerwise")
        engine.start()
        golden = pattern(0x5EED)
        prime_tokens = tokens_for(16, 11)
        for block in range(16):
            gpu.write_block(block, golden)
        plan = engine.stage_put(prime_tokens)
        put = engine.submit_put(plan, gpu_blocks=list(range(16)))
        assert engine.wait([put], timeout_ms=5000)[0][1] == "succeeded"

        def one_round() -> list:
            gets = []
            for _ in range(16):
                match = engine.match(prime_tokens)
                gets.append(engine.submit_get(match,
                                              gpu_slots=list(range(16))))
            return engine.wait(gets, timeout_ms=5000)

        beats = [0.0]
        stop = threading.Event()

        def heartbeat() -> None:
            while not stop.is_set():
                beats[0] += 1
                time.sleep(0.001)

        thread = threading.Thread(target=heartbeat)
        thread.start()
        try:
            started = time.monotonic()
            for _ in range(20):
                states = one_round()
                assert all(s == "succeeded" for _, s, _ in states)
            elapsed = time.monotonic() - started
        finally:
            stop.set()
            thread.join()
        # The GIL was released for ~elapsed seconds of blocked waiting;
        # a 1 kHz heartbeat must have ticked (5x slack for scheduler
        # jitter; floor of 2 because the window is nonzero by
        # construction — every round blocks on real completions).
        assert beats[0] >= max(2, int(elapsed * 1000 / 5)), (
            f"heartbeat starved: {beats[0]:.0f} beats in "
            f"{elapsed * 1000:.1f} ms of waiting")
        engine.shutdown(2000.0)
    finally:
        gpu.free()
