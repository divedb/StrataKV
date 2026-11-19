#!/usr/bin/env python3
"""Phase E: Python-layer per-task cost bench (STRATAKV_GPU builds).

Drives the REAL stack from Python — Engine.register_gpu_group/start +
stage_put/submit_put/wait, match/submit_get/wait — at the exact Phase 0
/ E0 geometry (24 layers x 8 KiB = 192 KiB model-wide block), so the
delta against the C++ bench (gpu_transport_bench) prices the pybind +
facade layer that the future vLLM connector will pay per step.

Byte verification (round trip via the debug device helpers) precedes
every timed scenario. Not a ctest entry (latency tables are read by
humans per roadmap §5). Run:

    PYTHONPATH=python python3 benchmarks/gpu_transport_py_bench.py
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "python"))

import stratakv  # noqa: E402
from stratakv import Engine, EngineConfig, FullAttn  # noqa: E402

LAYERS = 24
PER_LAYER = 8192
SLOT_BYTES = LAYERS * PER_LAYER
TPB = 16
GPU_SLOTS = 512
POOL_BLOCKS = 600


def die(msg: str) -> None:
    print(f"FATAL: {msg}", file=sys.stderr)
    sys.exit(1)


if not stratakv.GPU_TRANSPORT_AVAILABLE:
    die("native module built without STRATAKV_GPU (simulation only)")

try:
    probe = stratakv.debug_alloc_device(8)
    stratakv.debug_free_device(probe)
except RuntimeError as e:
    die(f"no usable CUDA device: {e}")


def pattern(seed: int) -> bytes:
    state = seed
    out = bytearray()
    for _ in range(SLOT_BYTES // 8):
        state = (state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        out += (z ^ (z >> 31)).to_bytes(8, "little")
    return bytes(out)


def tokens_for(blocks: int, salt: int) -> list[int]:
    state = salt
    out = []
    for _ in range(blocks * TPB):
        state = (state * 6364136223846793005
                 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        out.append((state >> 33) % 100000)
    return out


def pctl(samples: list[float], p: float) -> float:
    ordered = sorted(samples)
    return ordered[int(p * (len(ordered) - 1))]


def fit_intercept(rows: list[tuple[int, float]]) -> tuple[float, float]:
    n = len(rows)
    xs = [b * SLOT_BYTES for b, _ in rows]
    ys = [t for _, t in rows]
    sx, sy = sum(xs), sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sxx - sx * sx
    slope = (n * sxy - sx * sy) / denom if denom else 0.0
    return (sy - slope * sx) / n, slope


golden = pattern(0xC0FFEE)
addrs = [stratakv.debug_alloc_device(GPU_SLOTS * PER_LAYER)
         for _ in range(LAYERS)]


def write_block(slot: int) -> None:
    for layer in range(LAYERS):
        stratakv.debug_write_device(
            addrs[layer] + slot * PER_LAYER,
            golden[layer * PER_LAYER:(layer + 1) * PER_LAYER])


def read_block(slot: int) -> bytes:
    out = b""
    for layer in range(LAYERS):
        out += stratakv.debug_read_device(addrs[layer] + slot * PER_LAYER,
                                          PER_LAYER)
    return out


engine = Engine(
    [FullAttn(name="full_attn", layer_indices=list(range(LAYERS)),
              num_kv_heads=2, head_size=64, tokens_per_block=TPB)],
    EngineConfig(blocks_per_group=POOL_BLOCKS, state_objects_per_group=4,
                 use_gpu_transport=True, cpu_backing="pinned",
                 transport_streams=2))
engine.register_gpu_group(
    "full_attn", 0, [(a, GPU_SLOTS * PER_LAYER) for a in addrs], GPU_SLOTS)
engine.start()

print("== StrataKV Python-layer transport bench (real stack) ==")
print(f"block geometry: {LAYERS} layers x {PER_LAYER} B = {SLOT_BYTES} B "
      "| pinned CPU tier | in-process | CPython "
      f"{sys.version_info.major}.{sys.version_info.minor}")
print()

sizes = [1, 4, 16, 64]
iters = {1: 100, 4: 100, 16: 100, 64: 50}
salt = 1000
put_rows: list[tuple[int, float]] = []
get_rows: list[tuple[int, float]] = []
match_rows: list[tuple[int, float]] = []

print("== SYNC mode: Python submit -> wait per task ==")
print(f"{'task':<6}{'blocks':>7}{'bytes':>10}{'p50_us':>10}"
      f"{'p95_us':>10}{'p99_us':>10}{'GB/s':>8}")

# ---- put loop (fresh prefixes; evict between iterations) ----
for slot in range(64):
    write_block(slot)  # Source slots hold the golden pattern.
for n in sizes:
    # Byte verification: put -> get -> compare.
    tokens = tokens_for(n, salt)
    plan = engine.stage_put(tokens)
    put = engine.submit_put(plan, gpu_blocks=list(range(n)))
    if engine.wait([put], timeout_ms=10000)[0][1] != "succeeded":
        die(f"verify put failed (n={n})")
    match = engine.match(tokens)
    got = engine.submit_get(match, gpu_slots=[256 + b for b in range(n)])
    if engine.wait([got], timeout_ms=10000)[0][1] != "succeeded":
        die(f"verify get failed (n={n})")
    for b in range(n):
        if read_block(256 + b) != golden:
            die(f"bytes corrupt (n={n}, block {b})")
    engine.evict_blocks(n)
    salt += 1

    samples = []
    for _ in range(iters[n]):
        plan = engine.stage_put(tokens_for(n, salt))
        salt += 1
        t0 = time.perf_counter_ns()
        put = engine.submit_put(plan, gpu_blocks=list(range(n)))
        engine.wait([put], timeout_ms=10000)
        samples.append((time.perf_counter_ns() - t0) / 1000.0)
        engine.evict_blocks(n)
    p50 = pctl(samples, 0.50)
    put_rows.append((n, p50))
    print(f"{'put':<6}{n:>7}{n * SLOT_BYTES:>10}{p50:>10.1f}"
          f"{pctl(samples, 0.95):>10.1f}{pctl(samples, 0.99):>10.1f}"
          f"{n * SLOT_BYTES / (p50 * 1000):>8.2f}")

# ---- get loop (one pinned prime prefix) ----
prime = tokens_for(64, 7)
plan = engine.stage_put(prime)
put = engine.submit_put(plan, gpu_blocks=list(range(64)))
if engine.wait([put], timeout_ms=10000)[0][1] != "succeeded":
    die("prime put failed")

for n in sizes:
    # Match-only cost (metadata plane, priced separately).
    samples = []
    for _ in range(200):
        t0 = time.perf_counter_ns()
        match = engine.match(prime[: n * TPB])
        samples.append((time.perf_counter_ns() - t0) / 1000.0)
        engine.release_match(match)
    match_rows.append((n, pctl(samples, 0.50)))

    # Byte verification for the get leg.
    match = engine.match(prime[: n * TPB])
    got = engine.submit_get(match, gpu_slots=[256 + b for b in range(n)])
    engine.wait([got], timeout_ms=10000)
    for b in range(n):
        if read_block(256 + b) != golden:
            die(f"get bytes corrupt (n={n}, block {b})")

    samples = []
    for _ in range(iters[n]):
        match = engine.match(prime[: n * TPB])
        t0 = time.perf_counter_ns()
        got = engine.submit_get(match, gpu_slots=list(range(n)))
        engine.wait([got], timeout_ms=10000)
        samples.append((time.perf_counter_ns() - t0) / 1000.0)
    p50 = pctl(samples, 0.50)
    get_rows.append((n, p50))
    print(f"{'get':<6}{n:>7}{n * SLOT_BYTES:>10}{p50:>10.1f}"
          f"{pctl(samples, 0.95):>10.1f}{pctl(samples, 0.99):>10.1f}"
          f"{n * SLOT_BYTES / (p50 * 1000):>8.2f}")

print()
print("== Match-only latency (Python facade, p50 us) ==")
for n, p50 in match_rows:
    print(f"  {n:>3} blocks: {p50:.1f}")

put_a, put_b = fit_intercept(put_rows)
get_a, get_b = fit_intercept(get_rows)
print()
print("== Per-task fixed overhead (0-byte intercept, sync p50) ==")
print(f"{'task':<6}{'intercept_us':>15}{'marginal_GB/s':>17}")
print(f"{'put':<6}{put_a:>15.1f}"
      f"{(1.0 / (put_b * 1000)) if put_b > 0 else 0:>17.2f}")
print(f"{'get':<6}{get_a:>15.1f}"
      f"{(1.0 / (get_b * 1000)) if get_b > 0 else 0:>17.2f}")

print()
print("== References (same box, same geometry) ==")
put4 = dict(put_rows).get(4)
get4 = dict(get_rows).get(4)
if put4 is not None and get4 is not None:
    worst = max(put4, get4)
    print(f"acceptance bar (roadmap E): <= 1000 us at 4 blocks -> "
          f"{'PASS' if worst <= 1000 else 'FAIL'} "
          f"(python put {put4:.1f} / get {get4:.1f} us)")
print("C++ full stack (gpu_transport_bench): put 184.8 / get 194.6 us "
      "at 4 blocks; intercepts 135.6 / 163.6 us")
print("Phase 0 raw kernel floor: h2d 74.9 / d2h 82.9 us at 4 blocks")
print("FlexKV v1.0.0 (E0): 10850-21040 us at ~4 blocks")
print("verifications passed")

engine.shutdown(3000.0)
for a in addrs:
    stratakv.debug_free_device(a)
