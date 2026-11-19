"""StrataKV Python package.

Typed, torch-free facade over the native core (pybind11 module
``stratakv._core``). Phase D surface (roadmap.md / design.md):

- layout description (heterogeneous groups from day one, ADR 0003);
- :class:`Engine` — match / stage_put / submit_put / submit_get /
  try_wait / wait / cancel / shutdown driving MetadataPlane + TaskEngine
  through the InlineExecutor simulation transport (no bytes are moved;
  Phase E installs the CUDA transport).

The native module is optional at import time so that the package
metadata stays importable in docs/CI shells, but :class:`Engine`
fails loudly when the extension is absent (no silent stubs).
"""

from __future__ import annotations

import dataclasses
from typing import Literal, Sequence

try:  # pragma: no cover - depends on build configuration
    from stratakv import _core as _native

    NATIVE_AVAILABLE = True
    NotEnoughSpaceError = _native.NotEnoughSpaceError
    LogicError = _native.LogicError
except ImportError:  # pragma: no cover
    _native = None
    NATIVE_AVAILABLE = False

__version__ = "0.1.0"

__all__ = [
    "NATIVE_AVAILABLE",
    "Engine",
    "EngineConfig",
    "FullAttn",
    "LinearState",
    "LogicError",
    "MatchInfo",
    "Mla",
    "NotEnoughSpaceError",
    "PutPlan",
    "SlidingWindow",
    "TaskHandle",
    "device_kinds",
    "spec_names",
    "version",
]

GroupKind = Literal["full_attn", "mla", "sliding_window", "linear_state"]


def version() -> str:
    """Native core version (falls back to package version offline)."""
    return _native.version() if _native is not None else __version__


def spec_names() -> list[str]:
    return list(_native.spec_names()) if _native is not None else []


def device_kinds() -> list[str]:
    return list(_native.device_kinds()) if _native is not None else []


@dataclasses.dataclass(frozen=True)
class FullAttn:
    """Full-attention KV geometry (paged token blocks)."""

    name: str
    layer_indices: Sequence[int]
    num_kv_heads: int
    head_size: int
    tokens_per_block: int = 16

    kind: GroupKind = dataclasses.field(default="full_attn", init=False)


@dataclasses.dataclass(frozen=True)
class Mla:
    """Multi-head latent attention (absorbed single latent tensor)."""

    name: str
    layer_indices: Sequence[int]
    kv_lora_rank: int
    rope_dim: int
    tokens_per_block: int = 16

    kind: GroupKind = dataclasses.field(default="mla", init=False)


@dataclasses.dataclass(frozen=True)
class SlidingWindow:
    """Sliding-window attention KV (window-bounded reuse)."""

    name: str
    layer_indices: Sequence[int]
    num_kv_heads: int
    head_size: int
    window_blocks: int
    tokens_per_block: int = 16

    kind: GroupKind = dataclasses.field(default="sliding_window", init=False)


@dataclasses.dataclass(frozen=True)
class LinearState:
    """Linear-attention / SSM recurrent state (sidecar, ADR 0003)."""

    name: str
    layer_indices: Sequence[int]
    state_bytes: int

    kind: GroupKind = dataclasses.field(default="linear_state", init=False)


@dataclasses.dataclass
class EngineConfig:
    """Pool bounds + engine backpressure + simulation latency."""

    blocks_per_group: int = 1 << 20
    state_objects_per_group: int = 1024
    state_object_bytes: int = 1 << 16
    max_in_flight: int = 1024
    inline_delay_ms: int = 0  # Simulated per-op latency (testing only).


@dataclasses.dataclass(frozen=True)
class TaskHandle:
    """Opaque task id returned by submit_* (0/None = invalid)."""

    task_id: int


@dataclasses.dataclass(frozen=True)
class MatchInfo:
    """Result of Engine.match (pin held until submit_get/release)."""

    match_id: int
    matched_blocks: int
    matched_blocks_reserved: int
    blocks: list[tuple[int, int]]  # (group, tier-local block id)
    states: list[tuple[int, int]]  # (group, state object id)
    token_mask: list[int]


@dataclasses.dataclass(frozen=True)
class PutPlan:
    """Result of Engine.plan_put / Engine.stage_put."""

    plan_id: int
    matched_blocks: int
    new_blocks: int
    feasible: bool


class Engine:
    """Owning handle to a StrataKV runtime instance (one per process).

    GIL policy (ADR 0004): ``wait``/``submit_*``/``shutdown`` release
    the interpreter lock while blocked in native code; ``match`` holds
    it (sub-millisecond budget); completion delivery is poll-based via
    ``completion_fd`` — no native thread ever calls into Python.
    """

    def __init__(self, groups: Sequence[FullAttn | Mla | SlidingWindow
                                        | LinearState],
                 config: EngineConfig | None = None) -> None:
        if _native is None:
            raise RuntimeError(
                "stratakv native module not built — configure CMake with "
                "-DSTRATAKV_PYTHON=ON (see bindings/python/CMakeLists.txt)")
        native_groups = []
        for g in groups:
            ng = _native.LayoutGroup()
            ng.name = g.name
            ng.kind = g.kind
            ng.layer_indices = list(g.layer_indices)
            tokens = getattr(g, "tokens_per_block", 0)
            ng.tokens_per_block = tokens
            for field in ("num_kv_heads", "head_size", "kv_lora_rank",
                          "rope_dim", "window_blocks", "state_bytes"):
                setattr(ng, field, getattr(g, field, 0))
            native_groups.append(ng)
        cfg = _native.EngineConfig()
        if config is not None:
            cfg.blocks_per_group = config.blocks_per_group
            cfg.state_objects_per_group = config.state_objects_per_group
            cfg.state_object_bytes = config.state_object_bytes
            cfg.max_in_flight = config.max_in_flight
            cfg.inline_delay_ms = config.inline_delay_ms
        self._engine = _native.Engine(native_groups, cfg)

    # ---- match (GIL held; sub-ms budget) ----

    def match(self, token_ids: Sequence[int]) -> MatchInfo:
        """Pinned readiness-filtered prefix match."""
        m = self._engine.match([int(t) for t in token_ids])
        return MatchInfo(m.match_id, m.matched_blocks,
                         m.matched_blocks_reserved, list(m.blocks),
                         list(m.states), list(m.token_mask))

    def release_match(self, match: MatchInfo) -> None:
        """Drop a match's pins (request abort). Idempotent per match."""
        self._engine.release_match(match.match_id)

    # ---- put path ----

    def plan_put(self, token_ids: Sequence[int]) -> PutPlan:
        p = self._engine.plan_put([int(t) for t in token_ids])
        return PutPlan(p.plan_id, p.matched_blocks, p.new_blocks, p.feasible)

    def stage_put(self, token_ids: Sequence[int]) -> PutPlan:
        p = self._engine.stage_put([int(t) for t in token_ids])
        return PutPlan(p.plan_id, p.matched_blocks, p.new_blocks, p.feasible)

    def submit_put(self, plan: PutPlan,
                   gpu_blocks: Sequence[int],
                   gpu_state_slots: Sequence[int] = ()) -> TaskHandle:
        """Launch the offload (GIL released)."""
        return TaskHandle(
            self._engine.submit_put(plan.plan_id,
                                    [int(b) for b in gpu_blocks],
                                    [int(s) for s in gpu_state_slots]))

    def abort_put(self, plan: PutPlan) -> None:
        """Abort a staged put; recycles its reservation."""
        self._engine.abort_put(plan.plan_id)

    # ---- get path ----

    def submit_get(self, match: MatchInfo,
                   gpu_slots: Sequence[int],
                   gpu_state_slots: Sequence[int] = ()) -> TaskHandle:
        """Launch the load for a match (GIL released)."""
        return TaskHandle(
            self._engine.submit_get(match.match_id,
                                    [int(s) for s in gpu_slots],
                                    [int(s) for s in gpu_state_slots]))

    # ---- task state ----

    def try_wait(
            self, handles: Sequence[TaskHandle]
    ) -> list[tuple[TaskHandle, str, str]]:
        """Non-blocking poll: [(handle, state, error)]."""
        ids = [h.task_id for h in handles]
        return [(TaskHandle(i), s, e)
                for i, s, e in self._engine.try_wait(ids)]

    def wait(self, handles: Sequence[TaskHandle], timeout_ms: float = 1000.0
             ) -> list[tuple[TaskHandle, str, str]]:
        """Blocking wait (GIL released while blocked in C++)."""
        ids = [h.task_id for h in handles]
        return [(TaskHandle(i), s, e)
                for i, s, e in self._engine.wait(ids, timeout_ms)]

    def cancel(self, handle: TaskHandle) -> bool:
        """Cancel a non-terminal task; False once terminal."""
        return self._engine.cancel(handle.task_id)

    # ---- lifecycle / introspection ----

    def completion_fd(self) -> int:
        """Pollable fd signalled on task completion (ADR 0004)."""
        return self._engine.completion_fd()

    def drain_completions(self) -> None:
        self._engine.drain_completions()

    def shutdown(self, grace_ms: float = 100.0) -> None:
        """Cancel outstanding tasks and drain (GIL released)."""
        self._engine.shutdown(grace_ms)

    def num_in_flight(self) -> int:
        return self._engine.num_in_flight()

    def num_free_blocks(self, group: int) -> int:
        return self._engine.num_free_blocks(group)

    def num_free_states(self, group: int) -> int:
        return self._engine.num_free_states(group)
