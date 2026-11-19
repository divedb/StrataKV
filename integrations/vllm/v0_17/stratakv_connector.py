"""StrataKV connector for mainline-era vLLM (>= 0.17.2 connector API).

Version-isolated adapter (roadmap F1; ``design.md`` Phase F). Maps the
vLLM ``KVConnectorBase_V1`` hook set onto the native Python facade:

- **Worker role**: ``register_kv_caches`` only — parse per-group GPU
  tensors, build the :class:`stratakv.Engine`, register every group
  (raw descriptors; torch stays at this boundary by design), start.
  All per-forward hooks are no-ops: the native data plane moves KV
  without vLLM involvement.
- **Scheduler role**: native match/submit/try_wait per step.
  ``build_connector_meta`` returns ``None`` (the native engine needs
  no worker-side metadata); ``update_connector_output`` is the single
  per-step poll — which is also the engine's launch point
  (``ITaskExecutor::Flush`` fires on every Wait/TryWait).

Topology (v1): **single-process vLLM** (``VLLM_ENABLE_V1_MULTIPROCESSING=0``
debug topology, or any runner keeping scheduler+worker in one
process). Both roles share one native engine held in a module-level
singleton — the worker registers GPU groups, the scheduler role
requires them (loudly, per the no-silent-stubs rule). The
cross-process registration-endpoint topology (ADR 0005/0006) is
deliberately not guessed at here; it lands with the E2 integration
environment.

Hybrid (F3, ADR 0003): multi-group layouts with LinearState sidecars
register a state pool per LinearState group; snapshot/restore ride
``gpu_state_slots`` (``HybridSidecarMixin`` below). The
no-partial-caching invariant is enforced by the metadata plane — the
connector only surfaces truncated matches.

Torch is NOT imported: ``register_kv_caches`` accepts anything
tensor-shaped (``data_ptr()``, ``shape``, ``element_size()``), which is
exactly what the test harness substitutes for torch tensors.
"""

from __future__ import annotations

import threading
from typing import Any, Sequence

import stratakv
from stratakv import (Engine, EngineConfig, FullAttn, LinearState, Mla,
                      SlidingWindow)

__all__ = ["StrataKVConnectorV1", "ConnectorConfig", "HybridSidecarMixin"]


class ConnectorError(RuntimeError):
    """Loud adapter failure (misordered roles, missing registration)."""


class ConnectorConfig:
    """Adapter-side configuration (vLLM side passes its own config).

    ``layout``: the model's heterogeneous layout (facade group objects
    — FullAttn/Mla/SlidingWindow/LinearState). Required: vLLM tensor
    shapes alone do not encode geometry reliably across versions.
    ``kv_cache_keys``: layout-group name -> key(s) in the
    ``register_kv_caches`` dict (str for one flat/BLOCKWISE tensor;
    list[str] for per-layer LAYERWISE tensors). Omitted groups are
    matched heuristically by layer-name prefix.
    """

    def __init__(self, layout: Sequence[Any], *,
                 kv_cache_keys: dict[str, Any] | None = None,
                 blocks_per_group: int = 1 << 16,
                 state_objects_per_group: int = 256,
                 cpu_backing: str = "pinned",
                 transport_streams: int = 2,
                 use_gpu_transport: bool | None = None) -> None:
        self.layout = list(layout)
        self.kv_cache_keys = dict(kv_cache_keys or {})
        self.blocks_per_group = blocks_per_group
        self.state_objects_per_group = state_objects_per_group
        self.cpu_backing = cpu_backing
        self.transport_streams = transport_streams
        # None = auto: real transport when the native module supports
        # it (simulation otherwise, loudly flagged).
        self.use_gpu_transport = use_gpu_transport


def _group_kind(group: Any) -> str:
    return getattr(group, "kind", "full_attn")


class HybridSidecarMixin:
    """State-sidecar helpers (design.md Phase F; ADR 0003).

    Snapshot/restore themselves ride the normal submit_put/submit_get
    ``gpu_state_slots`` argument — these helpers only own the slot
    assignment bookkeeping (v1: one slot per request, freed when its
    put completes).
    """

    def assign_state_slot(self, request_id: str) -> int:
        return self._shared.assign_state_slot(request_id)  # type: ignore[attr-defined]

    def release_state_slot(self, request_id: str) -> None:
        self._shared.release_state_slot(request_id)  # type: ignore[attr-defined]


class _SharedEngine:
    """Process-wide native engine + adapter bookkeeping (both roles)."""

    def __init__(self, config: ConnectorConfig) -> None:
        self.config = config
        self.engine: Engine | None = None
        self.started = False
        self.simulation = False
        self.lock = threading.Lock()
        # request_id -> state
        self.matches: dict[str, Any] = {}
        self.pending_gets: dict[str, Any] = {}
        self.pending_puts: dict[str, Any] = {}
        self.state_slots: dict[str, int] = {}
        self.free_state_slots: list[int] = []
        self.finished_sending: set[str] = set()
        self.finished_recving: set[str] = set()
        self._group_index: dict[str, int] = {}
        self._token_groups: list[str] = []
        self._state_groups: list[str] = []

    # ---- construction / registration (worker role) ----

    def build_engine(self) -> None:
        if self.engine is not None:
            return
        use_gpu = self.config.use_gpu_transport
        if use_gpu is None:
            use_gpu = stratakv.GPU_TRANSPORT_AVAILABLE
        if not use_gpu:
            self.simulation = True  # InlineExecutor: no bytes move.
        cfg = EngineConfig(
            blocks_per_group=self.config.blocks_per_group,
            state_objects_per_group=self.config.state_objects_per_group,
            use_gpu_transport=use_gpu,
            cpu_backing=self.config.cpu_backing,
            transport_streams=self.config.transport_streams,
        )
        self.engine = Engine(self.config.layout, cfg)
        for index, group in enumerate(self.config.layout):
            self._group_index[group.name] = index
            if _group_kind(group) == "linear_state":
                self._state_groups.append(group.name)
            else:
                self._token_groups.append(group.name)
        self.free_state_slots = list(
            range(self.config.state_objects_per_group))

    def register_group(self, name: str, tensors: list[Any]) -> int:
        """Register one layout group's GPU tensors (LAYERWISE: one per
        layer; a single tensor is accepted as BLOCKWISE)."""
        if self.engine is None:
            raise ConnectorError("register before engine construction")
        buffers = []
        for tensor in tensors:
            per_layer_bytes = tensor.numel() * tensor.element_size()
            buffers.append((tensor.data_ptr(), per_layer_bytes))
        capacity = tensors[0].shape[0]
        layout_kind = "layerwise" if len(tensors) > 1 else "blockwise"
        return self.engine.register_gpu_group(
            name, self._group_index[name], buffers, capacity,
            layout=layout_kind)

    def start(self) -> None:
        if self.simulation:
            self.started = True
            return
        if self.engine is None:
            raise ConnectorError(
                "start() before register_kv_caches — the worker role "
                "must register GPU groups first (single-process "
                "topology: construct/enable the worker role connector "
                "before the scheduler role runs)")
        if not self.started:
            self.engine.start()
            self.started = True

    # ---- scheduler-role bookkeeping ----

    def assign_state_slot(self, request_id: str) -> int:
        if not self.free_state_slots:
            raise ConnectorError("state-slot pool exhausted")
        slot = self.free_state_slots.pop(0)
        self.state_slots[request_id] = slot
        return slot

    def release_state_slot(self, request_id: str) -> None:
        slot = self.state_slots.pop(request_id, None)
        if slot is not None:
            self.free_state_slots.append(slot)


_SHARED: _SharedEngine | None = None
_SHARED_CONFIG: ConnectorConfig | None = None


def _shared() -> _SharedEngine:
    global _SHARED
    if _SHARED is None:
        raise ConnectorError(
            "StrataKVConnectorV1(config=...) must construct the worker "
            "role first (it owns the layout + engine configuration)")
    return _SHARED


class StrataKVConnectorV1(HybridSidecarMixin):
    """Thin, version-isolated adapter (design.md Phase F).

    One instance per (process, role). The worker-role instance owns
    adapter construction (pass ``config=`` to it); scheduler-role
    instances attach to the shared engine.
    """

    def __init__(self, vllm_config: Any = None, role: str = "scheduler",
                 *, config: ConnectorConfig | None = None) -> None:
        self.vllm_config = vllm_config
        self.role = role
        global _SHARED, _SHARED_CONFIG
        if config is not None:
            if _SHARED is not None:
                raise ConnectorError("shared engine already constructed")
            _SHARED_CONFIG = config
            _SHARED = _SharedEngine(config)
            _SHARED.build_engine()
        elif _SHARED is None:
            raise ConnectorError(
                "first StrataKVConnectorV1 needs config=ConnectorConfig"
                "(...) — normally the worker-role instance")
        self._shared = _shared()
        if self._shared.simulation:
            import warnings
            warnings.warn(
                "StrataKV connector running in SIMULATION mode "
                "(InlineExecutor: no bytes move) — build the native "
                "module with STRATAKV_PYTHON=ON + STRATAKV_GPU=ON for "
                "the real data plane", stacklevel=2)

    # ---- worker role ----

    def register_kv_caches(self, kv_caches: dict[str, Any]) -> None:
        """Parse + register every layout group's GPU tensors, start.

        Group -> kv_caches key mapping: explicit ``kv_cache_keys`` if
        given, else layer-name-prefix heuristic (keys starting with the
        group name), else (single token group) all remaining keys as
        per-layer tensors.
        """
        shared = self._shared
        shared.build_engine()
        consumed: set[str] = set()
        resolved: list[tuple[Any, list[Any]]] = []
        for group in shared.config.layout:
            keys = shared.config.kv_cache_keys.get(group.name)
            if keys is None:
                prefix = group.name
                keys = sorted(k for k in kv_caches
                              if k.startswith(prefix) and k not in consumed)
            if isinstance(keys, str):
                keys = [keys]
            tensors = [kv_caches[k] for k in keys]
            if not tensors:
                raise ConnectorError(
                    f"no kv_caches entries for group '{group.name}' "
                    f"(have: {sorted(kv_caches)})")
            expected = len(group.layer_indices)
            if len(tensors) not in (1, expected):
                raise ConnectorError(
                    f"group '{group.name}': {len(tensors)} tensors for "
                    f"{expected} layers (pass 1 flat or {expected} "
                    f"per-layer tensors)")
            resolved.append((group, tensors))
            consumed.update(keys)
        if shared.simulation:
            # CPU-only module: nothing to register (the InlineExecutor
            # engine auto-started) — the validation above IS the test.
            shared.started = True
            return
        for group, tensors in resolved:
            shared.register_group(group.name, tensors)
        shared.start()

    # Per-forward hooks: no-ops by design (native data plane).
    def start_load_kv(self, forward_context: Any, **kwargs: Any) -> None:
        pass

    def wait_for_layer_load(self, layer_name: str) -> None:
        pass

    def save_kv_layer(self, name: str, kv_layer: Any,
                      attn_metadata: Any, **kwargs: Any) -> None:
        pass

    def wait_for_save(self) -> None:
        pass

    def get_finished(
            self, finished_req_ids: set[str]
    ) -> tuple[set[str], set[str]] | None:
        shared = self._shared
        with shared.lock:
            sending = shared.finished_sending - finished_req_ids
            recving = shared.finished_recving - finished_req_ids
            shared.finished_sending -= sending
            shared.finished_recving -= recving
        return (sending, recving) if (sending or recving) else None

    # ---- scheduler role ----

    def get_num_new_matched_tokens(
            self, request: Any, num_computed_tokens: int
    ) -> tuple[int, bool]:
        """Readiness-filtered prefix match (GIL held; sub-ms budget).

        Returns (matched_tokens, load_async). The metadata plane
        enforces ADR 0003's no-partial-caching invariant: the match
        truncates to the last COMPLETE bundle boundary (KV blocks AND
        sidecar states present) on its own — nothing to re-check here.
        """
        shared = self._shared
        tokens = list(request.all_token_ids)
        match = shared.engine.match(tokens)
        # Replace any stale match for this request (rescheduling).
        stale = shared.matches.pop(request.request_id, None)
        if stale is not None:
            shared.engine.release_match(stale)
        shared.matches[request.request_id] = match
        if match.matched_blocks == 0:
            shared.engine.release_match(match)
            shared.matches.pop(request.request_id, None)
            return (0, False)
        tpb = self._tokens_per_block(shared)
        return (match.matched_blocks * tpb, True)

    def update_state_after_alloc(
            self, request: Any, blocks: Any, num_external_tokens: int
    ) -> None:
        """Submit the load for the matched prefix.

        ``blocks`` contract (v1, single-process): a flat block-id list
        for single-token-group layouts, or ``{group_name: [ids]}`` for
        hybrid layouts. Ids are position-major within each group —
        exactly parallel to ``match.blocks`` (group-major overall).
        """
        shared = self._shared
        match = shared.matches.get(request.request_id)
        if match is None:
            raise ConnectorError(
                f"update_state_after_alloc without a live match for "
                f"{request.request_id}")
        slots_by_group = self._group_slots(blocks)
        name_of = {index: name for name, index
                   in shared._group_index.items()}
        # match.blocks is position-major / group-minor (one ref per
        # token group per matched position, group INDEX keyed) —
        # interleave the group's slot ids accordingly.
        cursor = {group: 0 for group in slots_by_group}
        gpu_slots = []
        for group_index, _block_id in match.blocks:
            group = name_of[group_index]
            ids = slots_by_group[group]
            gpu_slots.append(ids[cursor[group]])
            cursor[group] += 1
        gpu_state_slots = [
            shared.state_slots[request.request_id]
            for _ in match.states
        ] if match.states else []
        handle = shared.engine.submit_get(
            match, gpu_slots, gpu_state_slots)
        shared.matches.pop(request.request_id, None)
        shared.pending_gets[request.request_id] = handle

    def build_connector_meta(self, scheduler_output: Any) -> None:
        """Engine launch point — returns empty metadata by design.

        The native engine flushes its scheduler batch on the next
        poll (``update_connector_output``): N submits per step, one
        flush per step (design.md diagram 2).
        """
        return None

    def update_connector_output(self, connector_output: Any) -> None:
        """One native poll per step: launches pending batches and
        moves completed tasks into the finished sets."""
        shared = self._shared
        handles = list(shared.pending_gets.values()) + list(
            shared.pending_puts.values())
        if not handles:
            return
        results = dict((h.task_id, state)
                       for h, state, _error
                       in shared.engine.try_wait(handles))
        for request_id, handle in list(shared.pending_gets.items()):
            if results.get(handle.task_id) == "succeeded":
                shared.pending_gets.pop(request_id)
                with shared.lock:
                    shared.finished_recving.add(request_id)
        for request_id, handle in list(shared.pending_puts.items()):
            if results.get(handle.task_id) == "succeeded":
                shared.pending_puts.pop(request_id)
                shared.release_state_slot(request_id)
                with shared.lock:
                    shared.finished_sending.add(request_id)

    def request_finished(self, request: Any,
                         block_ids: Any) -> tuple[bool, dict | None]:
        """Offload the finished request's KV (+ sidecar states).

        Returns (True, None): block freeing stays delayed until the
        put completes (``get_finished`` reports it).
        """
        shared = self._shared
        tokens = list(request.all_token_ids)
        plan = shared.engine.stage_put(tokens)
        if plan.new_blocks == 0 and plan.num_states == 0:
            # Nothing new to store (fully matched earlier).
            return (False, None)
        slots_by_group = self._group_slots(block_ids)
        token_groups = shared._token_groups
        if len(token_groups) == 1:
            gpu_blocks = slots_by_group.get(token_groups[0],
                                            [])[:plan.new_blocks]
        else:
            # blocks_to_store is position-major / group-minor, same as
            # match refs — interleave per position (gpu_blocks must
            # parallel plan.blocks_to_store).
            gpu_blocks = []
            for position in range(plan.new_blocks):
                for group in token_groups:
                    gpu_blocks.append(slots_by_group[group][position])
        gpu_state_slots = [
            shared.assign_state_slot(request.request_id)
            for _ in range(plan.num_states)
        ] if plan.num_states else []
        handle = shared.engine.submit_put(plan, gpu_blocks,
                                          gpu_state_slots)
        shared.pending_puts[request.request_id] = handle
        return (True, None)

    def shutdown(self) -> None:
        shared = self._shared
        if shared.engine is not None:
            shared.engine.shutdown(2000.0)
        global _SHARED, _SHARED_CONFIG
        _SHARED = None
        _SHARED_CONFIG = None

    # ---- helpers ----

    def _tokens_per_block(self, shared: _SharedEngine) -> int:
        for group in shared.config.layout:
            tpb = getattr(group, "tokens_per_block", 0)
            if tpb:
                return tpb
        raise ConnectorError("layout has no token-addressable group")

    def _group_slots(self, blocks: Any) -> dict[str, list[int]]:
        """Normalize vLLM's block argument into per-group id lists."""
        if isinstance(blocks, dict):
            return {group: list(ids) for group, ids in blocks.items()}
        shared = self._shared
        if len(shared._token_groups) == 1:
            return {shared._token_groups[0]: list(blocks)}
        raise ConnectorError(
            "hybrid layouts must pass per-group block tables "
            "{group_name: [block_ids]} (vLLM hybrid KV allocator)")


def reset_shared_engine() -> None:
    """Test/teardown hook: drop the process-wide engine."""
    global _SHARED, _SHARED_CONFIG
    if _SHARED is not None and _SHARED.engine is not None:
        _SHARED.engine.shutdown(1000.0)
    _SHARED = None
    _SHARED_CONFIG = None
