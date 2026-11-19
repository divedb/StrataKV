"""StrataKV connector for mainline-era vLLM (>= 0.17.2 connector API).

Version-isolated adapter (roadmap F1; ``design.md`` Phase F). Maps the
vLLM ``KVConnectorBase_V1`` hook set onto the native Python facade:

- **Worker role**: ``register_kv_caches`` only — parse per-group GPU
  tensors, build the :class:`stratakv.Engine`, register every group
  (raw descriptors; torch stays at this boundary by design), start.
  All per-forward hooks are no-ops: the native data plane moves KV
  without vLLM involvement.
- **Scheduler role**: native match/submit/try_wait per step.
  ``build_connector_meta`` returns empty metadata (the native engine needs
  no worker-side metadata); ``update_connector_output`` is the single
  scheduler-side poll; worker ``get_finished`` also polls so empty
  async-load steps make progress — polling is the engine's launch point
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

try:  # Real vLLM (>= 0.17; verified against 0.34.0).
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        KVConnectorBase_V1 as _VllmBase)
    _HAS_VLLM = True
except ImportError:  # Fake-harness / CPU-CI path: standalone base.
    class _VllmBase:
        pass

    _HAS_VLLM = False

__all__ = ["StrataKVConnectorV1", "ConnectorConfig", "HybridSidecarMixin",
           "HAS_VLLM"]


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

    @classmethod
    def from_dict(cls, spec: dict[str, Any]) -> "ConnectorConfig":
        """Build from the JSON-serializable dict carried in
        KVTransferConfig.kv_connector_extra_config (see
        StrataKVConnectorV1.layout_extra_config)."""
        groups = []
        for group in spec["layout"]:
            kind = group.get("kind", "full_attn")
            if kind == "linear_state":
                groups.append(LinearState(
                    name=group["name"],
                    layer_indices=list(group["layer_indices"]),
                    state_bytes=group["state_bytes"]))
            elif kind == "sliding_window":
                groups.append(SlidingWindow(
                    name=group["name"],
                    layer_indices=list(group["layer_indices"]),
                    num_kv_heads=group["num_kv_heads"],
                    head_size=group["head_size"],
                    window_blocks=group.get("window_blocks", 8),
                    tokens_per_block=group.get("tokens_per_block", 16)))
            elif kind == "mla":
                groups.append(Mla(
                    name=group["name"],
                    layer_indices=list(group["layer_indices"]),
                    kv_lora_rank=group["kv_lora_rank"],
                    rope_dim=group["rope_dim"],
                    tokens_per_block=group.get("tokens_per_block", 16)))
            else:
                groups.append(FullAttn(
                    name=group["name"],
                    layer_indices=list(group["layer_indices"]),
                    num_kv_heads=group["num_kv_heads"],
                    head_size=group["head_size"],
                    tokens_per_block=group.get("tokens_per_block", 16)))
        return cls(
            layout=groups,
            kv_cache_keys=spec.get("kv_cache_keys"),
            blocks_per_group=spec.get("blocks_per_group", 1 << 16),
            state_objects_per_group=spec.get(
                "state_objects_per_group", 256),
            cpu_backing=spec.get("cpu_backing", "pinned"),
            transport_streams=spec.get("transport_streams", 2),
            use_gpu_transport=spec.get("use_gpu_transport"))

    @classmethod
    def from_vllm(cls, vllm_config: Any,
                  kv_cache_config: Any = None) -> "ConnectorConfig":
        """Derive the adapter config from a real VllmConfig.

        Priority: kv_connector_extra_config (explicit layout dict) >
        kv_cache_config (authoritative: per-group specs + exact
        layer_names + num_blocks) > HF-config heuristic.
        """
        extra = {}
        kv_transfer = getattr(vllm_config, "kv_transfer_config", None)
        if kv_transfer is not None:
            extra = dict(getattr(kv_transfer, "kv_connector_extra_config",
                                 None) or {})
        if "layout" in extra:
            return cls.from_dict(extra)
        if kv_cache_config is not None and getattr(
                kv_cache_config, "kv_cache_groups", None):
            groups = []
            key_map: dict[str, Any] = {}
            for index, group in enumerate(
                    kv_cache_config.kv_cache_groups):
                spec = group.kv_cache_spec
                kind = type(spec).__name__
                name = f"group{index}"
                tpb = int(getattr(spec, "block_size", 16) or 16)
                if "Full" in kind:
                    groups.append(FullAttn(
                        name=name, layer_indices=list(range(
                            len(group.layer_names))),
                        num_kv_heads=int(getattr(spec, "num_kv_heads", 1)),
                        head_size=int(getattr(spec, "head_size", 1)),
                        tokens_per_block=tpb))
                elif "Sliding" in kind or "Local" in kind:
                    groups.append(SlidingWindow(
                        name=name, layer_indices=list(range(
                            len(group.layer_names))),
                        num_kv_heads=int(getattr(spec, "num_kv_heads", 1)),
                        head_size=int(getattr(spec, "head_size", 1)),
                        window_blocks=int(getattr(spec, "sliding_window",
                                                  8) or 8),
                        tokens_per_block=tpb))
                elif "MLA" in kind:
                    groups.append(Mla(
                        name=name, layer_indices=list(range(
                            len(group.layer_names))),
                        kv_lora_rank=int(getattr(spec, "kv_lora_rank", 1)),
                        rope_dim=int(getattr(spec, "rope_dim", 0)),
                        tokens_per_block=tpb))
                elif "Mamba" in kind or "HiddenState" in kind:
                    groups.append(LinearState(
                        name=name, layer_indices=list(range(
                            len(group.layer_names))),
                        state_bytes=int(getattr(spec, "num_bytes",
                                                getattr(spec, "block_size",
                                                        1 << 16)))))
                else:
                    raise ConnectorError(
                        f"unmapped vLLM kv cache spec {kind} for group "
                        f"{name} ({group.layer_names})")
                key_map[name] = list(group.layer_names)
            extra["layout"] = [
                {"kind": _group_kind(g), "name": g.name,
                 "layer_indices": list(g.layer_indices),
                 "num_kv_heads": getattr(g, "num_kv_heads", 0),
                 "head_size": getattr(g, "head_size", 0),
                 "kv_lora_rank": getattr(g, "kv_lora_rank", 0),
                 "rope_dim": getattr(g, "rope_dim", 0),
                 "window_blocks": getattr(g, "window_blocks", 0),
                 "state_bytes": getattr(g, "state_bytes", 0),
                 "tokens_per_block": getattr(g, "tokens_per_block", 0)
                 or 16}
                for g in groups]
            extra["kv_cache_keys"] = key_map
            extra.setdefault("blocks_per_group",
                             int(kv_cache_config.num_blocks))
            config = cls.from_dict(extra)
            config.capacities = {
                f"group{index}": int(kv_cache_config.num_blocks)
                for index in range(len(kv_cache_config.kv_cache_groups))}
            return config
        hf = vllm_config.model_config.hf_config
        num_layers = getattr(hf, "num_hidden_layers", None)
        num_kv_heads = getattr(hf, "num_key_value_heads",
                               getattr(hf, "num_attention_heads"))
        head_size = getattr(hf, "head_dim", None)
        if head_size is None:
            head_size = hf.hidden_size // hf.num_attention_heads
        block_size = (getattr(vllm_config.cache_config, "block_size",
                              None) or 16)
        spec = cls.layout_spec(
            num_layers=num_layers, num_kv_heads=num_kv_heads,
            head_size=head_size, tokens_per_block=block_size)
        spec.update({k: v for k, v in extra.items()
                     if k != "layout"})
        return cls.from_dict(spec)

    @staticmethod
    def layout_spec(num_layers: int, num_kv_heads: int, head_size: int,
                    tokens_per_block: int = 16) -> dict[str, Any]:
        """Single FullAttn group spec dict covering every layer."""
        return {
            "layout": [{
                "kind": "full_attn", "name": "full_attn",
                "layer_indices": list(range(num_layers)),
                "num_kv_heads": num_kv_heads, "head_size": head_size,
                "tokens_per_block": tokens_per_block,
            }],
        }

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
        # Per-group GPU slot capacity override: from kv_cache_config
        # (shape[0] of vLLM tensors is token slots, NOT blocks).
        self.capacities: dict[str, int] = {}


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
        # Integration-test observability.
        self.connector_puts = 0
        self.connector_gets = 0
        self.connector_match_hits = 0
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
        capacity = self.config.capacities.get(name,
                                                tensors[0].shape[0])
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


class StrataKVConnectorV1(HybridSidecarMixin, _VllmBase):
    """Thin, version-isolated adapter (design.md Phase F).

    One instance per (process, role). When real vLLM is installed the
    class subclasses KVConnectorBase_V1 and vLLM constructs it as
    ``(vllm_config, role, kv_cache_config)`` — the adapter config is
    then derived from VllmConfig (geometry from the HF config; see
    ConnectorConfig.from_vllm) or supplied via
    ``kv_connector_extra_config``. Without vLLM (fake harness, CPU CI)
    the standalone base is used and ``config=`` is passed explicitly.

    The worker-role instance owns engine construction; scheduler-role
    instances attach to the shared engine (single-process topology).
    """

    def __init__(self, vllm_config: Any = None, role: Any = "scheduler",
                 kv_cache_config: Any = None, *,
                 config: ConnectorConfig | None = None) -> None:
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        # Base's `role` is a read-only property in 0.34 — keep ours
        # under an adapter-local name.
        self.stratakv_role = getattr(role, "name", str(role)).lower()
        if _HAS_VLLM and vllm_config is not None:
            super().__init__(vllm_config, role, kv_cache_config)
        derived_from_vllm = False
        if config is None and vllm_config is not None:
            config = ConnectorConfig.from_vllm(vllm_config,
                                               kv_cache_config)
            derived_from_vllm = True
        global _SHARED, _SHARED_CONFIG
        if config is not None:
            if _SHARED is not None:
                if not derived_from_vllm:
                    raise ConnectorError(
                        "shared engine already constructed")
                # Real vLLM constructs one connector per role from the
                # SAME VllmConfig: the second instance attaches to the
                # existing engine (config identical by construction).
            else:
                _SHARED_CONFIG = config
                _SHARED = _SharedEngine(config)
                _SHARED.build_engine()
        elif _SHARED is None:
            raise ConnectorError(
                "first StrataKVConnectorV1 needs config=ConnectorConfig"
                "(...) or a real vllm_config — normally the worker-role "
                "instance")
        self._shared = _shared()

    @staticmethod
    def layout_extra_config(**geometry: Any) -> dict[str, Any]:
        """kv_connector_extra_config dict for a single FullAttn group
        (pass num_layers/num_kv_heads/head_size/tokens_per_block)."""
        return ConnectorConfig.layout_spec(**geometry)
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
        import os as _os
        if _os.environ.get("STRATAKV_DEBUG_REG"):
            import sys as _sys
            kcc = getattr(self, "kv_cache_config", None)
            for key, tensor in list(kv_caches.items())[:1]:
                print(f"[stratakv-reg] {key}: shape={tensor.shape} "
                      f"dtype={getattr(tensor, 'dtype', '?')} "
                      f"es={tensor.element_size()}", file=_sys.stderr)
            if kcc is not None:
                print(f"[stratakv-reg] num_blocks={kcc.num_blocks} "
                      f"layout={getattr(kcc, 'kv_cache_layout', '?')}",
                      file=_sys.stderr)
                for i, g in enumerate(kcc.kv_cache_groups):
                    spec = g.kv_cache_spec
                    print(f"[stratakv-reg] group{i} kind="
                          f"{type(spec).__name__} block_size="
                          f"{getattr(spec, 'block_size', '?')} "
                          f"page_padded="
                          f"{getattr(spec, 'page_size_padded', '?')} "
                          f"heads={getattr(spec, 'num_kv_heads', '?')} "
                          f"head_size={getattr(spec, 'head_size', '?')} "
                          f"head_size_v="
                          f"{getattr(spec, 'head_size_v', '?')} "
                          f"stride_align="
                          f"{getattr(spec, 'block_stride_alignment', '?')}",
                          file=_sys.stderr)
        consumed: set[str] = set()
        resolved: list[tuple[Any, list[Any]]] = []
        for group in shared.config.layout:
            keys = shared.config.kv_cache_keys.get(group.name)
            if keys is None:
                prefix = group.name
                keys = sorted(k for k in kv_caches
                              if k.startswith(prefix) and k not in consumed)
                if not keys and len(shared.config.layout) == 1:
                    # Real vLLM key names don't carry our group prefix —
                    # a single-group layout takes every tensor.
                    keys = sorted(k for k in kv_caches
                                  if k not in consumed)
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
    ) -> tuple[set[str] | None, set[str] | None]:
        """0.34 shape: always a (sending, recving) tuple, None-empty."""
        # Empty async-load steps have no model output to trigger the
        # scheduler poll. Launch and reap transfers here as well.
        self.update_connector_output(None)
        shared = self._shared
        with shared.lock:
            sending = set(shared.finished_sending)
            recving = set(shared.finished_recving)
            shared.finished_sending -= sending
            shared.finished_recving -= recving
        return (sending or None, recving or None)

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
        shared.connector_match_hits += 1
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
            if num_external_tokens == 0:
                # vLLM 0.34 calls this hook for every scheduled
                # request; zero external tokens = nothing to load.
                return
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
        shared.connector_gets += 1
        shared.matches.pop(request.request_id, None)
        shared.pending_gets[request.request_id] = handle

    def build_connector_meta(self, scheduler_output: Any) -> Any:
        """Engine launch point — empty metadata by design.

        The native engine flushes its scheduler batch on the next
        poll (``update_connector_output``): N submits per step, one
        flush per step (design.md diagram 2). With real vLLM the base
        class requires a KVConnectorMetadata return; an empty one is
        correct (the native engine needs no worker-side metadata).
        """
        if _HAS_VLLM:
            from vllm.distributed.kv_transfer.kv_connector.v1.base import (
                KVConnectorMetadata)
            return KVConnectorMetadata()
        return None

    def update_connector_output(self, connector_output: Any) -> None:
        """One native poll per step: launches pending batches and
        moves completed tasks into the finished sets."""
        shared = self._shared
        handles = list(shared.pending_gets.values()) + list(
            shared.pending_puts.values())
        if not handles:
            return
        items = shared.engine.try_wait(handles)
        results = {}
        for handle, state, error in items:
            if state == "failed":
                raise ConnectorError(
                    f"native KV transfer {handle.task_id} failed: {error}")
            results[handle.task_id] = state
        for request_id, handle in list(shared.pending_gets.items()):
            if results.get(handle.task_id) == "succeeded":
                shared.pending_gets.pop(request_id)
                shared.engine.forget(handle)
                with shared.lock:
                    shared.finished_recving.add(request_id)
        for request_id, handle in list(shared.pending_puts.items()):
            if results.get(handle.task_id) == "succeeded":
                shared.pending_puts.pop(request_id)
                shared.engine.forget(handle)
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
        import os as _os
        if _os.environ.get("STRATAKV_DEBUG_PUT"):
            import sys as _sys
            print(f"[stratakv-finish] req={request.request_id} "
                  f"blocks={block_ids if not hasattr(block_ids, 'blocks') else 'KVCB'}",
                  file=_sys.stderr)
        slots_by_group = self._group_slots(block_ids)
        token_groups = shared._token_groups
        all_ids = (slots_by_group.get(token_groups[0], [])
                   if len(token_groups) == 1 else [])
        # vLLM 0.34 hands request_finished only the blocks still held
        # (it may already have freed some) — stage only the tokens
        # whose KV actually exists, and store the newest blocks.
        tpb = self._tokens_per_block(shared)
        max_tokens = len(all_ids) * tpb
        tokens = list(request.all_token_ids)
        if len(token_groups) == 1 and max_tokens and \
                len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
        plan = shared.engine.stage_put(tokens)
        if plan.new_blocks == 0 and plan.num_states == 0:
            # Nothing new to store (fully matched earlier).
            return (False, None)
        if len(token_groups) == 1:
            # Skip the cached prefix, then take complete new blocks.
            # The final allocated block can contain an incomplete tail.
            gpu_blocks = all_ids[plan.matched_blocks:
                                 plan.matched_blocks + plan.new_blocks]
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
        import os as _os
        if _os.environ.get("STRATAKV_DEBUG_PUT"):
            import sys as _sys
            print(f"[stratakv-put] req={request.request_id} "
                  f"all={len(slots_by_group.get(token_groups[0], []))} "
                  f"matched={plan.matched_blocks} "
                  f"new={plan.new_blocks} "
                  f"gpu_blocks={len(gpu_blocks)}",
                  file=_sys.stderr)
        handle = shared.engine.submit_put(plan, gpu_blocks,
                                          gpu_state_slots)
        shared.connector_puts += 1
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
        """Normalize vLLM's block argument into per-group id lists:
        dict (harness), flat list (0.17-era request_finished), or a
        real 0.34 KVCacheBlocks (per-group block objects)."""
        if isinstance(blocks, dict):
            return {group: list(ids) for group, ids in blocks.items()}
        block_seqs = getattr(blocks, "blocks", None)
        if block_seqs is not None:
            # KVCacheBlocks: tuple[Sequence[KVCacheBlock], ...] in
            # kv-cache group order — map onto token groups in order.
            shared = self._shared
            ids_per_group = [
                [b.block_id for b in seq] for seq in block_seqs]
            if len(ids_per_group) != len(shared._token_groups):
                raise ConnectorError(
                    f"vLLM allocated {len(ids_per_group)} block groups "
                    f"for {len(shared._token_groups)} token groups")
            return {name: ids for name, ids
                    in zip(shared._token_groups, ids_per_group)}
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
