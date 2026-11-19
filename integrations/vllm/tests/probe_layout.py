"""F4 environment probe: dump the REAL vLLM 0.34 hybrid layout.

Boots a plain LLM with a no-op probe connector and prints, per model:
- kv_cache_config: num_blocks, layout, hash block size, alignment,
  per-group spec attributes (authoritative geometry);
- register_kv_caches tensor table (keys, shapes, dtypes, strides).

Run on the GPU host (see integrations/vllm/README.md):
    python3 probe_layout.py /path/to/model [mamba_cache_mode]
"""

from __future__ import annotations

import os
import sys

# Must precede vllm import.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

from vllm import LLM, SamplingParams  # noqa: E402
from vllm.config import KVTransferConfig  # noqa: E402
from vllm.distributed.kv_transfer.kv_connector.v1.base import (  # noqa: E402
    KVConnectorBase_V1, KVConnectorMetadata, SupportsHMA)

SPEC_ATTRS = ("num_kv_heads", "head_size", "head_size_v", "sliding_window",
              "extra_retained_tokens", "kv_lora_rank", "rope_dim",
              "num_bytes", "shapes", "dtypes", "mamba_type",
              "mamba_cache_mode", "tokens_per_state", "num_heads",
              "tp_replicated", "page_size_bytes", "page_size_padded",
              "prefix_cacheable")


class ProbeConnector(KVConnectorBase_V1, SupportsHMA):
    def __init__(self, vllm_config, role, kv_cache_config):
        super().__init__(vllm_config, role, kv_cache_config)
        self.probe_role = getattr(role, "name", str(role))
        if self.probe_role != "SCHEDULER":
            return
        kcc = kv_cache_config
        print(f"[probe] num_blocks={kcc.num_blocks} "
              f"layout={getattr(kcc, 'kv_cache_layout', None)} "
              f"hash_block_size={getattr(kcc, 'hash_block_size', None)} "
              f"alignment={getattr(kcc, 'cache_hit_alignment_tokens', None)}")
        print(f"[probe] transfer_group_ids={kcc.transfer_group_ids} "
              f"has_mamba={kcc.has_mamba_layers}")
        for i, group in enumerate(kcc.kv_cache_groups):
            spec = group.kv_cache_spec
            attrs = " ".join(
                f"{a}={getattr(spec, a)}" for a in SPEC_ATTRS
                if hasattr(spec, a))
            print(f"[probe] group{i} spec={type(spec).__name__} "
                  f"block_size={spec.block_size} "
                  f"layers={len(group.layer_names)} "
                  f"first={group.layer_names[:2]} "
                  f"last={group.layer_names[-1:]} "
                  f"xfer={group.enable_kv_transfer}")
            print(f"[probe]   {attrs}")
        sys.stdout.flush()

    # ---- worker ----
    def register_kv_caches(self, kv_caches):
        for key, tensor in kv_caches.items():
            print(f"[probe-kv] {key}: shape={tuple(tensor.shape)} "
                  f"dtype={tensor.dtype} stride={tuple(tensor.stride())}")
        print(f"[probe-kv] total_keys={len(kv_caches)}")
        sys.stdout.flush()

    def start_load_kv(self, *a, **k):
        pass

    def wait_for_layer_load(self, layer_name):
        pass

    def save_kv_layer(self, *a, **k):
        pass

    def wait_for_save(self):
        pass

    def get_finished(self, finished_req_ids):
        self.update_connector_output(None)
        return (None, None)

    # ---- scheduler ----
    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        return (0, False)

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        print(f"[probe-alloc] req={request.request_id} "
              f"num_external_tokens={num_external_tokens} "
              f"blocks={blocks}")

    def build_connector_meta(self, scheduler_output):
        return KVConnectorMetadata()

    def update_connector_output(self, connector_output):
        pass

    def request_finished_all_groups(self, request, block_ids):
        print(f"[probe-finish] req={request.request_id} "
              f"n_tokens={len(request.all_token_ids)} "
              f"block_ids={block_ids}")
        return (False, None)


def main() -> None:
    model = sys.argv[1]
    mamba_mode = sys.argv[2] if len(sys.argv) > 2 else None
    kwargs = {}
    if mamba_mode:
        kwargs["mamba_cache_mode"] = mamba_mode
    kv_config = KVTransferConfig(
        kv_connector="ProbeConnector",
        kv_connector_module_path="probe_layout",
        kv_role="kv_both",
    )
    llm = LLM(model=model, gpu_memory_utilization=0.85, max_model_len=1024,
              enforce_eager=True, enable_prefix_caching=False,
              kv_transfer_config=kv_config, **kwargs)
    out = llm.generate(["Hello, world."],
                       SamplingParams(temperature=0.0, max_tokens=4))
    print(f"[probe] generated: {out[0].outputs[0].text!r}")


if __name__ == "__main__":
    main()
