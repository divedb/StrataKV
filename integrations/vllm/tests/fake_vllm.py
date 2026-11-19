"""Fake-vLLM harness for the StrataKV connector (Phase F1/F3 tests).

Stands in for the vLLM pieces the connector touches — KVConnectorBase_V1
hook call sites, request objects, and torch tensors — WITHOUT vLLM or
torch installed. Tensors are duck-typed (``data_ptr``/``shape``/
``element_size``/``numel``) over memory from the facade's debug device
helpers on GPU builds; on CPU builds (simulation mode) they wrap plain
bytearrays and no bytes are expected to move.

The harness models the single-process vLLM V1 topology
(``VLLM_ENABLE_V1_MULTIPROCESSING=0``): one scheduler step loop calling
the connector hooks in vLLM's order, backed by "worker" GPU buffers the
test fills and verifies.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
for path in (os.path.join(REPO_ROOT, "python"),
             os.path.join(REPO_ROOT, "integrations", "vllm", "v0_17"),
             os.path.dirname(os.path.abspath(__file__))):
    if path not in sys.path:
        sys.path.insert(0, path)

import stratakv  # noqa: E402

GPU = stratakv.GPU_TRANSPORT_AVAILABLE


def _device_alloc(bytes_count: int) -> Any:
    """GPU builds: real device memory. CPU builds: a bytearray."""
    if GPU:
        return ("gpu", stratakv.debug_alloc_device(bytes_count))
    return ("cpu", bytearray(bytes_count))


def _device_free(handle: Any) -> None:
    if GPU and handle[0] == "gpu":
        stratakv.debug_free_device(handle[1])


class FakeTensor:
    """Duck-typed stand-in for a torch KV-cache tensor (fp16)."""

    def __init__(self, shape: tuple[int, ...], fill: int | None = None):
        assert len(shape) >= 2
        self.shape = shape
        self.element = 2  # fp16 bytes
        self.bytes_count = 1
        for dim in shape:
            self.bytes_count *= dim
        self.bytes_count *= self.element
        self.handle = _device_alloc(self.bytes_count)
        self._fill = fill

    # torch-ish surface used by the connector + tests.
    def data_ptr(self) -> int:
        if self.handle[0] == "gpu":
            return self.handle[1]
        return id(self.handle[1])  # Simulation: address never deref'd.

    def element_size(self) -> int:
        return self.element

    def numel(self) -> int:
        total = 1
        for dim in self.shape:
            total *= dim
        return total

    # Test-side fill/verify (offset-addressed for per-block access).
    def write_at(self, offset: int, data: bytes) -> None:
        assert offset + len(data) <= self.bytes_count
        if self.handle[0] == "gpu":
            existing = self.read()
            stratakv.debug_write_device(
                self.handle[1], existing[:offset] + data +
                existing[offset + len(data):])
        else:
            self.handle[1][offset:offset + len(data)] = data

    def read_at(self, offset: int, size: int) -> bytes:
        assert offset + size <= self.bytes_count
        if self.handle[0] == "gpu":
            return stratakv.debug_read_device(
                self.handle[1] + offset, size)
        return bytes(self.handle[1][offset:offset + size])

    def write(self, data: bytes) -> None:
        assert len(data) == self.bytes_count
        if self.handle[0] == "gpu":
            stratakv.debug_write_device(self.handle[1], data)
        else:
            self.handle[1] = bytearray(data)

    def read(self) -> bytes:
        if self.handle[0] == "gpu":
            return stratakv.debug_read_device(self.handle[1],
                                              self.bytes_count)
        return bytes(self.handle[1])

    def free(self) -> None:
        _device_free(self.handle)


class FakeRequest:
    """vLLM-ish request: id + full token sequence."""

    def __init__(self, request_id: str, tokens: list[int]):
        self.request_id = request_id
        self.all_token_ids = list(tokens)
        self.num_computed_tokens = 0


class FakeSchedulerStep:
    """Drives one vLLM scheduler step's connector hook sequence."""

    @staticmethod
    def run(connector: Any, requests: list[FakeRequest],
            block_tables: dict[str, Any]) -> None:
        # 1. Matching for (re)scheduled requests.
        loads: dict[str, tuple[int, bool]] = {}
        for request in requests:
            loads[request.request_id] = (
                connector.get_num_new_matched_tokens(
                    request, request.num_computed_tokens))
        # 2. Allocation happened (caller-provided block tables);
        #    submit external loads.
        for request in requests:
            matched_tokens, async_load = loads[request.request_id]
            if matched_tokens > 0 and async_load:
                connector.update_state_after_alloc(
                    request, block_tables[request.request_id],
                    matched_tokens)
                request.num_computed_tokens = matched_tokens
        # 3. build_connector_meta: engine launch point (empty meta).
        connector.build_connector_meta(None)
        # 4. Poll completions once (the single native call per step).
        connector.update_connector_output(None)


def pump(scheduler_connector: Any, worker_connector: Any,
         request_id: str, *, sending: bool = True,
         timeout_ms: float = 3000.0) -> bool:
    """Run scheduler steps until `request_id` shows up in the finished
    set (transfers are async: completions arrive on later steps, as in
    real vLLM)."""
    deadline = time.monotonic() + timeout_ms / 1000.0
    while time.monotonic() < deadline:
        scheduler_connector.update_connector_output(None)
        sending_done, recving_done = worker_connector.get_finished(set())
        done = sending_done if sending else recving_done
        if done and request_id in done:
            return True
        time.sleep(0.0005)
    return False
