# ADR 0005 — GPU memory ownership and IPC handle lifetime

- Status: Accepted
- Date: 2026-11-19

## Context

The engine (vLLM) allocates and owns the GPU KV tensors; attention
kernels read/write them every step. StrataKV must move KV in and out
without copying through the engine and without owning that memory. The
audited mechanism (FlexKV `TensorSharedHandle`, mirrored and validated in
`cuda_ipc_demo/`): export CUDA IPC handles at registration, map them in
the data-plane process, address blocks via strides.

Lifetime hazards validated in the demo: handles stay valid only while the
owning allocation lives; the owner must not free before the importer
opens; the importer must release with `cudaIpcCloseMemHandle`, never
`cudaFree`.

## Decision

1. `DeviceMemoryRegistry` is the sole gateway: registration takes explicit
   per-group descriptors (`BufferView`s + device id + layout strides),
   never torch pointers; it returns a `RegistrationId`.
2. **Ownership stays with the engine.** StrataKV never frees registered
   memory. `Unregister(registration_id)` invalidates all exported handles
   immediately and is the engine's responsibility at shutdown/realloc.
3. Registered buffers must remain valid until `Unregister`; the adapter
   contract states this (mirrors vLLM's `register_kv_caches` lifetime).
4. The core stays libtorch-free: handles are opaque 64-byte payloads
   (`IpcHandle`); no torch ABI linkage below the adapter.

## Consequences

- In-process mode needs no IPC at all but uses the same registry contract
  (ADR 0002), keeping one ownership story.
- Adapter tests must cover the shutdown ordering: drain tasks → unregister
  → engine frees.
