# ADR 0006 — Endpoint naming: registration vs control

- Status: Accepted
- Date: 2026-11-19

## Context

FlexKV's `KVManager` carries two endpoint names, `gpu_register_port` and
`server_recv_port`, inherited from the v0.1.0 client-server design. In the
audited v1.0.0 vLLM adapter they **alias the same string** (the config's
`server_recv_port` value is passed as `gpu_register_port`), which caused
real confusion during debugging: one socket file, two names, both
misleading. The names describe topology that no longer exists.

## Decision

1. StrataKV names endpoints by role, never by topology:
   - `registration_endpoint` — where GPU groups announce their memory
     (startup handshake, data-plane setup);
   - `control_endpoint` — where task commands flow (serving-time RPC,
     cross-process modes only).
2. Neither name implies client/server topology; both are configuration
   values of the runtime, documented per mode.
3. The FlexKV terms are banned in new code and in bindings.

## Consequences

- Configuration review can reason about lifetime (registration is
  startup-scoped; control is serving-scoped) directly from names.
- Cross-mode (in-process vs isolated) configs differ only in whether the
  endpoints are instantiated, not in what they are called.
