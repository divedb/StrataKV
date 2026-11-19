# Documentation

- [API and ownership](api.md), including metrics and Prometheus export
- [vLLM integration](../integrations/vllm/README.md)
- [Benchmarking](../benchmarks/README.md) and [measurement policy](../roadmap.md#5-benchmark-policy-all-phases)
- [Hardening and release checks](hardening.md)
- [Toolchain requirements](toolchains.md)

Architecture decisions:

- ADR 0001, adopt versus build: pending Phase A baseline ladder
- [0002 — Transfer isolation](adr/0002-transfer-isolation-mode.md)
- [0003 — Hybrid sidecars and no partial caching](adr/0003-hybrid-sidecars-and-no-partial-caching.md)
- [0004 — GIL and dispatch](adr/0004-gil-and-thread-dispatch.md)
- [0005 — GPU ownership and IPC lifetime](adr/0005-gpu-memory-ownership-and-ipc-lifetime.md)
- [0006 — Endpoint naming](adr/0006-endpoint-naming.md)
