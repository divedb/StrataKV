# vLLM integration (Phase F — not yet implemented)

This directory will hold the thin, version-isolated Python connector
(`StrataKVConnectorV1`) per `design.md`:

- first target: mainline-era vLLM (≥ 0.17.2 connector API, E2
  environment); 0.10.1.1 only if parity with the audited E0 environment
  is required;
- worker role: multi-group GPU registration only (all per-forward hooks
  are no-ops by design — the native data plane moves KV without vLLM
  involvement);
- scheduler role: native match/submit/try_wait mapped onto the connector
  hooks; `build_connector_meta` returns empty metadata;
- hybrid support via `HybridSidecarMixin` under ADR 0003.

Nothing here yet — by the no-silent-stubs rule this README is the entire
contents until Phase F lands.
