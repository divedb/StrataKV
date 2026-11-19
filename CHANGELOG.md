# Changelog

## Unreleased

- Add metadata match, scheduler queue residence, and host enqueue latency
  histograms to native/Python snapshots and Prometheus export.

- Validate real vLLM 0.34 two-round full-attention prefix reuse with identical
  model outputs; fix async load progress and partial-tail offload mapping.
- Add a separate CPU ThreadSanitizer build and CI job.
- Add lifecycle, registration churn, concurrent task completion/cancellation,
  and task-result reclamation stress coverage.
- Keep completion dispatchers alive during concurrent replacement.
- Add cumulative task metrics, bounded get/put latency histograms, and
  dependency-free Prometheus text export.
- Add C++/Python terminal-task `Forget`/`forget` APIs to reclaim results.
- Add optional Doxygen generation, API/ownership documentation and release
  validation instructions.

No release tag yet: the baseline decision and real hybrid-model validation
remain open in the roadmap.
