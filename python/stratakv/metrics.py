"""Prometheus text export without a server or third-party dependencies.

Call ``prometheus_text(engine.metrics())`` from the application's existing
metrics endpoint. Counters are per engine lifetime, not per scrape.
"""


def prometheus_text(snapshot: dict) -> str:
    """Render a native task-metrics snapshot in Prometheus text format."""
    lines = []
    for field in ("submitted_gets", "submitted_puts", "succeeded", "failed",
                  "cancelled", "rejected"):
        name = f"stratakv_tasks_{field}_total"
        lines += [f"# TYPE {name} counter", f"{name} {snapshot[field]}"]
    for field in ("in_flight", "peak_in_flight", "retained_tasks"):
        name = f"stratakv_tasks_{field}"
        lines += [f"# TYPE {name} gauge", f"{name} {snapshot[field]}"]
    for direction in ("get", "put"):
        hist = snapshot[f"{direction}_latency"]
        name = f"stratakv_task_{direction}_duration_seconds"
        lines.append(f"# TYPE {name} histogram")
        cumulative = 0
        for bound, count in zip([*hist["bounds_us"], None], hist["buckets"]):
            cumulative += count
            le = "+Inf" if bound is None else format(bound / 1e6, ".9g")
            lines.append(f'{name}_bucket{{le="{le}"}} {cumulative}')
        lines += [f'{name}_count {hist["count"]}',
                  f'{name}_sum {hist["sum_seconds"]:.17g}']
    return "\n".join(lines) + "\n"
