# API and ownership guide

Build the C++ reference with Doxygen installed:

```sh
cmake -S . -B build-docs -DSTRATAKV_BUILD_DOCS=ON -DSTRATAKV_BUILD_TESTS=OFF
cmake --build build-docs --target stratakv_api_docs
```

Open `build-docs/html/index.html`. CI uploads the generated reference.
Public headers under `include/stratakv` define the C++ contract; the
Python facade lives in `python/stratakv/__init__.py`.

## Request lifecycle

1. Declare the model's layout groups and construct `Engine`.
2. GPU mode: register each group's borrowed tensors and call `start()`.
   Keep their allocation alive through transport shutdown. CPU mode uses
   a simulation executor: it exercises control flow without copying bytes.
3. Call `stage_put(tokens)`, then `submit_put(plan, gpu_blocks,
   gpu_state_slots)`. Block slots parallel `plan.blocks_to_store` in
   position-major, group-minor order. State slots parallel the sidecars.
4. Wait or poll for terminal status. Successful puts become matchable.
5. Call `match(tokens)`. A match pins CPU entries until `submit_get`
   reaches terminal status or `release_match` explicitly releases it.
6. `submit_get(match, gpu_slots, gpu_state_slots)` restores that bundle.
7. After consuming each terminal result, call `forget(handle)` to reclaim
   its record. Results otherwise remain queryable until engine destruction.
8. Call `shutdown()` before freeing borrowed GPU allocations.

`wait()` releases the GIL. Metadata calls and metrics snapshots hold it;
`try_wait()` flushes pending native transfers. `metrics()` does not flush.
The native completion descriptor is a nonblocking pipe. Drain notifications
on the application event-loop thread; worker threads never call Python.

## Failure and cancellation

Terminal states are `succeeded`, `failed`, and `cancelled`. Inspect the
error string returned by `wait`/`try_wait`. Cancelling is best effort for
already launched transfers; retain underlying GPU allocations until the
transport is drained. Failed/cancelled put reservations require the
explicit `abort_put(plan)` path after transfers stop using their buffers.
Submission validation or backpressure errors leave ownership with the
caller: release an unsubmitted match or abort an unsubmitted put plan.

`forget()` returns false for active or unknown tasks. It discards pending
completion callbacks for a forgotten result; a callback already being
dispatched may still run. Subsequent queries report an unknown handle.
Task IDs are never reused within an engine. Cumulative metrics survive
record reclamation.

## Metrics

`Engine.metrics()` returns a coherent cumulative snapshot. Submissions,
terminal outcomes, backpressure/shutdown rejections, active/peak task counts,
and retained results are tracked. Get and put histograms measure accepted
submission to terminal transition, including batching delay. They do not
measure GPU kernel time or metadata matching latency. Failed and cancelled
tasks contribute to the latency histograms. Buckets are disjoint, bounded
at 1/10/100 microseconds, 1/10/100 milliseconds, 1 second, and infinity.
`rejected` counts submissions rejected for saturation or shutdown, not
argument-validation exceptions.

```python
from stratakv.metrics import prometheus_text
text = prometheus_text(engine.metrics())
```

Serve this text from the application's metrics endpoint. The exporter
starts no threads, HTTP server, or dependency; histogram buckets are
converted to cumulative seconds as required by Prometheus. Counters reset
when the engine is recreated. Scraping does not advance transfers.

Phase histograms add `match_latency`, `queue_latency`, and `enqueue_latency`
to the snapshot. Match covers every `MatchForGet` call, including exceptions
and index lock waits. Queue measures residence from graph insertion until
flush, one observation per flushed graph; cancellation before flush produces
no sample. Enqueue measures host call duration per batch, including exceptions
and any synchronous callbacks. It excludes asynchronous CUDA completion.
Queue/enqueue are absent for the simulation executor. These histograms are
individually coherent; a snapshot across phases is not globally atomic.
Prometheus exports them as `stratakv_phase_<phase>_duration_seconds`.
