# ADR 0004 — GIL and thread dispatch policy

- Status: Accepted
- Date: 2026-11-19

## Context

The runtime is consumed from Python (vLLM adapter). Two hazards observed
in the FlexKV audit: (a) per-step Python polling (`try_wait`) burns
interpreter time on the engine's critical path; (b) completion callbacks
arriving on arbitrary native threads are unsafe to map into Python and
contend with the engine loop.

## Decision

1. **Blocking native work releases the GIL** (`pybind11::gil_scoped_release`)
   at every facade entry point that may wait; the GIL is reacquired before
   touching any Python object. Pure metadata calls (match) may hold the GIL
   only if measured in the tens of microseconds.
2. **Completion delivery is poll- or dispatch-based, never spontaneous**:
   - `TaskEngine::CompletionFd()` exposes an eventfd for event-loop
     integration (the vLLM EngineCore loop can poll it);
   - `DrainCompletions()` dispatches callbacks on the caller's thread;
   - if a callback mode is used, callbacks fire only on the single
     documented `CompletionDispatcher` thread, which acquires the GIL
     before entering Python.
3. **No native worker thread ever calls into Python directly.**

## Consequences

- Per-step cost in the engine is one `try_wait`-equivalent native call or
  a poll of the completion fd — replacing FlexKV's Python-side
  `_update_tasks`/ZMQ round trips.
- The bindings must document, per method, its GIL behavior.
