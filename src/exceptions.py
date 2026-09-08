"""Project-wide exception types.

`WorkerPreconditionError` replaces a small number of bare `assert`
statements that used to guard node-entry invariants (e.g. "offer_draft_node
must never be reached without a credit assessment"). Two reasons a real
exception is used instead of `assert`:

1. `assert` is compiled out entirely under `python -O` / `PYTHONOPTIMIZE`,
   silently removing the guard in an optimized deployment.
2. A named exception type is unambiguous in `src/graph/build_graph.py`'s
   `_observed_node` wrapper, which logs `error_type=type(exc).__name__` for
   any unhandled exception — `WorkerPreconditionError` is self-describing in
   that log line, where a bare `AssertionError` is not.

`offer_draft_node` is the one place this project still raises rather than
returning a `ReflectionNote`: unlike `kyc_check` and `credit_assessment`,
`offer_draft`'s only outgoing edge is a static `add_edge` to
`memory_consolidation` (see `src/graph/build_graph.py`) — there is no
conditional edge from `offer_draft` to `reflector`. Returning a
`ReflectionNote` from `offer_draft_node` would therefore be silently
ignored by routing: the graph would proceed straight to
`memory_consolidation` and `END` with `offer` left `None`, reporting a
*successful* run (exit status 0) that quietly produced no offer. Raising
here is deliberately louder than that — it surfaces the invariant
violation as a failed run (as `cli.py`'s own top-level exception handler
already expects) rather than papering over it as a graceful outcome the
current graph topology cannot actually route to reflection anyway.
"""
from __future__ import annotations


class WorkerPreconditionError(RuntimeError):
    """A node was entered without a precondition the graph topology should
    have guaranteed. Indicates a routing/graph-wiring bug, not a normal
    runtime failure — never raised for applicant-caused conditions."""
