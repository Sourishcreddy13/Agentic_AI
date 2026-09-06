# Single-vs-Multi-Agent Decision (NFR-06)

## Decision: Multi-Agent Supervisor Pattern

The system uses a **supervisor + four specialized worker agents**:

- intake
- KYC-check
- credit-assessment
- offer-draft

This is preferred over a single monolithic agent or a peer-to-peer swarm.

## Why Not a Single Agent?

A single agent handling all four stages would create:

1. **Context pollution** — KYC details, credit information, and offer
   parameters would accumulate in one context, making stage-specific
   behavior harder to isolate, test, and audit independently.
2. **Reduced testability** — A monolithic agent is harder to validate than
   specialized nodes with explicit input/output contracts and deterministic
   test doubles (see `docs/agent-contracts.md` and `tests/conftest.py`'s
   per-schema fake LLM).
3. **Diluted policy enforcement** — The project's core principle
   (deterministic Python owns every hard decision; the LLM only explains)
   is far easier to enforce and verify at four narrow node boundaries than
   inside one broad agent making four kinds of decisions in the same pass.

The multi-agent design gives each stage an explicit input/output contract.

## Why Not a Swarm?

Loan origination is a **strictly gated workflow**:

```text
Intake → KYC → Credit → Offer
```

Workers do not need peer-to-peer negotiation — there is nothing to
negotiate. Control flow is a directed graph driven entirely by policy
gates reading MCP facts. A peer-to-peer swarm would add coordination
overhead without a material benefit for this domain.

## Why Supervisor Pattern?

The supervisor is a pure Python entry router (`src/graph/supervisor.py`),
consulted through a **real conditional edge**
(`route_after_supervisor` in `src/graph/routing.py`, wired into
`src/graph/build_graph.py` via `add_conditional_edges`, not a static
`add_edge`). It inspects how much of the application state is already
populated and dispatches to the correct next stage — including resuming a
thread that already has partial or complete progress checkpointed, rather
than unconditionally restarting at intake. After that first hop, every
subsequent transition is driven by the corresponding worker's own output
through the graph's other conditional edges (`route_after_intake`,
`route_after_kyc`, `route_after_credit`, `route_after_reflection`).

This provides:

- a single, auditable entry point whose decision is actually followed
  (proven by `tests/test_supervisor_resume.py` — a completed thread
  receiving a follow-up message resumes straight to `END` with no worker
  re-executed);
- explicit, state-driven routing at every hop;
- deterministic, regulator-auditable policy gates independent of which LLM
  answered;
- a central reflection/recovery mechanism (`src/agents/reflector.py`) that
  every worker routes through on failure.

## Framework Choice: LangGraph over CrewAI

LangGraph was selected because:

- typed shared state (`TypedDict` + Pydantic sub-models) is a first-class
  primitive, not a convention layered on top;
- `add_conditional_edges` directly represents gated workflow decisions,
  rather than requiring a custom router abstraction;
- `SqliteSaver` checkpointing supports exact pause/resume per thread
  (AC-05) with no external database;
- the LangChain ecosystem integrates naturally with `langchain-mcp-adapters`
  and LangMem, avoiding a second set of tool/memory abstractions.

CrewAI could represent the workflow at a coarser granularity, but the
strict sequential/gated nature of loan origination — and the requirement
that routing decisions be state-driven and auditable rather than
role-negotiated — is more directly and verifiably expressed as an explicit
LangGraph state machine.
