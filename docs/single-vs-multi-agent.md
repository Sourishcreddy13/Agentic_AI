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

1. **Context pollution** — KYC details, credit information, and offer parameters
   accumulate in one context, making stage-specific behavior harder to isolate
   and audit.

2. **Reduced testability** — A monolithic agent is harder to validate independently
   than specialized nodes with explicit contracts and deterministic test doubles.

The multi-agent design gives each stage an explicit input/output contract.

## Why Not a Swarm?

Loan origination is a **strictly gated workflow**:

```text
Intake
  ↓
KYC
  ↓
Credit
  ↓
Offer
```

Workers do not need peer-to-peer negotiation. Control flow is a directed graph
driven by policy gates.

A peer-to-peer swarm adds coordination overhead without providing a material
benefit for this domain.

## Why Supervisor Pattern?

The supervisor is a pure Python entry router. After the first hop, worker output
drives conditional graph edges.

This provides:

- a single auditable entry point;
- explicit state-driven routing;
- deterministic regulatory/policy gates;
- a central reflection/recovery mechanism.

## Framework Choice: LangGraph over CrewAI

LangGraph was selected because:

- typed shared state is a first-class primitive;
- `add_conditional_edges` directly represents gated workflow decisions;
- checkpointing supports pause/resume;
- the LangChain ecosystem integrates naturally with MCP adapters and LangMem.

CrewAI could represent the workflow, but the strict sequential/gated nature of
loan origination is more directly expressed as an explicit LangGraph state
machine.
