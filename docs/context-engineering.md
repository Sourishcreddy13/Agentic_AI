# Context Engineering

The context layer implements four explicit strategies: **write, select,
compress, and isolate**.

## Write

Worker nodes write validated Pydantic objects into graph state — raw model
prose is never treated as durable state. Each worker (intake, KYC, credit,
offer) also appends a short, non-sensitive `AIMessage` describing what it
did (e.g. `"KYC status: pass."`, `"Credit decision: approve (score=740, dti=0.071)."`).
This keeps `state["messages"]` a genuinely growing record of the workflow
rather than a single static entry — which matters directly for **Compress**
below, since compression has nothing to summarize on a thread that never
grows.

Important application facts are also retained in long-term semantic memory
by `memory_consolidation` (see `docs/memory-policy.md`).

## Select

`src/context/select.py` defines per-worker context contracts. Each worker
receives only the domain fields required for its task — e.g. the offer
agent's context selection is `{applicant, credit_assessment,
compressed_summary}`, nothing else. Long-term memory hits
(`long_term_memory_hits`) are contextual information only and are never
passed into a deterministic KYC or credit-policy gate function.

## Compress

`src/context/compress.py::summarize_if_long` implements rolling
summarization. When the working message history exceeds 20 messages, the
older portion is summarized into a bounded `ContextSummary`, while the most
recent 6 messages remain verbatim. Compression runs before downstream
worker model calls, and the result is cached in `compressed_summary` (once
set, it is reused rather than recomputed on every subsequent node call in
the same run — see `src/context/middleware.py::prepare_worker_context`).

Critically, compression operates on a **trusted summary projection**
(`src/context/quarantine.py::build_safe_summary_projection`) rather than
raw applicant free text — a detected injection payload is never
reconstructable from the compressed summary.

**Reachability, not just correctness.** `tests/test_context_engineering.py`
proves `summarize_if_long()` is correct against a hand-built message list.
That alone doesn't show the 20-message threshold is ever actually crossed
in normal use of the shipped graph — a single application run through
intake → KYC → credit → offer produces only about 5 messages.
`tests/test_multi_turn_compression.py` closes that gap: it drives the real
`build_graph()` + real `SqliteSaver` checkpointer + `cli._build_initial_state()`
across 5 separate runs on the *same* `thread_id` (the same applicant
returning to reapply, or resubmitting, in later sessions — a realistic
scenario the resume-aware supervisor is built to support) and shows
`compressed_summary` genuinely gets set by the real compression path once
accumulated history crosses the threshold, while the final application
still completes normally. Evidence: `evidence/multi_turn_compression.json`.

## Isolate

`src/context/quarantine.py` classifies applicant-submitted free text as
untrusted content. Intake extraction uses only trusted structured
application fields; free text is wrapped in an explicit
`<untrusted_applicant_input>` envelope, retained for audit/evidence, and
excluded from every model prompt and from context compression. A detected
injection pattern (e.g. `"ignore previous instructions"`, `"system:"`,
`"you are now"`) is logged and recorded as a `ComplianceEvent` — see
`docs/agent-contracts.md` — but never changes routing or extraction.

This implements the context-isolation requirement (NFR-03) and prevents
applicant-submitted instructions from overriding the agent's control logic.
Proof: `tests/test_quarantine.py`,
`tests/test_compliance_events.py::test_injection_attempt_records_compliance_event_not_reflection_note`.

## Agentic RAG

`src/rag/agentic_rag.py` binds the MCP `lending_policy_search` tool to the
configured model provider and lets the model decide whether a policy
lookup is necessary. Zero tool calls is a valid, tested outcome. Retrieved
policy text is advisory context only — deterministic Python policy gates
remain the sole authority for lending decisions. See `docs/mcp-integration-decision.md`
and `docs/agent-contracts.md` for the tool classification.
