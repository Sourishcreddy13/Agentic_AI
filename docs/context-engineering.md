# Context Engineering

The context layer implements four explicit strategies: **write, select,
compress, and isolate**.

## Write

Worker nodes write validated Pydantic objects into graph state. Raw model prose
is never treated as durable state.

Important application facts are retained in typed state and, where appropriate,
in long-term semantic memory.

## Select

`src/context/select.py` defines per-worker context contracts. Each worker receives
only the domain fields required for its task.

Long-term memory hits are contextual information only and are never passed into
deterministic KYC or credit-policy gate functions.

## Compress

`src/context/compress.py` implements rolling summarization.

When the working message history exceeds 20 messages, the older portion is
summarized into a bounded `ContextSummary`, while only the most recent six
messages remain verbatim.

Compression is performed before downstream worker model calls and the resulting
summary is stored in `compressed_summary`.

Critically, compression operates on a **trusted summary projection** rather than
raw applicant free-text.

## Isolate

`src/context/quarantine.py` classifies applicant-submitted free text as
untrusted content.

Intake extraction uses trusted structured application fields. The quarantine
envelope is retained for audit/evidence, while raw applicant free text is
excluded from context compression and is never supplied to downstream workers
as instruction-bearing content.

This implements the context-isolation requirement and prevents applicant
instructions from overriding the agent's control logic.

## Agentic RAG

`src/rag/agentic_rag.py` binds the MCP `lending_policy_search` tool to the
configured model provider and allows the model to decide whether a policy lookup
is necessary.

Zero tool calls are valid.

Retrieved policy text is advisory context only. Deterministic Python policy gates
remain the sole authority for lending decisions.
