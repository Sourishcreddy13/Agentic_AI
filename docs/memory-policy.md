# Memory Policy

## Scope

The system uses two independent persistence tiers:

- **Short-term working state:** `SqliteSaver`, keyed by `thread_id`, for
  exact pause/resume of one application (AC-05).
- **Long-term semantic memory:** Chroma with Sentence-Transformers
  embeddings, keyed by `user_id`, for durable facts that may inform later
  sessions (AC-06, AC-07).

Long-term memory is context only. Python KYC and credit policy functions
never accept or read `long_term_memory_hits`, so historical memory cannot
override a current-session policy gate.

## Memory Lifecycle

1. A new graph invocation reads `memory_enabled` from LangGraph's
   `configurable` mapping. Application runs use memory by default; ordinary
   tests disable it unless they are memory-integration tests.
2. `intake_node` queries Chroma for the current `user_id` and writes
   retrieved fact strings to `state["long_term_memory_hits"]`.
3. Downstream workers may use those hits as explanatory context, but policy
   decisions use only current trusted application facts and current MCP
   results.
4. After `offer_draft` completes, `memory_consolidation` performs
   post-decision extraction and persistence.
5. Only `MemoryFact` records are written. Raw prompts, applicant free text,
   MCP payloads, routing internals, and entire graph state are never
   persisted as long-term memory.

## Memory Fact Importance

| Fact type | Importance |
|---|---:|
| `kyc_outcome` | 0.95 |
| `credit_outcome` | 0.90 |
| `prior_application_count` | 0.70 |
| `declared_income_band` | 0.60 |
| `employment` | 0.55 |
| `preferred_term` | 0.40 |

## Retention Score

```text
retention_score =
    0.45 * importance
  + 0.30 * recency_score
  + 0.25 * usage_score
```

`recency_score` is 1.0 for facts younger than 30 days and decays linearly
to 0 at 180 days. `usage_score` is normalized access count, capped at 10
accesses.

Additional limits:

- Hard TTL: 365 days. Older facts are always evicted.
- Eviction threshold: retention score below 0.25.
- Maximum: 50 facts per `user_id`.
- At capacity, the lowest-retention facts are evicted first; last-access
  time is the deterministic tie-break.

Semantic relevance is deliberately excluded from eviction — it belongs to
retrieval, not retention. See `src/memory/eviction.py` and
`tests/test_eviction_policy.py`.

## Storage Isolation

Chroma uses a single collection named `applicant_facts`. Every record
contains:

```text
user_id
fact_type
importance
session_ts
thread_id
usage_count
last_access_ts
```

Retrieval always applies the explicit metadata filter `user_id == current
user_id`, preventing cross-user memory leakage.

## LangMem Extraction

The preferred extractor is `langmem.create_memory_manager` with the
project's structured memory schema. If the LangMem extraction API cannot be
used cleanly with the project's provider/storage boundary, extraction falls
back to the shared Gemini-primary/Groq-fallback structured-output gateway
(`src/llm/gateway.py`).

## Failure Semantics

Memory failures are fail-open for the completed loan decision. If retrieval
or post-decision consolidation fails, the graph does not alter an
already-determined KYC, credit, or offer outcome.

## Performance Notes

`ChromaMemoryStore` is constructed fresh on nearly every node invocation
that touches long-term memory (intake's lookup, memory_consolidation's
post-decision write). The Sentence-Transformers model it wraps is loaded
once per process per model name via `functools.lru_cache`
(`src/memory/long_term_store.py::_embedding_model`) — the same pattern
`src/rag/policy_store.py` already uses for the RAG embedding model —
instead of being re-loaded from disk on every construction.

## Test Isolation

Ordinary tests use `memory_enabled=False` through the autouse test fixture
and therefore perform no persistent Chroma writes. Memory-integration tests
(`@pytest.mark.memory_integration`) use `memory_enabled=True` and isolated
temporary Chroma storage, so the default regression suite stays hermetic
and safe to run repeatedly without accumulating state.
