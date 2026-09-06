# Build Phases

- [x] **Phase 1 — Skeleton graph.** Typed state, supervisor, 4 workers +
      reflector, conditional routing, deterministic node logic, context
      quarantine, and prerequisite-safe reflection/retry routing.

- [x] **Phase 2 — MCP integration.** Custom MCP tools/resources consumed
      through `langchain-mcp-adapters` with committed tool/resource
      evidence.

- [x] **Phase 3 — LLM reasoning.** Gemini primary with a configured Groq
      backup failover (an approved deviation — see `docs/deviations.md`);
      structured outputs; deterministic KYC/credit policy gates and
      constrained offers.

- [x] **Phase 4 — Memory.** `SqliteSaver` pause/resume + LangMem/Chroma
      durable memory + importance-weighted eviction, with cross-process
      persistence evidence.

- [x] **Phase 5 — Context engineering + agentic RAG.** Write/select/compress/isolate
      strategies, long-thread summarization, and discretionary MCP
      `lending_policy_search` selection inside the credit-assessment
      reasoning loop.

- [x] **Phase 6 — Application interface, final evidence, and submission
      readiness.** Required CLI single-command execution, optional
      Streamlit visualization, finalized `.env.example`, committed evidence
      artifacts, and submission documentation. Git PR history remains a
      repository-delivery requirement (see README §18).

- [x] **Phase 6.1 — Independent architecture review + hardening pass.** An
      end-to-end review against the spec surfaced and fixed a set of
      correctness gaps, each with a committed regression test:
      - the supervisor's `next_node` decision was being computed and then
        discarded by a static `add_edge`, so every invocation restarted at
        intake regardless of state — now a real conditional edge
        (`route_after_supervisor`);
      - two compliance-relevant events (a KYC-fail referral, a detected
        prompt-injection attempt) were written into `reflection_log`, where
        they risked being misread as "the current failure" by a later,
        unrelated retry — moved to a dedicated `compliance_flags` channel;
      - the credit agent's one-stub-then-escalate rationale-failure budget
        was gated on the graph-wide `retry_count`, so an earlier unrelated
        retry could silently consume it — now counted from this node's own
        prior failures specifically;
      - `asyncio.run()` was called unconditionally from two sync/async
        bridge points, which would break under a host with its own running
        event loop — replaced with a loop-aware `run_sync()` helper;
      - the MCP tool-schema listing and the Sentence-Transformers embedding
        model were each reloaded on every call/construction — both are now
        cached;
      - PII-redaction logic was implemented twice (in `cli.py` and
        `audit_log.py`) and had already drifted — consolidated into
        `src/observability/redaction.py`;
      - the NFR-08 compression threshold had no proof it was ever reachable
        through real use of the shipped graph — `tests/test_multi_turn_compression.py`
        now drives 5 real, repeated graph runs on one thread and shows it
        genuinely fires.

      Full rationale for each fix, with file-by-file detail, mirrors the
      structure of this list; see the git history for the corresponding
      commits.

## Phase 6 Application Entry Points

### Required

```bash
python cli.py
```

The CLI runs the complete loan-origination workflow through the existing
LangGraph implementation.

### Optional visual interface

```bash
streamlit run app.py
```

The Streamlit interface visualizes graph topology, execution path, stage
status, events, errors, and final outcome using the same underlying graph.

## Final Validation

Deterministic regression:

```bash
pytest -q
```

```text
112 collected — 108 passed, 4 skipped, 0 failed
```

Real provider integration:

```bash
PHASE3_LIVE=1 pytest -q tests/test_phase3_live.py
```

Real agentic-RAG integration:

```bash
PHASE5_LIVE=1 pytest -q tests/test_phase5_live.py
```

A full `python cli.py` run against `sample_inputs/applicant_strong.json`,
using the real Gemini provider and the real MCP server, completes with
exit status 0 and the expected graph path:
`supervisor → intake → kyc_check → credit_assessment → offer_draft → memory_consolidation`.
