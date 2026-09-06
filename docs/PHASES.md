# Build Phases

- [x] **Phase 1 — Skeleton graph.** Typed state, supervisor, 4 workers + reflector,
      conditional routing, deterministic node logic, context quarantine, and
      prerequisite-safe reflection/retry routing.

- [x] **Phase 2 — MCP integration.** Custom MCP tools/resources consumed through
      `langchain-mcp-adapters` with committed tool/resource evidence.

- [x] **Phase 3 — LLM reasoning.** Gemini primary with configured Groq backup
      failover; structured outputs; deterministic KYC/credit policy gates and
      constrained offers.

- [x] **Phase 4 — Memory.** `SqliteSaver` pause/resume + LangMem/Chroma durable
      memory + importance-weighted eviction with cross-process persistence
      evidence.

- [x] **Phase 5 — Context engineering + agentic RAG.** Write/select/compress/isolate
      strategies, long-thread summarization, and discretionary MCP
      `lending_policy_search` selection inside the credit-assessment reasoning loop.

- [x] **Phase 6 — Application interface, final evidence, and submission readiness.**
      Required CLI single-command execution, optional Streamlit visualization,
      finalized `.env.example`, committed evidence artifacts, and submission
      documentation. Git PR history remains a repository-delivery requirement.

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

The Streamlit interface visualizes graph topology, execution path, stage status,
events, errors, and final outcome using the same underlying graph.

## Final Validation

Deterministic regression:

```bash
pytest
```

Real provider integration:

```bash
PHASE3_LIVE=1 pytest -q tests/test_phase3_live.py
```

Real agentic-RAG integration:

```bash
PHASE5_LIVE=1 pytest -q tests/test_phase5_live.py
```
