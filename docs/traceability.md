# AC Traceability Matrix

The committed requirements baseline is `specs/loan_origination_spec.md`.
Each Acceptance Criterion is covered by at least one test or committed evidence
artifact carrying its AC-NN identifier.

| AC / NFR | Evidence artifact / implementation | Test(s) |
|---|---|---|
| AC-01 | `src/state/schema.py` | `tests/test_state_schema.py` |
| AC-02 | `evidence/run_transcripts/strong_applicant.json` | `tests/test_routing.py::test_strong_applicant_reaches_approved_offer` |
| AC-03 | `evidence/run_transcripts/strong_applicant.json` | `tests/test_routing.py` routing tests |
| AC-04 | `src/agents/*.py` structured output + Pydantic contracts | `tests/test_llm_gateway.py`, `tests/test_phase3_live.py` |
| AC-05 | `evidence/checkpoint_pause.json`, `evidence/checkpoint_resume.json` | `tests/test_checkpoint_resume.py` |
| AC-06 | `evidence/ac06_same_session_memory.json`, `evidence/memory_write.json` | `tests/test_same_session_memory.py` |
| AC-07 | `evidence/cross_session_memory.log` | `tests/test_cross_session_memory.py` |
| AC-08 | `docs/memory-policy.md` | `tests/test_eviction_policy.py` |
| AC-09 | `evidence/mcp_transcript.json` — 3 tools + `policy://credit_policy_manual` resource + resource retrieval | `tests/test_mcp_tools.py`, `tests/test_mcp_adapter.py` |
| AC-10 | `evidence/mcp_transcript.json`, `evidence/run_transcripts/phase2_mcp_graph.json` | `tests/test_mcp_adapter.py`, `tests/test_mcp_graph_integration.py` |
| AC-11 | `src/rag/agentic_rag.py`, `evidence/run_transcripts/phase5_agentic_rag.json` | `tests/test_agentic_rag.py`, `tests/test_phase5_live.py` |
| AC-12 | `evidence/reflection_trace.json` | `tests/test_reflection_loop.py` |
| NFR-01 | `.env.example` | Configuration/security review |
| NFR-02 | `cli.py`, `README.md` | CLI execution + committed sample inputs |
| NFR-03 | `src/context/quarantine.py` | `tests/test_quarantine.py` |
| NFR-04 | `evidence/` structured traces | Evidence-generating tests |
| NFR-05 | `synthetic_data/`, redaction evidence | `tests/test_observability.py` |
| NFR-06 | `docs/single-vs-multi-agent.md` | Architecture decision record |
| NFR-07 | `src/llm/gateway.py`, `src/graph/routing.py` | `tests/test_llm_gateway.py`, `tests/test_reflection_loop.py` |
| NFR-08 | `src/context/compress.py` | `tests/test_context_engineering.py` |

## Application Entry Points

### Mandatory

```bash
python cli.py
```

### Optional Good-to-Have UI

```bash
streamlit run app.py
```

The CLI is the required single-command application path. Streamlit is a visual
interface and does not replace the CLI requirement.
