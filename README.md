# Loan Origination Copilot

A LangGraph multi-agent copilot for bank loan origination: a supervisor
dispatches a typed, checkpointed application through intake, KYC, credit
assessment, and offer-drafting agents, backed by a custom MCP tool server,
engineered context (write/select/compress/isolate), tiered short- and
long-term memory, discretionary agentic-RAG policy lookup, and a bounded
reflection/self-healing loop.

All data — applicants, bureau records, watchlist hits, the lending-policy
manual — is synthetic. No real applicant, bureau, account, or confidential
data is used anywhere in this repository.

**Run it:** `python cli.py` · **Test it:** `pytest -q` · **Validated:** `112 collected — 108 passed, 4 skipped, 0 failed`

---

## Table of Contents

1. [Problem & Approach](#1-problem--approach)
2. [Quick Start](#2-quick-start)
3. [Architecture](#3-architecture)
4. [The Core Design Principle](#4-the-core-design-principle)
5. [Agent Contracts](#5-agent-contracts)
6. [MCP Server & Integration](#6-mcp-server--integration)
7. [Context Engineering](#7-context-engineering)
8. [Memory](#8-memory)
9. [Agentic RAG](#9-agentic-rag)
10. [Reflection & Self-Healing](#10-reflection--self-healing)
11. [Testing](#11-testing)
12. [Evidence](#12-evidence)
13. [Configuration](#13-configuration)
14. [Requirements → Rubric Mapping](#14-requirements--rubric-mapping)
15. [Acceptance Criteria Traceability](#15-acceptance-criteria-traceability)
16. [Project Structure](#16-project-structure)
17. [Documentation Index](#20-documentation-index)

---

## 1. Problem & Approach

Manual loan origination is slow, inconsistent across officers, and opaque to
the applicant. This copilot automates the repeatable parts of the workflow —
capture, KYC, credit-policy assessment, indicative offer/referral — while
keeping every lending decision deterministic, auditable, and attributable to
an explicit Python policy gate rather than to an LLM's judgment call.

The full business case, actors, and success metrics are in
[`docs/business-case.md`](docs/business-case.md). The single-vs-multi-agent
and framework-choice rationale (NFR-06) is in
[`docs/single-vs-multi-agent.md`](docs/single-vs-multi-agent.md).

## 2. Quick Start

```bash
# 1. Python 3.11+ virtual environment
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Dependencies
pip install -r requirements.txt
```

> **No Python 3.11, or `python`/`pip` not found after activating (macOS)?**
> A venv's `python` symlink points at the exact interpreter that created it
> and breaks if that interpreter isn't present on the machine — this bites
> anyone who copies a `.venv` between machines. If you have
> [`uv`](https://docs.astral.sh/uv/) installed, it's the fastest fix:
> ```bash
> rm -rf .venv
> uv python install 3.11      # only needed once per machine
> uv venv --python 3.11 .venv
> source .venv/bin/activate
> uv pip install -r requirements.txt
> ```
> `python --version` should now report `3.11.x` inside the venv.

```bash
# 3. Provider credentials (never commit the real .env)
cp .env.example .env
# edit .env and set GOOGLE_API_KEY (required) and GROQ_API_KEY (backup — see §17)

# 4. Run the required single-command application
python cli.py

# 5. Run the deterministic regression suite
pytest -q
```

Run against a specific committed synthetic application:

```bash
python cli.py --input sample_inputs/applicant_strong.json
python cli.py --input sample_inputs/applicant_kyc_fail.json
python cli.py --input sample_inputs/applicant_thin_file.json
```

An optional visual interface is available (it drives the *same* graph, not a
reimplementation):

```bash
streamlit run app.py
```

### What a run looks like

The trace below is an actual `python cli.py` execution against
`applicant_strong.json`, using the real Gemini provider and the real MCP
server (abridged — full stage-by-stage JSON is printed to the terminal):

```text
✓ supervisor               completed        13 ms   →  next_node: intake
✓ intake                   completed     14778 ms   →  applicant SYN-0001 extracted
✓ kyc_check                completed      3529 ms   →  kyc_result.status: pass
✓ credit_assessment        completed     20030 ms   →  decision: approve (score=740, dti=0.071)
✓ offer_draft              completed      5832 ms   →  principal=3,000,000  apr=9.5%  term=48mo
✓ memory_consolidation     completed      5278 ms

GRAPH FLOW
  supervisor -> intake -> kyc_check -> credit_assessment -> offer_draft -> memory_consolidation
Exit status: 0
```

The offer's principal/APR/term are not the LLM's free choice — they are the
Python-enforced ceiling for a 700–749 synthetic bureau score with income
over $75k (`PRICE-001`, see `src/agents/offer_agent.py::_offer_constraints`).
Whatever Gemini proposes, `_enforce_constraints()` clips it to that tier.

## 3. Architecture

```text
                                   ┌───────────────────────┐
                                   │      supervisor        │
                                   │  (pure Python router)  │
                                   └───────────┬────────────┘
                                               │  route_after_supervisor
                    ┌──────────────────────────┼──────────────────────────┬─────────┐
                    │ no applicant             │ no kyc_result            │ no      │ all set
                    ▼                          ▼                          │ credit_ │  → END
              ┌──────────┐               ┌──────────┐                    │assessment│
              │  intake  │──kyc_check───▶│kyc_check │                    ▼         │
              └────┬─────┘               └────┬─────┘             ┌─────────────┐  │
                   │ reflector                 │ MCP:              │  credit_    │  │
                   │ (parse/LLM failure)       │ applicant_lookup  │ assessment  │◀─┘
                   ▼                           │                   └──────┬──────┘
              ┌──────────┐                     │ fail → offer_draft       │ MCP: bureau_check
              │reflector │◀────────────────────┤ manual/low-conf→reflector│ + agentic lending_policy_search
              └────┬─────┘                     ▼                          ▼
                   │ retry / replan       ┌──────────┐            manual/low-conf → reflector
                   │ (bounded, prereq-    │  (pass)  │                    │ decline/approve
                   │  safe re-dispatch)   └────┬─────┘                    ▼
                   │                           └──────────────────▶┌─────────────┐
                   └── escalate_to_human ──────────────────────────▶ offer_draft  │
                                                                    └──────┬──────┘
                                                                           ▼
                                                                 memory_consolidation
                                                                           ▼
                                                                          END
```

The supervisor is consulted by a **real conditional edge**
(`route_after_supervisor` in `src/graph/routing.py`, wired in
`src/graph/build_graph.py`), not a static pass-through — a thread that
already has partial or complete progress checkpointed dispatches straight
to its correct next stage instead of restarting at intake. See
[`tests/test_supervisor_resume.py`](tests/test_supervisor_resume.py) for the
proof (a completed application resumes to `END` on a follow-up message
without re-running any worker).

### Main components

| Path | Responsibility |
|---|---|
| `src/state/schema.py` | Typed `LoanApplicationState` (TypedDict) + Pydantic domain contracts, validated at every node handoff (AC-01, AC-04). |
| `src/graph/` | `supervisor.py` (entry router), `routing.py` (all conditional edges), `build_graph.py` (topology). |
| `src/agents/` | `intake_agent.py`, `kyc_agent.py`, `credit_agent.py`, `offer_agent.py`, `reflector.py`, `memory_agent.py`. |
| `src/mcp_client.py` + `mcp_server/` | Custom MCP server + `langchain-mcp-adapters` stdio client, with a cached tool-schema listing. |
| `src/llm/gateway.py` | Gemini-primary / Groq-fallback structured-output gateway. |
| `src/context/` | `select.py`, `compress.py`, `quarantine.py`, `middleware.py` — write/select/compress/isolate. |
| `src/memory/` | `checkpointer.py` (short-term), `long_term_store.py` (Chroma semantic memory), `eviction.py`, `runtime.py`. |
| `src/rag/` | `agentic_rag.py` — discretionary `lending_policy_search` tool-calling loop. |
| `src/async_utils.py` | Event-loop-safe `run_sync()` helper shared by the MCP client and the RAG loop. |
| `src/observability/` | `audit_log.py`, `redaction.py` — structured JSONL logging with PII redaction. |
| `cli.py` | Required single-command application entry point. |
| `app.py` | Optional Streamlit visual console over the same graph. |
| `mcp_server/server.py` | The domain MCP server (tools + resource). |
| `tests/` | Deterministic hermetic suite + gated live-provider suites. |
| `evidence/` | Committed run transcripts, traces, and test-run summaries (regenerated by `pytest`/`cli.py`). |
| `sample_inputs/` | Committed synthetic application fixtures. |
| `specs/` | The requirements baseline (`loan_origination_spec.md`) with AC-NN/NFR-NN identifiers. |

### Technology Stack

| Layer | Choice |
|---|---|
| Language | Python 3.11+ |
| Agent framework | LangGraph (`add_conditional_edges`, `SqliteSaver` checkpointing) |
| Structured contracts | Pydantic v2 |
| LLM provider | Google Gemini — sole provider per spec; Groq is a configured, approved backup (§17) |
| Interoperability | MCP Python SDK (stdio) + `langchain-mcp-adapters` |
| Memory | `langgraph-checkpoint-sqlite` (short-term) + Chroma + Sentence-Transformers (long-term semantic) |
| Retrieval | Chroma, local embeddings, for the agentic-RAG lookup tool |
| Interface | CLI (required) + Streamlit (optional visual console) |

No Docker and no external database service are required — SQLite and a
local Chroma directory are the only persistence.

## 4. The Core Design Principle

> **LLMs generate interpretations, explanations, and structured
> recommendations. Deterministic Python code and trusted MCP tool results
> enforce every hard lending-policy decision, state transition, eligibility
> gate, and safety constraint.**

This is enforced at every node boundary, not just claimed in prose:

| Field | Set by | Never set by |
|---|---|---|
| `kyc_result.status` | Python (`_apply_kyc_policy`, from the MCP `applicant_lookup` fact) | Gemini/Groq |
| `credit_assessment.decision` | Python (`_apply_policy_gates`, from the MCP `bureau_check` fact) | Gemini/Groq |
| `offer.principal` / `apr` / `term_months` | Python-enforced ceiling (`_enforce_constraints`, PRICE-001 tiers) | Gemini/Groq (proposes a draft; Python clips it) |
| `next_node` / routing | Python (`src/graph/routing.py` conditional edges) | Gemini/Groq |
| `kyc_result.rationale`, `credit_assessment.rationale`, offer copy | Gemini/Groq | — |

Full per-agent contracts are in
[`docs/agent-contracts.md`](docs/agent-contracts.md).

## 5. Agent Contracts

| Agent | Type | Tools | LLM role | Owns |
|---|---|---|---|---|
| **Supervisor** | Pure Python router | none | none | Resume-aware entry-hop dispatch (`next_node`) |
| **Intake** | LLM extraction + guard | none | Structured extraction into `ApplicantProfile` from trusted fields only | Quarantining applicant free text; rejecting LLM-mutated trusted fields |
| **KYC** | MCP fact + Python gate + LLM explanation | `applicant_lookup` | Rationale + confidence only | `KYCResult.status` classification |
| **Credit** | MCP fact + Python gate + LLM rationale | `bureau_check`, `lending_policy_search` (agentic) | Rationale + confidence only | `CreditAssessment.decision` |
| **Offer Draft** | Python guard + LLM draft + Python clip | `lending_policy_search` (agentic) | Drafts `OfferDraft` inside Python-supplied bounds | Hard pricing ceiling enforcement |
| **Reflector** | Pure Python failure classifier | none | none | Retry / replan / escalate taxonomy, bounded by `MAX_RETRIES=2` |
| **Memory Consolidation** | Extraction + persistence | Chroma write | Fact extraction (LangMem, gateway fallback) | Post-decision durable fact writes |

See [`docs/agent-contracts.md`](docs/agent-contracts.md) for full policy-gate
thresholds (DTI bands, score bands, thin-file rules) and the identity model
(`user_id` vs `thread_id`).

## 6. MCP Server & Integration

`mcp_server/server.py` exposes:

| Tool | Kind | Called by |
|---|---|---|
| `applicant_lookup` | Deterministic, required | KYC agent |
| `bureau_check` | Deterministic, required | Credit agent |
| `lending_policy_search` | Agentic, discretionary | Credit / Offer agent (only if the model decides it's needed) |

| Resource | Content |
|---|---|
| `policy://credit_policy_manual` | Full text of the synthetic credit policy manual |

Tools are consumed through `langchain-mcp-adapters` over stdio
(`src/mcp_client.py`). The tool-schema listing is cached at module scope
after the first successful load — earlier revisions rebuilt the MCP client
and re-ran the `ListTools` handshake on *every* tool invocation; see
[`tests/test_mcp_tools_caching.py`](tests/test_mcp_tools_caching.py) and
[`docs/mcp-integration-decision.md`](docs/mcp-integration-decision.md).

Committed transcript: [`evidence/mcp_transcript.json`](evidence/mcp_transcript.json).

## 7. Context Engineering

| Strategy | Implementation |
|---|---|
| **Write** | Validated Pydantic objects go into state; each worker also appends a short `AIMessage` describing what it did, so `state["messages"]` is a genuine growing record rather than one static entry. |
| **Select** | `src/context/select.py` gives each worker only the state fields its task needs. |
| **Compress** | `src/context/compress.py::summarize_if_long` — past 20 messages, everything but the most recent 6 is summarized into a bounded `ContextSummary` over a **trusted projection**, never raw applicant free text. |
| **Isolate** | `src/context/quarantine.py` wraps applicant-submitted free text in an explicit quarantine envelope; it is excluded from every prompt and from compression. |

The 20-message compression threshold is not just unit-tested against a
hand-built list — [`tests/test_multi_turn_compression.py`](tests/test_multi_turn_compression.py)
drives the real graph + real `SqliteSaver` checkpointer across 5 separate
runs on one `thread_id` (a returning applicant) and shows
`compressed_summary` genuinely gets set once real accumulated history
crosses the threshold. Details in
[`docs/context-engineering.md`](docs/context-engineering.md).

## 8. Memory

```text
Short-term  →  SqliteSaver checkpoint, keyed by thread_id (one application)
Long-term   →  Chroma + Sentence-Transformers, keyed by user_id (durable facts)
```

Long-term memory is context only — `long_term_memory_hits` is never read by
a Python policy-gate function, so historical memory cannot influence a
current decision. Retention uses an importance-weighted score with TTL and
a per-user cap; full formula and thresholds in
[`docs/memory-policy.md`](docs/memory-policy.md).

Cross-session persistence evidence: [`evidence/cross_session_memory.log`](evidence/cross_session_memory.log),
[`evidence/checkpoint_pause.json`](evidence/checkpoint_pause.json) / [`checkpoint_resume.json`](evidence/checkpoint_resume.json).

## 9. Agentic RAG

```text
Credit / Offer agent
     ↓
model decides whether a policy lookup is useful (zero calls is a valid outcome)
     ↓
lending_policy_search  (MCP tool, Chroma semantic search)
     ↓
clause-level policy evidence returned as advisory context
     ↓
feeds the LLM's rationale — never the Python decision
```

Retrieval is inside the reasoning loop (`src/rag/agentic_rag.py`), not a
fixed graph step — this is what makes it "agentic" for AC-11. Retrieved
text can never override `_apply_policy_gates`/`_apply_kyc_policy`.

## 10. Reflection & Self-Healing

`src/agents/reflector.py` classifies the last `reflection_log` entry into
one of three actions, bounded by `MAX_RETRIES = 2`:

```text
RETRYABLE  (MCP outage, LLM/tool failure)        → retry the same stage
REPLAN     (missing/invalid application payload)  → re-enter an earlier stage
ESCALATE   (KYC manual review, thin-file,
            retry budget exhausted)               → human officer, graph ends
```

Retry routing is prerequisite-safe — a retry never skips KYC to reach
credit assessment. Two compliance-relevant events (a KYC-fail referral, a
detected prompt-injection attempt) are recorded in a **separate**
`compliance_flags` channel rather than `reflection_log`, so they can never
be misread by this classifier as "the current failure" for an unrelated,
later retry — see [`tests/test_compliance_events.py`](tests/test_compliance_events.py).

Committed trace: [`evidence/reflection_trace.json`](evidence/reflection_trace.json).

## 11. Testing

```bash
pytest -q
```

```text
112 collected — 108 passed, 4 skipped, 0 failed
```

The 4 skipped tests are the explicitly gated live-provider suites (below) —
they skip by design in the default run so the deterministic suite stays
fast and hermetic. `evidence/test_run_summary.json` is regenerated on every
`pytest` invocation.

```bash
# Real Gemini + Groq-fallback path
PHASE3_LIVE=1 pytest -q tests/test_phase3_live.py

# Real agentic-RAG retrieval path
PHASE5_LIVE=1 pytest -q tests/test_phase5_live.py
```

**Network note:** `lending_policy_search` and the semantic-memory tests
download a Sentence-Transformers model from Hugging Face on first use. In a
network-restricted environment those specific tests may fail or hang on
that download — this is an environment limitation, not a code defect; they
pass cleanly with normal internet access (validated on macOS with 108/108
non-skipped tests passing).

Notable test files beyond straightforward AC coverage:

| Test file | What it proves |
|---|---|
| `tests/test_supervisor_resume.py` | The supervisor's resume-aware dispatch, and that a completed thread resumes to `END` without re-running workers. |
| `tests/test_compliance_events.py` | `compliance_flags` populate correctly and never leak into `reflection_log`. |
| `tests/test_credit_agent.py` | An unrelated earlier retry doesn't consume the credit agent's own rationale-failure budget. |
| `tests/test_mcp_tools_caching.py` | The MCP client is built once across repeated `get_mcp_tools()` calls. |
| `tests/test_multi_turn_compression.py` | NFR-08's threshold is reachable through real, repeated graph execution — not only a synthetic message list. |

## 12. Evidence

```text
evidence/
├── acceptance_evidence_manifest.json   AC-NN → test/artifact map
├── checkpoint_pause.json / checkpoint_resume.json   AC-05
├── ac06_same_session_memory.json                    AC-06
├── cross_session_memory.log                         AC-07
├── memory_write.json
├── mcp_transcript.json                              AC-09 / AC-10
├── reflection_trace.json                            AC-12
├── multi_turn_compression.json                      NFR-08 reachability
├── observability_redaction.json / observability_runtime.jsonl   NFR-04 / NFR-05
├── test_run_summary.json / phase3_live_test_summary.json / phase5_live_test_summary.json
└── run_transcripts/
```

Every artifact here is regenerated by running `pytest`, `python cli.py`, or
the gated live suites — nothing in `evidence/` is hand-edited.
`tests/test_evidence_artifacts.py` asserts none of it leaks a secret, an
email, or a raw applicant identifier.

## 13. Configuration

```bash
cp .env.example .env
```

```text
GOOGLE_API_KEY=...   # required — primary provider
GROQ_API_KEY=...     # backup provider
```

`.env` is git-ignored; only `.env.example` (no real values) is committed.
Runtime tuning (models, timeouts, memory/RAG persistence paths, logging)
lives in `config.yaml`.

## 14. Requirements → Rubric Mapping

Each graded category in `specs/loan_origination_spec.md` §7 maps to
concrete files/evidence below, so a reviewer can verify every mark without
guessing where to look.

| Category (marks) | Satisfied by |
|---|---|
| **7.1 Business & Requirements (10)** | [`docs/business-case.md`](docs/business-case.md); AC-NN table in `specs/loan_origination_spec.md`; [`docs/single-vs-multi-agent.md`](docs/single-vs-multi-agent.md) |
| **7.2 Agent Architecture & LangGraph (24)** | `src/state/schema.py` (typed state); `src/graph/build_graph.py` (topology, conditional edges); `src/memory/checkpointer.py` (pause/resume); Pydantic validation at every node boundary |
| **7.3 Patterns & Multi-Agent (18)** | Supervisor pattern (§3); `evidence/run_transcripts/strong_applicant.json`; reflection/self-healing loop (§10) with `evidence/reflection_trace.json` |
| **7.4 Context Engineering (12)** | [`docs/context-engineering.md`](docs/context-engineering.md); `src/context/{select,compress,quarantine}.py`; `tests/test_quarantine.py`, `tests/test_multi_turn_compression.py` |
| **7.5 Memory Systems (14)** | [`docs/memory-policy.md`](docs/memory-policy.md); `evidence/cross_session_memory.log`; `src/memory/eviction.py` |
| **7.6 MCP & Interoperability (14)** | §6 above; [`docs/mcp-integration-decision.md`](docs/mcp-integration-decision.md); `evidence/mcp_transcript.json` |
| **7.7 Agentic RAG & Reproducibility (8)** | §9 above; this Quick Start (§2); single-command execution; `.env.example` with no secrets |

## 15. Acceptance Criteria Traceability

The full AC-01…AC-12 / NFR-01…NFR-08 table (evidence artifact + test per
identifier) is maintained in [`docs/traceability.md`](docs/traceability.md)
and `evidence/acceptance_evidence_manifest.json` — kept as a single
source of truth rather than duplicated here.

## 16. Project Structure

```text
.
├── cli.py                     # required single-command entry point
├── app.py                     # optional Streamlit visual console
├── config.yaml                # runtime configuration
├── requirements.txt
├── .env.example
├── pytest.ini
├── mcp_server/
│   └── server.py               # applicant_lookup, bureau_check, lending_policy_search, credit_policy_manual
├── src/
│   ├── state/schema.py
│   ├── graph/{supervisor,routing,build_graph}.py
│   ├── agents/{intake,kyc,credit,offer}_agent.py, reflector.py, memory_agent.py
│   ├── llm/gateway.py
│   ├── context/{select,compress,quarantine,middleware}.py
│   ├── memory/{checkpointer,long_term_store,eviction,runtime}.py
│   ├── rag/{agentic_rag,policy_store}.py
│   ├── mcp_client.py
│   ├── async_utils.py
│   └── observability/{audit_log,redaction}.py
├── tests/                       # deterministic suite + gated live suites
├── sample_inputs/                # committed synthetic applications
├── synthetic_data/                # synthetic bureau/watchlist/policy datasets
├── evidence/                       # committed, regenerated run/trace evidence
├── docs/                             # design records — see §20
└── specs/
    └── loan_origination_spec.md       # requirements baseline (AC-NN / NFR-NN)
```

## 17. Documentation Index

| Doc | Covers |
|---|---|
| [`docs/business-case.md`](docs/business-case.md) | Problem, actors, success metrics (rubric §7.1) |
| [`docs/single-vs-multi-agent.md`](docs/single-vs-multi-agent.md) | Orchestration pattern + framework choice rationale (NFR-06) |
| [`docs/agent-contracts.md`](docs/agent-contracts.md) | Per-agent I/O, tools, LLM role, deterministic gate thresholds |
| [`docs/context-engineering.md`](docs/context-engineering.md) | Write/select/compress/isolate implementation detail |
| [`docs/memory-policy.md`](docs/memory-policy.md) | Tiered memory, importance/TTL eviction formula, storage isolation |
| [`docs/mcp-integration-decision.md`](docs/mcp-integration-decision.md) | Why a custom MCP server over alternatives (rubric §7.6) |
| [`docs/llm-provider-policy.md`](docs/llm-provider-policy.md) | Primary/fallback gateway behavior and failure handling |
| [`docs/deviations.md`](docs/deviations.md) | The one approved spec deviation (Groq backup), with scope and approval |
| [`docs/traceability.md`](docs/traceability.md) | Full AC-01…12 / NFR-01…08 → evidence/test mapping |
| [`docs/PHASES.md`](docs/PHASES.md) | Build-phase history and final validation commands |
| [`specs/loan_origination_spec.md`](specs/loan_origination_spec.md) | The requirements baseline itself |
