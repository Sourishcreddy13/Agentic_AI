# Loan Origination Copilot

A LangGraph-based multi-agent loan-origination copilot using a typed shared state, specialized worker agents, custom MCP tools/resources, engineered context, tiered memory, agentic RAG, and bounded reflection/self-healing.

The system uses synthetic data only. It is designed to demonstrate an end-to-end loan-origination workflow from application intake through KYC, credit assessment, indicative offer/referral, memory, policy retrieval, and recovery handling.

## Status

Core implementation, deterministic regression tests, live provider tests, MCP evidence, memory evidence, context-isolation tests, agentic-RAG validation, and Streamlit visualization are implemented.

The required single-command application entry point is:

```bash
python cli.py
```

The Streamlit interface is an additional visual interface:

```bash
streamlit run app.py
```

Required Git/PR history is a submission-time repository requirement.

---

## 1. Problem

The system models a bank loan-origination copilot that:

1. captures a loan application;
2. performs synthetic applicant lookup and KYC checks;
3. assesses creditworthiness against deterministic lending policy;
4. retrieves lending-policy information through agentic RAG when required;
5. drafts an indicative offer or referral;
6. maintains short-term and long-term memory;
7. carries application context across turns and sessions;
8. handles tool/model failures through bounded reflection and recovery.

The core architecture separates probabilistic LLM responsibilities from deterministic policy enforcement. The LLM is used for extraction, rationale, confidence, and constrained generation. Trusted MCP facts and Python policy gates own KYC/credit decisions, routing, and hard offer constraints.

---

## 2. Architecture

```text
                         ┌─────────────────────┐
                         │     Supervisor      │
                         └──────────┬──────────┘
                                    │
                ┌───────────────────┼───────────────────┐
                │                   │                   │
                ▼                   ▼                   ▼
           ┌─────────┐         ┌─────────┐        ┌──────────┐
           │ Intake  │         │   KYC   │        │  Credit  │
           └────┬────┘         └────┬────┘        └─────┬────┘
                │                   │                   │
                │                   │                   │
                │              MCP lookup          MCP bureau
                │                   │                   │
                │                   └──────────┬────────┘
                │                              │
                │                              ▼
                │                         Policy Gates
                │                              │
                │                    ┌─────────┴─────────┐
                │                    │                   │
                │                    ▼                   ▼
                │                 Offer Draft        Reflection
                │                    │                   │
                │                    ▼                   │
                │                   END ◄────────────────┘
                │
                ├── Context selection/compression
                ├── Context quarantine
                └── Tiered memory
```

### Main components

* `src/state/` — typed LangGraph state and Pydantic domain contracts.
* `src/graph/` — supervisor, worker nodes, conditional routing, and reflection/recovery.
* `src/agents/` — intake, KYC, credit, and offer-drafting agents.
* `src/mcp_client.py` and `mcp_server/` — custom MCP integration over stdio.
* `src/llm/gateway.py` — configured primary/fallback provider gateway.
* `src/context/` — context selection, compression, and untrusted-text quarantine.
* `src/memory/` — short-term checkpoint state and durable semantic memory.
* `src/rag/` — agentic lending-policy retrieval.
* `tests/` — deterministic and live integration tests.
* `evidence/` — committed test, trace, MCP, memory, observability, and live-run evidence.
* `synthetic_data/` — synthetic applicant and bureau datasets.
* `app.py` — optional Streamlit visualization.
* `cli.py` — required single-command application entry point.

---

## 3. Technology Stack

The project uses:

* Python 3.11+
* LangGraph
* Pydantic
* Google Gemini as the primary model provider
* Groq as the permitted configured backup provider
* MCP Python SDK
* `langchain-mcp-adapters`
* SQLite-based graph checkpointing
* Chroma semantic memory
* Sentence-Transformers/local embeddings
* Streamlit for the optional visual interface

No Docker or external database service is required.

---

## 4. Single-Command Application

The mandatory application command is:

```bash
python cli.py
```

The CLI executes the complete loan-origination graph using the existing LangGraph implementation.

An application can also be supplied explicitly:

```bash
python cli.py --input sample_inputs/applicant_strong.json
```

The CLI provides execution-oriented output including:

* application input;
* graph stages;
* node execution;
* routing decisions;
* MCP/tool activity;
* reflection/retry activity;
* final outcome;
* runtime errors.

Structured execution logging is written separately using the existing observability infrastructure.

---

## 5. Installation

Create and activate a Python 3.11+ virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Configure environment variables using `.env.example`.

Do not commit `.env` or API keys.

---

## 6. Run the Application

### Required CLI

```bash
python cli.py
```

Or with a committed synthetic application input:

```bash
python cli.py --input sample_inputs/applicant_strong.json
```

Additional synthetic application inputs are available under:

```text
sample_inputs/
```

### Optional Streamlit interface

```bash
streamlit run app.py
```

The Streamlit interface visualizes the execution graph, node status, executed path, runtime events, and final outcome. It uses the same underlying LangGraph workflow as the CLI.

---

## 7. Testing

### Deterministic regression suite

Run:

```bash
pytest
```

The default suite uses deterministic test doubles for model-dependent components and keeps ordinary tests hermetic.

The latest validated regression run is:

```text
94 tests collected
90 passed
4 skipped
0 failed
```

The skipped tests are the explicitly gated live-provider tests.

The generated regression artifact is:

```text
evidence/test_run_summary.json
```

### Phase 3 live provider tests

Run:

```bash
PHASE3_LIVE=1 pytest -q tests/test_phase3_live.py
```

These tests exercise the real provider path, including Gemini-primary execution and the configured Groq fallback path.

The generated evidence artifact is:

```text
evidence/phase3_live_test_summary.json
```

### Phase 5 live agentic-RAG test

Run:

```bash
PHASE5_LIVE=1 pytest -q tests/test_phase5_live.py
```

This validates discretionary lending-policy retrieval in the real agentic-RAG path.

The generated evidence artifact is:

```text
evidence/phase5_live_test_summary.json
```

---

## 8. Evidence

The repository contains committed evidence for the required acceptance criteria.

Key artifacts include:

```text
evidence/
├── acceptance_evidence_manifest.json
├── checkpoint_pause.json
├── checkpoint_resume.json
├── memory_write.json
├── cross_session_memory.log
├── ac06_same_session_memory.json
├── mcp_transcript.json
├── reflection_trace.json
├── observability_redaction.json
├── observability_runtime.jsonl
├── test_run_summary.json
├── phase3_live_test_summary.json
├── phase5_live_test_summary.json
└── run_transcripts/
```

### MCP evidence

`evidence/mcp_transcript.json` records:

* MCP server identity;
* stdio transport;
* `langchain-mcp-adapters` integration;
* available MCP tools;
* actual tool calls;
* the `credit_policy_manual` MCP resource;
* actual resource retrieval.

### Memory evidence

The committed memory artifacts demonstrate:

* short-term state persistence;
* same-session recall;
* cross-session persistence;
* memory writes;
* eviction/importance behavior.

### Reflection evidence

`evidence/reflection_trace.json` records bounded failure handling such as:

```text
tool failure
    ↓
reflection
    ↓
retry/replan
    ↓
successful recovery or explicit escalation
```

### Agentic-RAG evidence

The Phase 5 evidence records discretionary policy retrieval rather than a fixed retrieval step.

The retrieved policy information is advisory. Deterministic Python policy gates remain authoritative.

---

## 9. Context Engineering

The context layer implements four strategies:

### Write

Important application information is written into structured state/memory representations.

### Select

Only relevant state/context is passed to workers.

### Compress

Long conversation context is summarized using structured compression.

### Isolate

Applicant-submitted free text is treated as untrusted content.

Untrusted applicant text is quarantined and is not allowed to become model instructions.

The compression path uses a trusted projection rather than passing raw applicant free text into the summarizer.

Relevant tests include:

```text
test_compression_removes_untrusted_applicant_free_text
test_summary_projection_strips_raw_applicant_free_text
test_injection_attempt_in_application_does_not_change_routing_outcome
```

---

## 10. Memory

The project uses two memory layers:

```text
Short-term memory
    → graph/checkpoint state
    → thread/application scoped

Long-term memory
    → semantic Chroma memory
    → user scoped
```

Cross-session persistence is explicitly tested and evidenced.

The memory system also implements retention controls including TTL/importance-based eviction.

---

## 11. MCP

The custom MCP server exposes:

### Tools

```text
applicant_lookup
bureau_check
lending_policy_search
```

### Resource

```text
policy://credit_policy_manual
```

Tools are consumed through `langchain-mcp-adapters` using stdio transport.

No external database service is required.

---

## 12. Agentic RAG

The credit workflow can decide that additional lending-policy information is required.

The retrieval path is:

```text
Credit Agent
     ↓
decide whether policy retrieval is useful
     ↓
lending_policy_search
     ↓
semantic retrieval
     ↓
clause-level policy evidence
     ↓
credit reasoning
```

Retrieval is not a mandatory fixed graph step.

Retrieved policy text cannot override deterministic Python lending-policy gates.

---

## 13. Reflection and Self-Healing

The system implements a bounded deterministic recovery controller.

Examples include:

* MCP tool failure;
* low-confidence KYC;
* low-confidence credit assessment;
* retry exhaustion;
* explicit human escalation.

Recovery is bounded and preserves prerequisite constraints such as KYC before credit assessment.

---

## 14. Synthetic Data

All applicant, bureau, and application data is synthetic.

Synthetic datasets are stored under:

```text
synthetic_data/
```

Committed sample applications are stored under:

```text
sample_inputs/
```

No real applicant, bureau, customer, account, or confidential organizational data is used.

---

## 15. Configuration

Copy the example environment file and populate only the required variables locally.

```text
.env.example
```

Typical provider configuration uses:

```text
Google Gemini
    ↓ primary

Groq
    ↓ permitted backup
```

Secrets must remain in local environment variables and must never be committed.

---

## 16. Acceptance Criteria Traceability

The project tracks the acceptance criteria using `AC-NN` identifiers.

The major acceptance areas are:

```text
AC-01  Typed LangGraph state
AC-02  Supervisor + specialized workers
AC-03  Conditional state-driven routing
AC-04  Structured Pydantic handoffs
AC-05  Checkpoint pause/resume
AC-06  Same-session memory
AC-07  Cross-session persistence
AC-08  Eviction/importance policy
AC-09  MCP tools + resource
AC-10  MCP adapter/tool invocation
AC-11  Agentic RAG
AC-12  Reflection/self-healing
```

The authoritative mapping is maintained in:

```text
evidence/acceptance_evidence_manifest.json
```

and the repository specification under:

```text
specs/
```

---

## 17. Non-Functional Requirements

The implementation addresses the specified NFRs:

```text
NFR-01  Environment-based secrets; no committed API keys
NFR-02  Single-command local application execution
NFR-03  Applicant free-text context quarantine
NFR-04  Structured runtime logs/traces
NFR-05  Synthetic data and log redaction
NFR-06  Framework and orchestration rationale
NFR-07  Retry, timeout, fallback, and explicit failure handling
NFR-08  Context summarization/compression
```

---

## 18. Project Workflow

The recommended local workflow is:

```bash
# Install
pip install -r requirements.txt

# Run the application
python cli.py

# Run deterministic tests
pytest

# Run real provider integration
PHASE3_LIVE=1 pytest -q tests/test_phase3_live.py

# Run real agentic RAG
PHASE5_LIVE=1 pytest -q tests/test_phase5_live.py

# Optional visual interface
streamlit run app.py
```

---

## 19. Submission Checklist

Before submission, verify:

```text
[ ] python cli.py executes the complete application
[ ] README quick-start is up to date
[ ] committed synthetic sample inputs exist
[ ] docs/business-case.md exists
[ ] specs contain AC-NN acceptance criteria
[ ] typed LangGraph state is present
[ ] supervisor + workers are present
[ ] conditional routing is present
[ ] checkpointing is present
[ ] structured handoffs are validated
[ ] custom MCP server exposes >=2 tools + 1 resource
[ ] MCP adapter integration is evidenced
[ ] MCP transcript is committed
[ ] context write/select/compress/isolate are implemented
[ ] applicant free text is quarantined
[ ] tiered memory is implemented
[ ] cross-session persistence evidence is committed
[ ] eviction/importance policy is documented
[ ] agentic RAG is implemented and live-tested
[ ] reflection/self-healing trace is committed
[ ] .env.example is committed and contains no secrets
[ ] pytest passes with zero failures
[ ] Phase 3 live tests pass
[ ] Phase 5 live test passes
[ ] required Git PR history is present
```

---

## 20. Optional Good-to-Have Features

The following are optional differentiators rather than core mandatory requirements:

* single-agent vs multi-agent comparison run;
* second MCP server or A2A demonstration;
* importance-weighted semantic memory;
* policy-exception approval-authority path;
* Streamlit visualization of routing and memory state.

The current Streamlit interface is intended as the lightweight visual demonstration layer while `python cli.py` remains the required application execution path.
