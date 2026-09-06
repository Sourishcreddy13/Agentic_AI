# Loan Origination Copilot

A LangGraph multi-agent copilot for bank loan origination. A supervisor dispatches a typed, checkpointed application through intake, KYC, credit assessment, and offer-drafting agents. It is backed by a custom MCP tool server, tiered memory, discretionary agentic-RAG policy lookup, and a bounded reflection/self-healing loop.

All data — applicants, bureau records, watchlist hits, the lending-policy manual — is synthetic. No real applicant, bureau, account, or confidential data is used anywhere in this repository.

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [Architecture](#2-architecture)
3. [The Core Design Principle](#3-the-core-design-principle)
4. [Agent Contracts](#4-agent-contracts)
5. [Memory & Context](#5-memory--context)
6. [Testing](#6-testing)
7. [Project Structure](#7-project-structure)

---

## 1. Quick Start

```bash
# 1. Create and activate a Python 3.11+ virtual environment
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

> **Tip:** If you have [`uv`](https://docs.astral.sh/uv/) installed, you can use:
> `uv venv --python 3.11 .venv && source .venv/bin/activate && uv pip install -r requirements.txt`

```bash
# 3. Configure provider credentials
cp .env.example .env
# Edit .env and set GOOGLE_API_KEY (required) and GROQ_API_KEY (fallback)

# 4. Run the required single-command application
python cli.py
```

**Run against a specific synthetic application:**
## Usage
Provide short, scannable examples of the software running in production or development.

If you need to add, update, or delete sample inputs, visit the folder `sample_inputs/`.
```bash
python cli.py --input sample_inputs/applicant_strong.json
python cli.py --input sample_inputs/applicant_kyc_fail.json
python cli.py --input sample_inputs/applicant_thin_file.json
```

**Optional visual interface:**
```bash
streamlit run app.py
```

## 2. Architecture

```text
                                   ┌───────────────────────┐
                                   │      supervisor       │
                                   │  (pure Python router) │
                                   └───────────┬───────────┘
                                               │  
                    ┌──────────────────────────┼──────────────────────────┬─────────┐
                    │ no applicant             │ no kyc_result            │ no      │ all set
                    ▼                          ▼                          │ credit  │  → END
              ┌──────────┐               ┌──────────┐                    │ assess  │
              │  intake  │──kyc_check───▶│kyc_check │                    ▼         │
              └────┬─────┘               └────┬─────┘             ┌─────────────┐  │
                   │ reflector                 │ MCP:             │  credit_    │  │
                   │ (parse/LLM failure)       │ applicant_lookup │ assessment  │◀─┘
                   ▼                           │                  └──────┬──────┘
              ┌──────────┐                     │ fail → offer_draft      │ MCP: bureau_check
              │reflector │◀────────────────────┤ manual/low-conf         │ + agentic RAG
              └────┬─────┘                     ▼                         ▼
                   │ retry / replan       ┌──────────┐            manual/low-conf → reflector
                   │                      │  (pass)  │                   │ decline/approve
                   └── escalate_to_human  └────┬─────┘                   ▼
                                               └──────────────────▶┌─────────────┐
                                                                   │ offer_draft │
                                                                   └──────┬──────┘
                                                                          ▼
                                                                memory_consolidation
                                                                          ▼
                                                                         END
```

The supervisor utilizes a conditional edge router, allowing threads with partial or complete progress to dispatch straight to their next required stage instead of restarting at intake. 

## 3. The Core Design Principle

**LLMs generate interpretations, explanations, and structured recommendations. Deterministic Python code and trusted MCP tool results enforce every hard lending-policy decision, state transition, eligibility gate, and safety constraint.**

This is enforced at every node boundary:

| Field | Set by | Never set by |
|---|---|---|
| `kyc_result.status` | Python (`_apply_kyc_policy`) | LLM |
| `credit_assessment.decision` | Python (`_apply_policy_gates`) | LLM |
| `offer.principal` / `apr` / `term` | Python-enforced ceiling | LLM |
| `next_node` / routing | Python conditional edges | LLM |
| `kyc_result.rationale` | LLM | — |

## 4. Agent Contracts

| Agent | Type | Tools | LLM Role |
|---|---|---|---|
| **Supervisor** | Pure Python router | None | None |
| **Intake** | LLM extraction + guard | None | Structured extraction from trusted fields |
| **KYC** | MCP fact + Python gate | `applicant_lookup` | Rationale + confidence |
| **Credit** | MCP fact + Python gate | `bureau_check`, `policy_search` | Rationale + confidence |
| **Offer Draft**| Python guard + LLM draft | `policy_search` | Drafts offer inside Python bounds |
| **Reflector** | Pure Python classifier | None | None |
| **Memory** | Extraction + persistence | Chroma write | Fact extraction |

## 5. Memory & Context

*   **Short-term Memory:** `SqliteSaver` checkpointing, keyed by `thread_id` (application session).
*   **Long-term Memory:** Chroma + Sentence-Transformers, keyed by `user_id` (durable facts across sessions).
*   **Context Engineering:** Messages exceeding the 20-message threshold are compressed into a bounded summary. Applicant free-text is isolated in a quarantine envelope and excluded from prompts and compression to prevent injection.

## 6. Testing

The project includes a deterministic hermetic test suite and gated live-provider suites.

```bash
# Run the deterministic regression suite
pytest -q

# Real Gemini + Groq-fallback path
PHASE3_LIVE=1 pytest -q tests/test_phase3_live.py

# Real agentic-RAG retrieval path
PHASE5_LIVE=1 pytest -q tests/test_phase5_live.py
```

*Note: The semantic-memory and RAG tests will download a Sentence-Transformers model from Hugging Face on first use.*

## 7. Project Structure

```text
.
├── cli.py                     # Required single-command entry point
├── app.py                     # Optional Streamlit visual console
├── config.yaml                # Runtime configuration
├── requirements.txt
├── .env.example
├── mcp_server/
│   └── server.py              # Domain MCP server and tools
├── src/
│   ├── state/schema.py
│   ├── graph/                 # Supervisor, routing, and graph topology
│   ├── agents/                # Node implementations
│   ├── context/               # Context compression and quarantine
│   ├── memory/                # Short/long-term memory handling
│   ├── rag/                   # Agentic RAG implementation
│   └── observability/         # Audit logging and PII redaction
├── tests/                     # Deterministic and live suites
├── sample_inputs/             # Synthetic applications
└── synthetic_data/            # Synthetic datasets
```
