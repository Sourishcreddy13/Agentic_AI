# 3. Problem Statement & Expected Solution

## 3.1 Problem
A bank wants to accelerate loan origination. When an applicant submits a loan request, a copilot should capture the application, run KYC checks, assess creditworthiness against policy, and draft an indicative offer or referral for a human officer — carrying context across a multi-turn, potentially multi-session application. You build this as a LangGraph multi-agent system with a custom MCP tool server, engineered context, and tiered memory.

## 3.2 Your Role
Agentic AI Engineer. You design the agent graph and state, choose and justify single-vs-multi-agent orchestration, build a custom MCP server and wire it into the agent, engineer context (write / select / compress / isolate), and implement a tiered memory that survives across sessions — all evidenced by committed traces and tests.

## 3.3 Expected Solution
A working multi-agent application delivered as a Git repository that demonstrates:
* A LangGraph graph with a typed state object, a supervisor routing a loan application to specialized worker agents (intake, KYC-check, credit-assessment, offer-draft), and conditional edges driven by state.
* A custom MCP server exposing at least 2 tools and 1 resource, consumed by the agent via langchain-mcp-adapters, with a committed run transcript showing tool invocation.
* Engineered context (write / select / compress / isolate) with summarization middleware and quarantine of untrusted applicant-submitted text.
* A tiered memory layer whose cross-session persistence is proven by a committed test and its output log, plus an eviction / importance policy.
* An agentic-RAG tool the agent decides to call for lending-policy and eligibility lookups, and a reflection / self-healing loop with an evidenced trace.

## 3.4 Applicable Rules
* **Synthetic-Data Rule.** Use only synthetic / dummy data you generate yourself — no real applicant, bureau, or account data, and no Virtusa confidential data.
* **Evidence-in-Repo Rule.** Only committed artifacts are scored. Run transcripts, tool-call logs, the memory-persistence test and its output, and traces must be committed — uncommitted behavior does not count.
* **Reproducibility Rule.** The system must run from a single documented command with committed sample application inputs and a README quick-start.
* **AC-Traceability Rule.** Each Acceptance Criterion must be referenced by at least one test or committed evidence artifact carrying its AC-NN identifier.
* **Context-Isolation Rule.** Free-text applicant-submitted content is untrusted; it must be isolated (quarantined) and never treated as instructions to the agent.
* **Open-Source & No-Docker Rule.** Use the approved open-source stack with Google Gemini as the LLM provider. The project must build, run, and be evaluated with pip + Python alone — no Docker and no external database service. *A configured backup API provider (currently Groq), used solely for graceful degradation on primary-provider failure, is an administrator-approved deviation from this rule — see `docs/deviations.md` (DEV-001) for scope and approval, rather than treating it as part of the baseline requirement.*

---

# 4. Technology & Framework Stack

The stack is fixed to an open-source toolchain with Google Gemini as the model provider. The project must build, run, and be evaluated with pip + Python alone — no Docker and no external database service. (A configured backup API, used for graceful degradation on primary-provider failure, is an approved deviation — see `docs/deviations.md`, DEV-001.)

| Layer | Approved tool (open source unless noted) |
| :--- | :--- |
| Language | Python 3.11+ |
| Agent Framework | LangGraph (MIT, required); CrewAI optional for the single-vs-multi comparison |
| LLM Provider | Google Gemini (API) — sole provider per spec; configured backup API permitted only as an approved deviation, see `docs/deviations.md` (DEV-001) |
| Interoperability | MCP Python SDK (stdio) + langchain-mcp-adapters (MIT) |
| Memory | langgraph-checkpoint-sqlite (SQLite file) + LangMem; Chroma / FAISS for semantic memory |
| Embeddings | Sentence-Transformers (local, open source) |
| Retrieval (tool) | Chroma or FAISS for the agentic-RAG lookup tool |
| Interface (optional) | CLI · Streamlit · Gradio · FastAPI |

---

# 5. Acceptance Criteria & Non-Functional Requirements

## 5.1 Functional Acceptance Criteria
*Note. Each AC must have at least one test or committed evidence artifact referencing its AC-NN identifier.*

| ID | Criterion |
| :--- | :--- |
| AC-01 | The system is built on LangGraph with an explicit typed state object (TypedDict / Pydantic) shared across nodes. |
| AC-02 | A supervisor / orchestrator routes a loan application to specialized worker agents (intake, KYC-check, credit-assessment, offer-draft). |
| AC-03 | The graph uses conditional edges to route on state (e.g., route thin-file applicants to manual underwriting; decline applications failing KYC). |
| AC-04 | Node / agent outputs are validated structured objects (Pydantic) at handoff boundaries. |
| AC-05 | A checkpointer persists graph state so a application case can be paused and resumed. |
| AC-06 | The copilot maintains tiered memory (short-term working + long-term / semantic) and recalls a fact from an earlier turn. |
| AC-07 | Memory persists across sessions: a committed test starts a new session and shows recall of prior-session facts; its output log is committed. |
| AC-08 | A memory eviction / importance policy (TTL, LRU, or importance-weighted) is implemented and documented. |
| AC-09 | A custom MCP server exposes ≥ 2 tools and 1 resource relevant to the domain (e.g., applicant_lookup + bureau_check + lending_policy tools; credit_policy_manual resource). |
| AC-10 | The agent consumes the MCP server via langchain-mcp-adapters; a committed transcript shows the agent invoking an MCP tool. |
| AC-11 | An agentic-RAG tool is available and the agent decides to call it for lending-policy and eligibility lookups (retrieval inside the loop, not a fixed step). |
| AC-12 | The system implements a reflection or self-healing / fallback loop (e.g., re-plan on tool failure or low-confidence output) with an evidenced trace. |

## 5.2 Non-Functional Requirements

| ID | Requirement |
| :--- | :--- |
| NFR-01 | No secrets or API keys committed; .env-var configuration with a committed .env.example. |
| NFR-02 | System runs end-to-end from a single documented command with committed sample application inputs and a README quick-start. |
| NFR-03 | Untrusted free-text applicant-submitted content is isolated (context quarantine) and never trusted as instructions. |
| NFR-04 | Structured JSON logs / traces of agent runs are committed as evidence. |
| NFR-05 | All data is synthetic; any PII is synthetic and never written to logs in plaintext. |
| NFR-06 | The single-vs-multi-agent decision and the framework choice are documented with rationale. |
| NFR-07 | Graceful degradation on tool / model failure: timeouts, retries, and explicit exit conditions. |
| NFR-08 | Context-window management: summarization / compression is applied for long threads. |

---

# 6. Functional Scope

## 6.1 In Scope
* A LangGraph multi-agent graph: typed state, supervisor + worker agents, conditional routing, checkpointing, structured output.
* A custom MCP server (≥ 2 tools + 1 resource) integrated into the agent and exercised in a committed transcript.
* Context engineering (write / select / compress / isolate), summarization middleware, and quarantine of untrusted applicant-submitted text.
* Tiered memory with verified cross-session persistence and an eviction / importance policy.
* An agentic-RAG lending-policy and eligibility lookups tool and a reflection / self-healing loop.
* A minimal interface (CLI / Streamlit / Gradio / FastAPI) to drive a loan application through the graph.

## 6.2 Out of Scope
* Real bureau connectivity, credit scoring, or loan disbursement.
* Real customer or confidential data of any kind.
* Production deployment, multi-region, authentication, and real back-end system integration.
* Front-end visual polish and any real transaction / action execution — heuristics and stubs are sufficient.

---

# 7. Implementation Expectations

The rubric scores committed evidence. Each category below states what must exist in the repository. Marks per category are shown in Section 9.

## 7.1 Business & Requirements (10 marks)
* docs/business-case.md covering problem, actors, and success metrics for the loan-origination workflow.
* Acceptance criteria in testable form (AC-NN).
* A documented single-vs-multi-agent decision and framework-choice rationale.

## 7.2 Agent Architecture & LangGraph (24 marks)
* A typed state object (TypedDict / Pydantic) shared across nodes.
* A graph topology with a supervisor and specialized worker agents (nodes + edges).
* Conditional routing / decision edges driven by state.
* A configured checkpointer for pause / resume; structured output validated at node boundaries.

## 7.3 Patterns & Multi-Agent (18 marks)
* An implemented agent pattern (ReAct / plan-execute / reflection) with a short rationale.
* Working multi-agent orchestration (supervisor or swarm) evidenced by a committed run transcript.
* A reflection or self-healing / fallback loop with an evidenced trace.

## 7.4 Context Engineering (12 marks)
* Explicit write / select / compress / isolate strategies mapped in a short doc.
* Summarization / compression middleware for long threads.
* Context quarantine isolating untrusted applicant-submitted text.

## 7.5 Memory Systems (14 marks)
* Tiered memory (short-term working + long-term / semantic).
* Cross-session persistence proven by a committed test and its output log.
* An eviction / importance policy (TTL / LRU / importance-weighted).

## 7.6 MCP & Interoperability (14 marks)
* A custom MCP server exposing ≥ 2 tools and 1 resource.
* Integration via langchain-mcp-adapters with a committed tool-call transcript.
* An integration-decision writeup (MCP vs API vs direct-DB vs A2A).

## 7.7 Agentic RAG & Reproducibility (8 marks)
* An agentic-RAG tool the agent calls on demand for lending-policy and eligibility lookups.
* Engineering hygiene: README quick-start, single-command run, no secrets, committed traces.

---

# 8. Expected Outcomes & Deliverables

By the end of 15 hours, the submitted Git repository must contain the Mandatory items below. Good-to-Have items differentiate Merit and Distinction submissions.

## 8.1 Mandatory
* Working multi-agent system runnable locally via a single command, with committed sample application inputs and README quick-start.
* docs/business-case.md and specs with AC-NN acceptance criteria.
* LangGraph graph: typed state, supervisor + workers, conditional routing, checkpointer, structured output.
* Custom MCP server (≥ 2 tools + 1 resource) + adapter integration + committed tool-call transcript.
* Context engineering (write/select/compress/isolate) + summarization middleware + quarantine of untrusted applicant-submitted text.
* Tiered memory + committed cross-session persistence test and log + eviction policy.
* Agentic-RAG tool + reflection/self-healing loop with trace; .env.example; no committed secrets.
* PR-driven Git history: at least 3 PR-driven merges (`git merge --no-ff`); no direct pushes to main.

## 8.2 Good-to-Have
* A single-vs-multi-agent comparison run (supervisor variant vs single-agent variant) with observations.
* A second MCP server (filesystem / database) or an A2A demonstration.
* Importance-weighted memory with semantic recall over Chroma / FAISS / LangGraph Store.
* A policy-exception path that routes edge cases to the approval-authority matrix.
* A lightweight UI showing the graph's routing decisions and memory state.
