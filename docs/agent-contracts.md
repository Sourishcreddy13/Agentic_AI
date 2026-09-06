# Agent Contract Table

## Core Architectural Principle

> **LLMs generate interpretations, explanations, and structured recommendations.
> Deterministic Python code and trusted MCP tools enforce hard lending policy,
> state transitions, eligibility gates, and safety constraints.**

This is the primary design boundary of the system and is enforced at node
handoff boundaries, not merely documented. §"Enforcement Table" below lists
exactly which state field each agent contract forbids the LLM from setting.

---

## Agent Contracts

### Supervisor

| | |
|---|---|
| **Type** | Pure Python router |
| **Inputs** | `LoanApplicationState` |
| **Tools** | None |
| **LLM** | No |
| **Produces** | `next_node` |
| **Owns** | Resume-aware entry-hop dispatch |

The supervisor inspects how much of the application is already populated in
state (`applicant`, `kyc_result`, `credit_assessment`, `offer`) and decides
the correct next hop — `intake` → `kyc_check` → `credit_assessment` →
`offer_draft` → `END`. This decision is consulted by a **real conditional
edge** (`route_after_supervisor` in `src/graph/routing.py`), not a static
`add_edge`, so a thread with partial or complete progress already
checkpointed dispatches straight to its correct stage instead of restarting
at intake on every invocation. A completed application receiving a
follow-up message resumes to `END` without re-running any worker — see
`tests/test_supervisor_resume.py`.

### Intake Agent

| | |
|---|---|
| **Type** | LLM extraction + deterministic guard |
| **Inputs** | Raw application message, after structured field parsing |
| **Tools** | None |
| **LLM** | Yes — structured extraction into `ApplicantProfile` |
| **Produces** | `ApplicantProfile` (Pydantic-validated) |
| **Owns** | Quarantine of applicant-submitted free text; rejection of any LLM-mutated trusted field |

Only trusted structured fields (`applicant_id`, `full_name`, `dob_synthetic`,
`declared_income`, `declared_employment`) reach the extraction prompt.
`raw_free_text_notes` is wrapped in an explicit quarantine envelope and
never enters a model prompt. After extraction, Python re-compares every
trusted field against the LLM's output and raises if the model changed one
— the LLM extracts, it does not get to overwrite ground truth.

If the free text matches a known prompt-injection pattern, a
`ComplianceEvent` (`suspected_prompt_injection_in_free_text`) is appended to
`state["compliance_flags"]` for audit visibility. Routing and extraction are
unaffected — see "Compliance Events" below.

### KYC Agent

| | |
|---|---|
| **Type** | MCP fact + deterministic policy gate + LLM explanation |
| **Inputs** | `ApplicantProfile` |
| **Tools** | `applicant_lookup` (deterministic — MCP) |
| **LLM** | Yes — generates `rationale`/`confidence` only; Python sets `status` |
| **Produces** | `KYCResult` (Pydantic-validated) |
| **Owns** | KYC classification |
| **Python gate** | Synthetic watchlist hit → `fail`; unknown applicant → `manual_review`; otherwise → `pass`. The LLM cannot override `status`. |

A `status == "fail"` also appends a `ComplianceEvent`
(`kyc_fail_referral`) to `compliance_flags` — the routing decision itself
(straight to a declined-offer referral, never to credit assessment) is
still made entirely by `route_after_kyc`.

### Credit Assessment Agent

| | |
|---|---|
| **Type** | MCP fact + Python gate + LLM rationale |
| **Inputs** | `ApplicantProfile` |
| **Tools** | `bureau_check` (deterministic — MCP); `lending_policy_search` (agentic — optional) |
| **LLM** | Yes — generates `rationale` and `confidence` only |
| **Produces** | `CreditAssessment` (Pydantic-validated) |
| **Python gate** | Thin-file: `dti > 0.60` → `decline`, otherwise `manual_underwriting`. Score `< 600` → `manual_underwriting`. Non-thin standard `dti > 0.40` → `decline`. Premium (`income > 75k`) allows `dti` up to `0.50`; above that → `manual_underwriting`. **The LLM never sets `decision`.** |
| **Retry budget** | The one-stub-then-escalate fallback for a rationale-generation failure is counted from *this node's own* prior `llm_rationale_failure` entries in `reflection_log`, not the graph-wide `retry_count` — an unrelated earlier failure elsewhere in the run cannot silently consume this node's budget. See `tests/test_credit_agent.py`. |

### Offer Draft Agent

| | |
|---|---|
| **Type** | Python guard + LLM draft + Python constraint enforcement |
| **Inputs** | `ApplicantProfile`, `CreditAssessment` |
| **Tools** | `lending_policy_search` (agentic — optional pricing context) |
| **LLM** | Yes — drafts `OfferDraft` inside Python-provided bounds |
| **Produces** | `OfferDraft` (Pydantic-validated + constraint-clipped) |
| **Python gate** | `decision != approve` → zero-principal referral note, no LLM call. Otherwise, bounds follow the score-tier PRICE-001 table (`_offer_constraints`); the LLM's proposed `principal`/`apr`/`term_months` is clipped post-generation by `_enforce_constraints`, which also overwrites `conditions` with the three mandatory disclosure clauses. |

### Reflector

| | |
|---|---|
| **Type** | Pure Python failure classifier |
| **Inputs** | `LoanApplicationState` (`reflection_log`, `retry_count`) |
| **Tools** | None |
| **LLM** | No |
| **Produces** | `ReflectionNote`, increments `retry_count` |
| **Failure taxonomy** | `RETRYABLE` → retry the same stage; `REPLAN` → re-enter an earlier stage; `ESCALATE` → human officer, graph ends. Bounded by `MAX_RETRIES = 2`. |

The reflector classifies **only** the last `reflection_log` entry. Two
compliance-relevant events are deliberately kept out of that log entirely
(see "Compliance Events" below) so they can never be misread as "the
current failure" for an unrelated, later retry.

### Memory Consolidation

| | |
|---|---|
| **Type** | Post-decision extraction + persistence |
| **Inputs** | `LoanApplicationState` after `offer_draft` |
| **Tools** | Chroma write (`ChromaMemoryStore.upsert_facts`) |
| **LLM** | Preferred: LangMem structured extraction; falls back to the shared Gemini/Groq gateway |
| **Produces** | `MemoryFact` records (fact_id, fact_type, value, importance, timestamps) |
| **Owns** | Durable, cross-session fact persistence and TTL/importance eviction |

Failure here is fail-open: a memory-write failure never alters an
already-determined KYC, credit, or offer outcome. See `docs/memory-policy.md`.

---

## Compliance Events

`state["compliance_flags"]` (a separate `Annotated[list[ComplianceEvent], add]`
channel — `src/state/schema.py`) records audit-relevant events that must
stay visible for compliance review without ever being able to influence the
reflector's retry/replan/escalate classification:

| Event | Raised by | Detail |
|---|---|---|
| `suspected_prompt_injection_in_free_text` | Intake agent | Applicant free text matched a known injection pattern; quarantined, routing unaffected. |
| `kyc_fail_referral` | KYC agent | A KYC failure was referred to the compliance officer with a declined offer; no credit assessment run. |

Both used to be written into `reflection_log` as `ReflectionNote`s with
`action_taken="escalate_to_human"` — which risked a later, unrelated
failure being misclassified if one of these notes happened to be last in
the log when the reflector next ran. Moving them to `compliance_flags`
removed both trigger names from the reflector's `ESCALATE` set entirely.
See `tests/test_compliance_events.py`.

---

## Enforcement Table

| Field | Set by | Never set by |
|---|---|---|
| `kyc_result.status` | Python (`_apply_kyc_policy`) | Gemini / Groq |
| `credit_assessment.decision` | Python (`_apply_policy_gates`) | Gemini / Groq |
| `offer.principal` / `apr` / `term_months` | Python-enforced ceiling (`_enforce_constraints`) | Gemini / Groq (proposes a draft only) |
| `next_node` / graph routing | Python (`src/graph/routing.py`) | Gemini / Groq |
| `*.rationale`, offer `conditions` copy | Gemini / Groq | — |

---

## Tool Classification

| Tool | Type | Called by | Invocation |
|---|---|---|---|
| `applicant_lookup` | **Deterministic** — required workflow dependency | KYC agent | Required |
| `bureau_check` | **Deterministic** — required workflow dependency | Credit agent | Required |
| `lending_policy_search` | **Agentic** — discretionary RAG | Credit / Offer agent | Only when the model decides retrieval is useful |

The distinction matters for AC-11: `lending_policy_search` is agentic
because the model decides whether to call it; retrieval is not a mandatory
graph step. Zero tool calls is a valid, tested outcome.

---

## Identity Model

```text
user_id   → long-term memory key (survives across threads/sessions)
thread_id → checkpoint key (one application instance)

user_id="U-001"
   ├── thread-A  (one application instance)
   └── thread-B  (later application instance)
```

The CLI and Streamlit interfaces generate a fresh `thread_id` for a new
application run unless an explicit resume workflow is being used. Reusing a
`thread_id` for a second, distinct application (e.g. a resubmission) is
exactly the scenario the supervisor's resume-aware routing and the
long-thread compression path (NFR-08) are designed around — see
`tests/test_multi_turn_compression.py`.
