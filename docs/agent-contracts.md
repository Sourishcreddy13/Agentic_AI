# Agent Contract Table

## Core Architectural Principle

> **LLMs generate interpretations, explanations, and structured recommendations.
> Deterministic Python code and trusted MCP tools enforce hard lending policy,
> state transitions, eligibility gates, and safety constraints.**

This is the primary design boundary of the system and is enforced at node
handoff boundaries.

---

## Agent Contracts

### Supervisor

| | |
|---|---|
| **Type** | Pure Python router |
| **Inputs** | `LoanApplicationState` |
| **Tools** | None |
| **LLM** | No |
| **Produces** | `next_node` for entry routing |
| **Owns** | Lifecycle entry and first-hop dispatch |

### Intake Agent

| | |
|---|---|
| **Type** | LLM extraction |
| **Inputs** | Raw application message after structured parsing |
| **Tools** | None |
| **LLM** | Yes — structured extraction into `ApplicantProfile` |
| **Produces** | `ApplicantProfile` (Pydantic-validated) |
| **Owns** | Quarantine of applicant-submitted free text; only trusted structured fields reach the extraction prompt |

### KYC Agent

| | |
|---|---|
| **Type** | MCP fact + deterministic policy gate + LLM explanation |
| **Inputs** | `ApplicantProfile` |
| **Tools** | `applicant_lookup` (deterministic — MCP) |
| **LLM** | Yes — generates rationale/confidence only; Python sets status |
| **Produces** | `KYCResult` (Pydantic-validated) |
| **Owns** | KYC classification; graph routing is controlled by conditional edges |
| **Python gate** | Watchlist hit → fail; unknown applicant → manual review; LLM cannot override status |

### Credit Assessment Agent

| | |
|---|---|
| **Type** | MCP fact + Python gate + LLM rationale |
| **Inputs** | `ApplicantProfile` |
| **Tools** | `bureau_check` (deterministic — MCP); `lending_policy_search` (agentic — optional) |
| **LLM** | Yes — generates `rationale` and `confidence` only |
| **Produces** | `CreditAssessment` (Pydantic-validated) |
| **Python gate** | Thin-file: `dti > 0.60` → decline, otherwise manual underwriting; score < 600 → manual underwriting; non-thin standard `dti > 0.40` → decline; premium (`income > 75k`) allows up to `dti = 0.50`; `dti > 0.50` → manual underwriting. **LLM never sets `decision`.** |

### Offer Draft Agent

| | |
|---|---|
| **Type** | Python guard + LLM draft + Python constraint enforcement |
| **Inputs** | `ApplicantProfile`, `CreditAssessment` |
| **Tools** | `lending_policy_search` (agentic — optional for pricing context) |
| **LLM** | Yes — drafts `OfferDraft` within Python-provided bounds |
| **Produces** | `OfferDraft` (Pydantic-validated + constraint-clipped) |
| **Python gate** | `decision != approve` → zero-principal referral note, no LLM call. Offer bounds follow score-tier policy (PRICE-001). LLM output is clipped post-generation. |

### Reflector

| | |
|---|---|
| **Type** | Pure Python failure classifier |
| **Inputs** | `LoanApplicationState` (`reflection_log`, `retry_count`) |
| **Tools** | None |
| **LLM** | No |
| **Produces** | `ReflectionNote`, increments `retry_count` |
| **Failure taxonomy** | `RETRYABLE` → retry; `REPLAN` → re-enter an earlier stage; `ESCALATE` → human officer |

---

## Tool Classification

| Tool | Type | Called by | Invocation |
|---|---|---|---|
| `applicant_lookup` | **Deterministic** — required workflow dependency | KYC agent | Required |
| `bureau_check` | **Deterministic** — required workflow dependency | Credit agent | Required |
| `lending_policy_search` | **Agentic** — discretionary RAG | Credit / Offer agent | When the model decides retrieval is useful |

The distinction matters for AC-11: `lending_policy_search` is agentic because the
model decides whether to call it; retrieval is not a mandatory graph step.

---

## Identity Model

```text
user_id   → long-term memory key (survives across threads/sessions)
thread_id → checkpoint key (one application instance)

user_id="U-001"
   ├── thread-A  (one application instance)
   └── thread-B  (later application instance)
```

The CLI and Streamlit interfaces should generate a fresh `thread_id` for a new
application run unless an explicit resume workflow is being used.
