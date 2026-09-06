# Business Case — Loan Origination Copilot

## Problem

Manual loan origination is slow (3–5 days), inconsistent across officers,
and expensive. Applicants drop off when the process is opaque. Compliance
risk increases when KYC and credit-policy checks are applied inconsistently
between officers or under time pressure.

## Proposed Solution

An AI copilot that captures the application, runs KYC and credit checks
against policy, and drafts an indicative offer or human-referral note —
carrying context across a multi-turn, potentially multi-session
application. Every lending decision (KYC status, credit decision, offer
bounds) is made by a deterministic Python policy gate reading trusted MCP
facts; the LLM's role is limited to extraction, rationale, and constrained
generation inside those bounds. This separation is what makes the system's
decisions consistent and auditable regardless of which LLM answered — see
`docs/agent-contracts.md`.

The mandatory application path is a local CLI (`python cli.py`). A
Streamlit interface (`streamlit run app.py`) provides an optional visual
execution console over the same graph — routing, stage status, memory, and
outcomes — for demonstration purposes.

## Actors

| Actor | Role |
|---|---|
| Loan Applicant | Submits application; receives an indicative offer or referral |
| Intake Agent | Extracts and validates the applicant profile from raw input; quarantines untrusted free text |
| KYC Agent | Screens the applicant against synthetic watchlist data; classifies the outcome |
| Credit Assessment Agent | Applies synthetic bureau data and deterministic policy gates to reach a decision |
| Offer Draft Agent | Produces an indicative offer, or a decline/referral note, within Python-enforced pricing bounds |
| Human Loan Officer | Reviews escalated, thin-file, unknown-applicant, or manual-underwriting cases |
| Compliance Officer | Reviews KYC-fail referrals and flagged prompt-injection attempts |

## Success Metrics

| Metric | Target |
|---|---|
| End-to-end origination time (auto-eligible path) | Under 2 minutes |
| KYC gate accuracy (synthetic test suite) | 100% on committed test cases |
| Credit decision consistency | Deterministic and reproducible for identical trusted inputs, independent of LLM provider |
| Human escalation rate | Expected and correct for thin-file, KYC-fail, unknown-applicant, and low-confidence cases |
| Prompt-injection incidents affecting a decision | 0 (quarantine enforced; detections are logged for compliance, never acted on) |
| Cross-session fact recall | Proven by a committed memory test and its output log |
| Recovery from transient tool/model failure | Bounded retry/replan, with explicit escalation rather than a silent wrong decision |

## Scope Boundaries

**In scope:** intake, KYC check, credit assessment, indicative offer,
tiered memory, agentic-RAG policy lookup, context quarantine, MCP
interoperability, CLI execution, and an optional Streamlit visualization.

**Out of scope:** real bureau connectivity, loan disbursement, production
authentication, real customer data, and real transaction execution.

## Data

All applicant, bureau, policy, and application test data is synthetic. No
real applicant, bureau, account, or confidential financial data is used
anywhere in the system. See `synthetic_data/` and `sample_inputs/`.
