# Business Case — Loan Origination Copilot

## Problem

Manual loan origination is slow (3–5 days), inconsistent across officers,
and expensive. Applicants drop off when the process is opaque. Compliance
risk increases when KYC and credit-policy checks are applied inconsistently.

## Proposed Solution

An AI copilot that captures the application, runs KYC and credit checks
against policy, and drafts an indicative offer or human-referral note —
carrying context across a multi-turn, potentially multi-session application.

The mandatory application path is a local CLI. A Streamlit interface provides
an optional visual execution console for graph routing, stage status, memory,
events, and outcomes.

## Actors

| Actor | Role |
|---|---|
| Loan Applicant | Submits application; receives indicative offer or referral |
| Intake Agent | Extracts and validates applicant profile from raw input |
| KYC Agent | Screens applicant against synthetic watchlist data; classifies outcome |
| Credit Assessment Agent | Applies synthetic bureau data and policy gates to decision |
| Offer Draft Agent | Produces indicative offer or decline/referral note |
| Human Loan Officer | Reviews escalated, thin-file, unknown-applicant, or manual-underwriting cases |
| Compliance Officer | Reviews KYC-fail referrals |

## Success Metrics

| Metric | Target |
|---|---|
| End-to-end origination time (auto-eligible) | < 2 minutes |
| KYC gate accuracy (synthetic test suite) | 100% on committed test cases |
| Credit decision consistency | Deterministic for identical trusted inputs |
| Human escalation rate | Expected for thin-file, KYC-fail, unknown-applicant, and low-confidence cases |
| Prompt-injection incidents | 0 (quarantine enforced) |
| Cross-session fact recall | Proven by committed memory test |

## Scope Boundaries

In scope: intake, KYC check, credit assessment, indicative offer,
tiered memory, agentic-RAG policy lookup, context quarantine,
MCP interoperability, CLI execution, and optional Streamlit visualization.

Out of scope: real bureau connectivity, loan disbursement, production
authentication, real customer data, and real transaction execution.

## Data

All applicant, bureau, policy, and application test data is synthetic.
No real applicant, bureau, account, or confidential financial data is used
anywhere in the system.
