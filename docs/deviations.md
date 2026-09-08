# Approved Deviations from the Specification

This document is the single place where a deviation from `specs/loan_origination_spec.md`
is recorded. A deviation is recorded here — with rationale, scope, and approval —
instead of being folded into the spec text itself, so the spec continues to state
the original requirement and any gap between "spec" and "implementation" stays
visible rather than silently disappearing.

## DEV-001 — Groq as the specific backup LLM provider

**Spec clause affected:** "Open-Source & No-Docker Rule" and the "LLM
Provider" row of the requirements table (`specs/loan_origination_spec.md`),
which already read *"Google Gemini (with fallback API required)"* — i.e.
the spec itself mandates a fallback provider. **The deviation recorded here
is narrower than "a fallback exists": it is specifically the choice of
Groq as that fallback**, a proprietary hosted-inference API, rather than
an alternative reading of "the approved open-source stack" (e.g. a
self-hosted open-weights model, or a second Gemini-compatible endpoint).
An earlier revision of this document described the deviation as
introducing the fallback requirement itself, which overstated what was
actually being deviated from — corrected here to avoid implying the
fallback requirement was optional in the original spec.

**Original requirement:** Google Gemini as the primary LLM provider, with a
fallback API required (spec as written) — but without specifying which
platform that fallback should be.

**Deviation:** A second, configured provider (Groq) is used strictly as a
fallback when the primary Gemini call fails (timeout, rate limit, transient
provider error). Gemini remains the default/primary provider for every call;
Groq is never selected first and is never used to broaden functionality
beyond what Gemini already provides.

**Rationale:** Gemini is a paid, metered API. A single-provider design has no
graceful-degradation path if the Gemini quota is exhausted or the API is
briefly unavailable during grading/demo, which would fail the run for reasons
unrelated to the system's own logic. Groq was chosen as the specific backup
platform because it gives the system a bounded, observable failure-handling
path (see `docs/llm-provider-policy.md`) instead of a hard stop, without
requiring a locally-hosted model server (out of scope per the No-Docker /
no-external-service constraint).

**Scope limits (what this deviation does *not* change):**
- Neither Gemini nor Groq ever sets a policy decision (`kyc_result.status`,
  `credit_assessment.decision`, offer terms, or `next_node`/routing). Both
  providers are restricted to extraction, rationale, confidence, and
  constrained generation, exactly as for the primary-only design the spec
  describes. Decision authority stays entirely inside the deterministic
  Python policy gates (`src/agents/*`, `src/graph/*`).
- The fallback is only ever invoked after a primary-provider failure; it does
  not run speculatively, in parallel, or for load-balancing.
- No additional providers beyond this one configured backup are introduced.

**Approval:** Verbally approved by the course administrators as an acceptable
deviation given Gemini's paid/metered status. This file is the durable record of that approval;
it is referenced from the spec clause it modifies (see
`specs/loan_origination_spec.md`) and from `docs/llm-provider-policy.md`,
`docs/traceability.md`, and `README.md` rather than each of those documents
separately asserting the backup was in the original spec's scope.

**Status:** Approved, active, scope as described above.
