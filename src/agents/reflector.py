"""
Reflection / self-healing node (AC-12).

Failure taxonomy
----------------
RETRYABLE   — transient failures (MCP outage, LLM timeout, malformed output).
REPLAN      — structural failures (missing prerequisite, wrong stage output).
ESCALATE    — policy/business outcomes reflector_node itself decides to
              escalate (KYC manual review, thin-file, max retries).

Every value in RETRYABLE / REPLAN / ESCALATE below is a `ReflectionNote.
triggered_by` value that can legitimately be the *last* entry in
reflection_log when reflector_node runs, because reflector_node's own
classification logic reads exactly that last entry to decide what to do
next. Intake and offer-drafting are deterministic Python (no LLM call), so
neither produces a distinct failure trigger of its own any more; an intake
shape mismatch on unexpected input still surfaces as REPLAN's
`intake_validation_error`, and offer construction has no failure path left
to classify. Two related but distinct conditions — a KYC-fail referral, and a
detected prompt-injection attempt in applicant free text — are NOT included
here even though they are escalation-worthy: they are compliance events
that must be visible for audit without ever being able to influence this
control loop (a KYC-fail is routed straight to a declined-offer referral by
route_after_kyc, never through the reflector at all; an injection attempt
must never change routing, by design). Both are recorded instead as
`ComplianceEvent` entries in `state["compliance_flags"]` — see
src/agents/kyc_agent.py and src/agents/intake_agent.py — a channel this
module deliberately never reads.

Budget rule: both RETRYABLE and REPLAN triggers count against retry_count.
When retry_count reaches MAX_RETRIES - 1 the reflector writes a
max_retries_exceeded note so routing can see the terminal state and END.
"""
from __future__ import annotations

from src.state.schema import LoanApplicationState, OfferDraft, ReflectionNote
from src.graph.routing import MAX_RETRIES


# --------------------------------------------------------------------------
# Explicit failure taxonomy
# --------------------------------------------------------------------------

RETRYABLE: frozenset[str] = frozenset({
    "mcp_bureau_check_failure",
    "mcp_applicant_lookup_failure",
    "llm_kyc_failure",
    "llm_credit_failure",
    "llm_rationale_failure",
    "credit_assembly_error",
    "low_confidence_or_missing_output",
})

REPLAN: frozenset[str] = frozenset({
    "intake_validation_error",
    "missing_application_payload",
})

ESCALATE: frozenset[str] = frozenset({
    "kyc_manual_review",
    "thin_file_manual_underwriting",
    "max_retries_exceeded",
    # Raised by kyc_check_node / credit_assessment_node when a routing
    # invariant is violated (a node was entered without its prerequisite
    # state) — see src/exceptions.py for why offer_draft_node instead
    # raises WorkerPreconditionError rather than reaching this taxonomy at
    # all. Not a normal runtime failure, so it is not retried.
    "missing_prerequisite_state",
})


def _classify(triggered_by: str) -> str:
    if triggered_by in RETRYABLE:
        return "retry"
    if triggered_by in REPLAN:
        return "replan"
    return "escalate_to_human"   # unknown triggers default to escalate (fail-safe)


def reflector_node(state: LoanApplicationState) -> dict:

    # --- Global circuit breaker: already at max (NFR-07) ---
    if state["retry_count"] >= MAX_RETRIES:
        note = ReflectionNote(
            triggered_by="max_retries_exceeded",
            action_taken="escalate_to_human",
            detail=f"Exceeded {MAX_RETRIES} retries; escalating to a human officer.",
        )
        return {"reflection_log": [note], "retry_count": 1}

    # --- KYC manual review: always escalate immediately ---
    kyc = state["kyc_result"]
    if kyc is not None and kyc.status == "manual_review":
        note = ReflectionNote(
            triggered_by="kyc_manual_review",
            action_taken="escalate_to_human",
            detail="KYC requires manual review; escalating before credit assessment.",
        )
        return {"reflection_log": [note], "retry_count": 1}

    # --- Thin-file: always escalate to manual underwriting ---
    ca = state["credit_assessment"]
    if ca is not None and ca.thin_file:
        note = ReflectionNote(
            triggered_by="thin_file_manual_underwriting",
            action_taken="escalate_to_human",
            detail="Thin-file applicant routed to manual underwriting.",
        )
        # This escalation ends the graph (route_after_reflection sends
        # escalate_to_human straight to END) without ever reaching
        # offer_draft — so unlike a KYC-fail referral or a declined credit
        # decision (both of which do reach offer_draft and get an explicit
        # zero-principal OfferDraft with a human-readable referral note),
        # a thin-file manual-underwriting escalation used to leave no
        # committed outcome record at all beyond an ephemeral
        # ReflectionNote. Write the same shape of zero-value referral offer
        # here so a human-officer queue consuming `offer` has something to
        # act on for every escalation path, not just two of the three.
        referral_offer = OfferDraft.model_validate({
            "principal": 0, "apr": 0, "term_months": 0,
            "conditions": [
                "Thin-file application referred to manual underwriting "
                "team; no automated offer issued."
            ],
            "is_indicative": True,
        })
        return {
            "reflection_log": [note],
            "retry_count": 1,
            "offer": referral_offer,
        }

    # --- Classify the most recent failure and apply budget check ---
    if state["reflection_log"]:
        last = state["reflection_log"][-1]
        action = _classify(last.triggered_by)

        if action == "escalate_to_human":
            # Pass-through: routing will see the existing escalate note and END.
            return {"retry_count": 1}

        # Budget check: escalate on the LAST permitted attempt (both retry and replan)
        if state["retry_count"] >= MAX_RETRIES - 1:
            note = ReflectionNote(
                triggered_by="max_retries_exceeded",
                action_taken="escalate_to_human",
                detail=f"Exceeded {MAX_RETRIES} retries; escalating to a human officer.",
            )
            return {"reflection_log": [note], "retry_count": 1}

        # Still within budget — increment and let routing re-dispatch
        return {"retry_count": 1}

    # --- Default: no prior log entry, treat as retryable ---
    note = ReflectionNote(
        triggered_by="low_confidence_or_missing_output",
        action_taken="retry",
        detail="Retrying the upstream node once before escalating.",
    )
    return {"reflection_log": [note], "retry_count": 1}
