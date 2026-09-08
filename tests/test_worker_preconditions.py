"""
Regression tests for the corrective action replacing bare `assert`
node-entry guards in kyc_agent.py / credit_agent.py / offer_agent.py.

Previously these three nodes used `assert applicant is not None` /
`assert credit is not None` to guard an invariant the graph topology is
supposed to already enforce. Two problems with that: `assert` is stripped
entirely under `python -O`, and an `AssertionError` bypasses the
retry/replan/escalate self-healing loop (AC-12) that every other failure
mode in this project goes through.

kyc_check and credit_assessment both have a real conditional edge back to
"reflector" (route_after_kyc / route_after_credit already send a `None`
result there), so those two now fail soft into a `ReflectionNote` instead
of raising. offer_draft has no such edge (its only outgoing edge is a
static add_edge to memory_consolidation), so a ReflectionNote returned
there would be silently ignored by routing — it still raises, now as a
named `WorkerPreconditionError` (src/exceptions.py) instead of a bare
`assert`, so the failure is not silently compiled away and is
self-describing in the audit log's `error_type` field.
"""
from __future__ import annotations

import pytest

from src.agents.credit_agent import credit_assessment_node
from src.agents.kyc_agent import kyc_check_node
from src.agents.offer_agent import offer_draft_node
from src.exceptions import WorkerPreconditionError
from src.state.schema import CreditAssessment, new_state


def test_kyc_check_node_without_applicant_returns_reflection_note_not_assertion_error():
    state = new_state(thread_id="precondition-kyc")
    assert state["applicant"] is None

    result = kyc_check_node(state)

    assert result["reflection_log"][0].triggered_by == "missing_prerequisite_state"
    assert result["reflection_log"][0].action_taken == "escalate_to_human"
    assert "kyc_result" not in result


def test_credit_assessment_node_without_applicant_returns_reflection_note_not_assertion_error():
    state = new_state(thread_id="precondition-credit")
    assert state["applicant"] is None

    result = credit_assessment_node(state)

    assert result["reflection_log"][0].triggered_by == "missing_prerequisite_state"
    assert result["reflection_log"][0].action_taken == "escalate_to_human"
    assert "credit_assessment" not in result


def test_missing_prerequisite_state_is_classified_as_escalate_by_reflector():
    from src.agents.reflector import _classify

    assert _classify("missing_prerequisite_state") == "escalate_to_human"


def test_offer_draft_node_without_credit_assessment_raises_worker_precondition_error():
    """offer_draft has no conditional edge back to reflector (static
    add_edge to memory_consolidation only), so a routing-invariant
    violation here must fail loudly rather than be silently absorbed as a
    successful run with offer left None."""
    state = new_state(thread_id="precondition-offer")
    state["kyc_result"] = None  # not a KYC-fail referral
    assert state["credit_assessment"] is None

    with pytest.raises(WorkerPreconditionError):
        offer_draft_node(state)


def test_offer_draft_node_without_applicant_raises_worker_precondition_error():
    state = new_state(thread_id="precondition-offer-2")
    state["kyc_result"] = None
    state["credit_assessment"] = CreditAssessment(
        thin_file=False, bureau_score_synthetic=700, dti_ratio=0.2,
        decision="approve", rationale="r", confidence=0.9,
    )
    assert state["applicant"] is None

    with pytest.raises(WorkerPreconditionError):
        offer_draft_node(state)
