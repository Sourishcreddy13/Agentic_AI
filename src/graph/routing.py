"""Conditional edges (AC-03). Every function inspects LoanApplicationState
and returns the name of the next node (or "END")."""
from src.state.schema import LoanApplicationState
from src.observability.audit_log import log_event

MAX_RETRIES = 2


def _record(state: LoanApplicationState, stage: str, destination: str) -> str:
    log_event(
        "routing_decision",
        user_id=state.get("user_id"),
        thread_id=state.get("thread_id"),
        stage=stage,
        destination=destination,
    )
    return destination


def route_after_supervisor(state: LoanApplicationState) -> str:
    """Dispatch exactly the hop the supervisor computed.

    This is the fix for the previously "dead" next_node field: the
    supervisor's decision used to be discarded by a static add_edge to
    "intake". It is now consulted directly, so a resumed/continued thread
    (applicant/kyc/credit already populated in state) dispatches to its
    correct next stage instead of always restarting at intake.
    """
    destination = state.get("next_node") or "intake"
    return _record(state, "after_supervisor", destination)


def route_after_intake(state: LoanApplicationState) -> str:
    destination = "kyc_check" if state["applicant"] else "reflector"
    return _record(state, "after_intake", destination)


def route_after_kyc(state: LoanApplicationState) -> str:
    kyc = state["kyc_result"]
    if kyc is None:
        destination = "reflector"
    elif kyc.status == "fail":
        destination = "offer_draft"
    elif kyc.status == "manual_review":
        destination = "reflector"
    elif kyc.confidence < 0.55:
        destination = "reflector"
    else:
        destination = "credit_assessment"
    return _record(state, "after_kyc", destination)


def route_after_credit(state: LoanApplicationState) -> str:
    ca = state["credit_assessment"]
    if ca is None or ca.confidence < 0.55:
        destination = "reflector"
    elif ca.decision == "decline":
        destination = "offer_draft"
    elif ca.decision == "manual_underwriting" or ca.thin_file:
        destination = "reflector"
    else:
        destination = "offer_draft"
    return _record(state, "after_credit", destination)


def route_after_reflection(state: LoanApplicationState) -> str:
    if state["retry_count"] >= MAX_RETRIES:
        return _record(state, "after_reflection", "END")
    if not state["reflection_log"]:
        return _record(state, "after_reflection", "END")

    last = state["reflection_log"][-1]
    if last.action_taken == "escalate_to_human":
        return _record(state, "after_reflection", "END")
    if last.action_taken == "replan":
        destination = "intake" if state["applicant"] is None else "kyc_check"
        return _record(state, "after_reflection", destination)
    if last.action_taken == "retry":
        if state["applicant"] is None:
            destination = "intake"
        elif state["kyc_result"] is None:
            destination = "kyc_check"
        else:
            destination = "credit_assessment"
        return _record(state, "after_reflection", destination)
    return _record(state, "after_reflection", "END")
