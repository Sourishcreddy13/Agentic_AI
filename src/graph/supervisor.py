"""
Supervisor node (AC-02).

This is a genuine entry-point router, not a pass-through: it inspects how
much of the application has already been completed in state and decides
which worker should run next. That decision is consulted by a real
conditional edge (`route_after_supervisor` in routing.py, wired in
build_graph.py) rather than a static `add_edge`, so re-invoking a thread
that already has partial progress (e.g. resumed after an `interrupt_before`
pause, or a follow-up message added to an already-decided application)
dispatches straight to the correct next stage instead of unconditionally
restarting at intake.

Every subsequent hop after the first is still decided by the
worker-output-driven conditional edges in routing.py, exactly as before.
"""
from src.state.schema import LoanApplicationState


def supervisor(state: LoanApplicationState) -> dict:
    """Decide the entry hop from however much of the application already exists.

    - No applicant profile yet          -> intake
    - Applicant but no KYC result       -> kyc_check
    - KYC passed but no credit decision -> credit_assessment
    - Credit decision but no offer      -> offer_draft
    - Everything already decided        -> END (e.g. a follow-up message on
      an already-completed application should not re-run underwriting)
    """
    if state["applicant"] is None:
        return {"next_node": "intake"}
    if state["kyc_result"] is None:
        return {"next_node": "kyc_check"}
    if state["credit_assessment"] is None:
        return {"next_node": "credit_assessment"}
    if state["offer"] is None:
        return {"next_node": "offer_draft"}
    return {"next_node": "END"}
