"""
Supervisor node (AC-02). Pure router — decides the entry hop; every
subsequent hop is decided by the conditional edges in routing.py, driven
by worker outputs.
"""
from src.state.schema import LoanApplicationState


def supervisor(state: LoanApplicationState) -> dict:
    if state["applicant"] is None:
        return {"next_node": "intake"}
    return {"next_node": state.get("next_node") or "intake"}
