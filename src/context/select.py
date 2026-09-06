"""Context selection helpers for Phase 5."""
from __future__ import annotations

from typing import Any

from src.state.schema import LoanApplicationState


def select_context(state: LoanApplicationState, purpose: str) -> dict[str, Any]:
    """Return only the state fields needed for a specific worker purpose."""
    if purpose == "intake":
        return {
            "user_id": state["user_id"],
            "compressed_summary": state.get("compressed_summary"),
        }
    if purpose == "kyc":
        applicant = state["applicant"]
        return {
            "applicant": applicant,
            "long_term_memory_hits": state.get("long_term_memory_hits") or [],
            "compressed_summary": state.get("compressed_summary"),
        }
    if purpose == "credit":
        applicant = state["applicant"]
        kyc = state["kyc_result"]
        return {
            "applicant": applicant,
            "kyc_result": kyc,
            "long_term_memory_hits": state.get("long_term_memory_hits") or [],
            "compressed_summary": state.get("compressed_summary"),
        }
    if purpose == "offer":
        return {
            "applicant": state["applicant"],
            "credit_assessment": state["credit_assessment"],
            "compressed_summary": state.get("compressed_summary"),
        }
    raise ValueError(f"Unknown context purpose: {purpose}")
