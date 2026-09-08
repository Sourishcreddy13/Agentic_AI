"""Offer-draft worker (AC-02, AC-04).

Offer drafting is deterministic, not LLM-based. Every PRICE-001 pricing tier
pins min_apr == max_apr, so APR is never a free variable in the first place;
principal and term are issued at the tier ceiling. There is nothing left for
an LLM to propose that Python doesn't already fully determine, so the offer
is built directly from the tier constraints and validated through the
OfferDraft Pydantic schema at the handoff boundary (AC-04).
"""
from __future__ import annotations

from langchain_core.messages import AIMessage

from src.exceptions import WorkerPreconditionError
from src.state.schema import LoanApplicationState, OfferDraft


def _offer_constraints(bureau_score: int, annual_income: float) -> dict | None:
    """Return hard pricing bounds from PRICE-001."""
    if bureau_score >= 750:
        return {"max_principal": 5_000_000, "min_apr": 8.5, "max_apr": 8.5, "max_term": 60}
    if bureau_score >= 700:
        if annual_income > 75_000:
            return {"max_principal": 3_000_000, "min_apr": 9.5, "max_apr": 9.5, "max_term": 48}
        return {"max_principal": 1_500_000, "min_apr": 10.5, "max_apr": 10.5, "max_term": 36}
    if bureau_score >= 650:
        return {"max_principal": 750_000, "min_apr": 12.0, "max_apr": 12.0, "max_term": 36}
    if bureau_score >= 600:
        return {"max_principal": 500_000, "min_apr": 13.5, "max_apr": 13.5, "max_term": 24}
    return None


MANDATORY_OFFER_CONDITIONS = [
    "Indicative offer subject to final underwriter sign-off.",
    "APR is representative; final rate may vary by +/-0.5% on approval.",
    "Offer expires 30 days from date of issue.",
]


def offer_draft_node(state: LoanApplicationState) -> dict:
    kyc = state["kyc_result"]
    credit = state["credit_assessment"]
    applicant = state["applicant"]

    if kyc is not None and kyc.status == "fail":
        offer = OfferDraft.model_validate({
            "principal": 0, "apr": 0, "term_months": 0,
            "conditions": ["Application declined at KYC stage; referred to compliance officer."],
            "is_indicative": True,
        })
        return {
            "offer": offer,
            "next_node": "END",
            "messages": [AIMessage(content="Offer: declined referral (KYC fail).")],
        }

    if credit is None or applicant is None:
        # Routing invariant violation: reachable here only if
        # route_after_kyc / route_after_credit were changed to send an
        # incomplete state to offer_draft. Deliberately raised, not
        # returned as a ReflectionNote: offer_draft's only outgoing edge is
        # a static add_edge to memory_consolidation (see
        # src/graph/build_graph.py) — there is no conditional edge back to
        # "reflector" the way kyc_check/credit_assessment have. A
        # ReflectionNote returned here would be silently ignored by
        # routing and the run would report success with offer left None.
        # WorkerPreconditionError (src/exceptions.py) replaces what used to
        # be a bare `assert` — same "this should never happen, fail loudly"
        # intent, but not silently stripped under python -O, and
        # self-describing in _observed_node's error_type log field.
        raise WorkerPreconditionError(
            "offer_draft_node reached without both a credit assessment and "
            "an applicant profile (and KYC did not fail)."
        )

    if credit.decision != "approve":
        offer = OfferDraft.model_validate({
            "principal": 0, "apr": 0, "term_months": 0,
            "conditions": [f"Not approved for indicative offer: {credit.decision}."],
            "is_indicative": True,
        })
        return {
            "offer": offer,
            "next_node": "END",
            "messages": [AIMessage(content=f"Offer: not approved ({credit.decision}).")],
        }

    constraints = _offer_constraints(
        credit.bureau_score_synthetic,
        applicant.declared_income,
    )
    if constraints is None:
        offer = OfferDraft.model_validate({
            "principal": 0, "apr": 0, "term_months": 0,
            "conditions": ["Pricing tier requires manual review; no indicative offer issued."],
            "is_indicative": True,
        })
        return {
            "offer": offer,
            "next_node": "END",
            "messages": [AIMessage(content="Offer: pricing tier requires manual review.")],
        }

    # Deterministic construction: the offer is fully determined by the tier
    # the applicant already qualified for (principal at the ceiling, APR at
    # the tier's single pinned rate, term at the ceiling). Still validated
    # through the Pydantic schema at the handoff boundary (AC-04).
    offer = OfferDraft.model_validate({
        "principal": constraints["max_principal"],
        "apr": constraints["min_apr"],
        "term_months": constraints["max_term"],
        "conditions": list(MANDATORY_OFFER_CONDITIONS),
        "is_indicative": True,
    })

    return {
        "offer": offer,
        "next_node": "END",
        "messages": [
            AIMessage(
                content=(
                    f"Offer drafted: principal={offer.principal}, "
                    f"apr={offer.apr}%, term={offer.term_months}mo."
                )
            )
        ],
    }
