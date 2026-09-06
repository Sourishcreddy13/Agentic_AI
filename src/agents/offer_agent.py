"""Offer-draft worker (AC-02, AC-04)."""
from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from src.llm.gateway import invoke_structured_with_fallback
from src.context.middleware import prepare_worker_context
from src.state.schema import LoanApplicationState, OfferDraft, ReflectionNote


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


def _enforce_constraints(offer: OfferDraft, constraints: dict) -> OfferDraft:
    """Hard-enforce pricing policy after model generation."""
    clipped = offer.model_copy(
        update={
            "principal": max(0, min(offer.principal, constraints["max_principal"])),
            "apr": max(constraints["min_apr"], min(offer.apr, constraints["max_apr"])),
            "term_months": max(1, min(offer.term_months, constraints["max_term"])),
            "is_indicative": True,
        }
    )

    mandatory_conditions = [
        "Indicative offer subject to final underwriter sign-off.",
        "APR is representative; final rate may vary by +/-0.5% on approval.",
        "Offer expires 30 days from date of issue.",
    ]
    return clipped.model_copy(update={"conditions": mandatory_conditions})


OFFER_SYSTEM_PROMPT = """You are an offer-drafting specialist in a synthetic loan-origination workflow.
Draft an indicative offer inside the exact policy bounds supplied by Python.
Never exceed the principal ceiling, APR range, or term ceiling.
The offer is indicative and subject to final human underwriter sign-off.
Return only the requested structured offer."""


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
        return {"offer": offer, "next_node": "END"}

    assert credit is not None, "offer_draft_node reached without a credit assessment"
    assert applicant is not None, "offer_draft_node reached without an applicant profile"

    if credit.decision != "approve":
        offer = OfferDraft.model_validate({
            "principal": 0, "apr": 0, "term_months": 0,
            "conditions": [f"Not approved for indicative offer: {credit.decision}."],
            "is_indicative": True,
        })
        return {"offer": offer, "next_node": "END"}

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
        return {"offer": offer, "next_node": "END"}

    selected, _compression = prepare_worker_context(state, "offer")
    try:
        result = invoke_structured_with_fallback(
            OfferDraft,
            [
                SystemMessage(content=OFFER_SYSTEM_PROMPT),
                HumanMessage(
                    content=(
                        f"Applicant: {applicant.model_dump_json()}\n"
                        f"Credit assessment: {credit.model_dump_json()}\n"
                        f"Prior-context summary: {selected.get('compressed_summary') or 'None'}\n"
                        "Policy constraints:\n"
                        f"  max_principal = {constraints['max_principal']:.0f}\n"
                        f"  APR range     = {constraints['min_apr']}% – {constraints['max_apr']}%\n"
                        f"  max_term      = {constraints['max_term']} months"
                    )
                ),
            ],
        )
        offer = OfferDraft.model_validate(result)
        offer = _enforce_constraints(offer, constraints)
    except Exception as exc:
        return {
            "reflection_log": [
                ReflectionNote(
                    triggered_by="llm_offer_failure",
                    action_taken="retry",
                    detail=f"Offer model invocation/validation failed: {str(exc)[:240]}",
                )
            ]
        }

    return {"offer": offer, "next_node": "END"}
