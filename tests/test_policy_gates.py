"""Phase 3 deterministic policy gate tests."""
from src.agents.credit_agent import BureauCheckResult, _apply_policy_gates
from src.agents.offer_agent import _enforce_constraints, _offer_constraints
from src.state.schema import OfferDraft


def _bureau(score: int, dti: float, thin: bool = False):
    return BureauCheckResult(
        applicant_id="SYN-POLICY",
        found_in_bureau=True,
        synthetic_score=score,
        delinquencies_24m=0,
        thin_file=thin,
        dti_estimate=dti,
    )


def test_standard_dti_040_boundary_is_approved():
    decision, _ = _apply_policy_gates(_bureau(700, 0.40), 60_000)
    assert decision == "approve"


def test_standard_dti_above_040_is_declined():
    decision, _ = _apply_policy_gates(_bureau(700, 0.401), 60_000)
    assert decision == "decline"


def test_premium_dti_050_boundary_is_approved():
    decision, _ = _apply_policy_gates(_bureau(700, 0.50), 100_000)
    assert decision == "approve"


def test_dti_above_050_is_manual_underwriting():
    decision, _ = _apply_policy_gates(_bureau(700, 0.501), 100_000)
    assert decision == "manual_underwriting"


def test_500_to_599_score_is_manual_underwriting():
    decision, _ = _apply_policy_gates(_bureau(550, 0.10), 60_000)
    assert decision == "manual_underwriting"


def test_offer_constraints_follow_price_001_700_749_premium():
    constraints = _offer_constraints(740, 85_000)
    assert constraints == {
        "max_principal": 3_000_000,
        "min_apr": 9.5,
        "max_apr": 9.5,
        "max_term": 48,
    }


def test_offer_constraints_follow_price_001_700_749_standard():
    constraints = _offer_constraints(740, 60_000)
    assert constraints == {
        "max_principal": 1_500_000,
        "min_apr": 10.5,
        "max_apr": 10.5,
        "max_term": 36,
    }


def test_offer_post_generation_constraints_are_hard_enforced():
    constraints = _offer_constraints(740, 85_000)
    offer = OfferDraft(
        principal=9_000_000,
        apr=2.0,
        term_months=120,
        conditions=["bad"],
        is_indicative=False,
    )
    enforced = _enforce_constraints(offer, constraints)
    assert enforced.principal == 3_000_000
    assert enforced.apr == 9.5
    assert enforced.term_months == 48
    assert enforced.is_indicative is True
    assert len(enforced.conditions) == 3



def test_kyc_watchlist_gate_cannot_be_overridden_by_llm_result():
    from src.agents.kyc_agent import ApplicantLookupResult, _apply_kyc_policy

    lookup = ApplicantLookupResult(
        applicant_id="SYN-0003",
        found=True,
        prior_applications=2,
        synthetic_watchlist_hit=True,
    )
    status, flags = _apply_kyc_policy(lookup)
    assert status == "fail"
    assert flags == ["synthetic_watchlist_hit"]


def test_unknown_applicant_kyc_gate_requires_manual_review():
    from src.agents.kyc_agent import ApplicantLookupResult, _apply_kyc_policy

    lookup = ApplicantLookupResult(
        applicant_id="SYN-UNKNOWN",
        found=False,
        prior_applications=0,
        synthetic_watchlist_hit=False,
    )
    status, flags = _apply_kyc_policy(lookup)
    assert status == "manual_review"
    assert flags == ["applicant_not_found"]


def test_low_confidence_kyc_routes_to_reflector():
    from src.graph.routing import route_after_kyc
    from src.state.schema import KYCResult

    state = {
        "kyc_result": KYCResult(
            status="pass",
            checks_performed=["watchlist_screen"],
            confidence=0.4,
        )
    }
    assert route_after_kyc(state) == "reflector"


def test_thin_file_dti_above_060_is_declined():
    decision, confidence = _apply_policy_gates(_bureau(490, 0.61, thin=True), 60_000)
    assert decision == "decline"
    assert confidence == 1.0
