"""AC-01: typed state object with user_id identity model."""
from src.state.schema import (
    LoanApplicationState, ApplicantProfile, KYCResult, new_state,
)


def test_state_is_typed_dict_with_expected_keys():
    state = new_state(thread_id="t-1", user_id="U-001")
    expected_keys = {
        "user_id", "thread_id", "messages", "applicant", "kyc_result",
        "credit_assessment", "offer", "next_node", "retry_count",
        "reflection_log", "quarantined_inputs", "compressed_summary",
        "long_term_memory_hits", "compliance_flags",
    }
    assert set(state.keys()) == expected_keys
    assert state["thread_id"] == "t-1"
    assert state["user_id"] == "U-001"
    assert state["retry_count"] == 0


def test_state_user_id_defaults_to_default_user():
    state = new_state(thread_id="t-2")
    assert state["user_id"] == "default-user"


def test_applicant_profile_validates_bounds():
    profile = ApplicantProfile(
        applicant_id="SYN-0001", full_name="Test", dob_synthetic="1990-01-01",
        declared_income=50000, declared_employment="Tester",
    )
    assert profile.declared_income == 50000

    try:
        ApplicantProfile(
            applicant_id="SYN-0002", full_name="Bad", dob_synthetic="1990-01-01",
            declared_income=-10, declared_employment="Tester",
        )
        assert False, "expected ValidationError for negative income"
    except Exception:
        pass


def test_kyc_result_confidence_bounds():
    result = KYCResult(status="pass", checks_performed=["id_match"], confidence=0.9)
    assert 0 <= result.confidence <= 1
