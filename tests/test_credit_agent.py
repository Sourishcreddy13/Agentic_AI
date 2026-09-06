"""AC-04/AC-12: credit_agent's rationale-failure retry budget is decoupled
from the graph-wide retry_count.

credit_assessment_node stubs a fallback rationale exactly once when the LLM
rationale call fails, then escalates on a second such failure. This used to
be gated on `state["retry_count"] == 0` -- the *graph-wide* counter, which
is incremented by reflector_node for ANY prior failure (an MCP hiccup in an
earlier stage, an intake validation retry, etc.), not just a prior rationale
failure in this node. That meant an unrelated earlier retry silently
consumed this node's one-stub-then-escalate budget, so the very first
rationale failure in *this* node would escalate immediately instead of
being stubbed. The fix counts prior `llm_rationale_failure` entries in
`reflection_log` specifically.
"""
from __future__ import annotations

from src.agents.credit_agent import credit_assessment_node
from src.rag.agentic_rag import PolicyLookupResult
from src.state.schema import ApplicantProfile, ReflectionNote, new_state


def _state_with_applicant(*, retry_count: int, reflection_log: list[ReflectionNote]) -> dict:
    state = new_state(thread_id="credit-retry-decoupling")
    state["applicant"] = ApplicantProfile(
        applicant_id="SYN-CREDIT-1",
        full_name="Retry Decoupling Test",
        dob_synthetic="1990-01-01",
        declared_income=80000,
        declared_employment="Synthetic engineer",
    )
    state["retry_count"] = retry_count
    state["reflection_log"] = reflection_log
    return state


def _fail_rationale(*args, **kwargs):
    raise RuntimeError("synthetic rationale generation failure")


def test_first_rationale_failure_is_stubbed_even_after_an_unrelated_earlier_retry(monkeypatch):
    """
    retry_count is already 1 from an EARLIER, unrelated failure (e.g. an
    upstream MCP hiccup) -- but this is credit_assessment_node's FIRST
    rationale failure. It must still be stubbed with a fallback rationale,
    not escalated, because the old `retry_count == 0` check would have
    wrongly escalated here.
    """
    import src.agents.credit_agent as credit_agent

    monkeypatch.setattr(credit_agent, "invoke_mcp_tool_sync", lambda *a, **k: {
        "applicant_id": "SYN-CREDIT-1",
        "found_in_bureau": True,
        "synthetic_score": 720,
        "delinquencies_24m": 0,
        "thin_file": False,
        "dti_estimate": 0.2,
    })
    monkeypatch.setattr(credit_agent, "invoke_structured_with_fallback", _fail_rationale)
    monkeypatch.setattr(
        credit_agent, "agentic_policy_lookup",
        lambda task, max_steps=2: PolicyLookupResult(False, (), (), 0),
    )

    state = _state_with_applicant(
        retry_count=1,
        reflection_log=[
            ReflectionNote(
                triggered_by="mcp_applicant_lookup_failure",
                action_taken="retry",
                detail="an earlier, unrelated MCP hiccup",
            )
        ],
    )

    result = credit_assessment_node(state)

    assert "credit_assessment" in result, (
        "expected the rationale failure to be stubbed (fallback rationale, "
        "decision still produced) rather than escalated, since no prior "
        "llm_rationale_failure exists for THIS node"
    )
    assert result["credit_assessment"].decision == "approve"
    assert "rationale unavailable" in result["credit_assessment"].rationale.lower()


def test_second_rationale_failure_in_this_node_escalates(monkeypatch):
    """A genuine second llm_rationale_failure for credit must still escalate."""
    import src.agents.credit_agent as credit_agent

    monkeypatch.setattr(credit_agent, "invoke_mcp_tool_sync", lambda *a, **k: {
        "applicant_id": "SYN-CREDIT-1",
        "found_in_bureau": True,
        "synthetic_score": 720,
        "delinquencies_24m": 0,
        "thin_file": False,
        "dti_estimate": 0.2,
    })
    monkeypatch.setattr(credit_agent, "invoke_structured_with_fallback", _fail_rationale)
    monkeypatch.setattr(
        credit_agent, "agentic_policy_lookup",
        lambda task, max_steps=2: PolicyLookupResult(False, (), (), 0),
    )

    state = _state_with_applicant(
        retry_count=1,
        reflection_log=[
            ReflectionNote(
                triggered_by="llm_rationale_failure",
                action_taken="retry",
                detail="a prior rationale failure for this same node",
            )
        ],
    )

    result = credit_assessment_node(state)

    assert "credit_assessment" not in result
    assert result["reflection_log"][0].triggered_by == "llm_rationale_failure"
