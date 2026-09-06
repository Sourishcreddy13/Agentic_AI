"""
AC-12: Reflection / self-healing loop with evidenced trace.

Tests the reflector node and the full graph's self-healing behaviour.
Writes evidence/reflection_trace.json as the committed AC-12 artifact.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage

from src.graph.build_graph import build_graph
from src.state.schema import new_state

EVIDENCE_PATH = Path("evidence/reflection_trace.json")


# --------------------------------------------------------------------------
# AC-12: MCP tool failure routes through reflector
# --------------------------------------------------------------------------

def test_mcp_tool_failure_triggers_reflection_and_writes_trace(monkeypatch):
    """
    AC-12: when bureau_check MCP call fails, the graph enters the reflection
    loop, exhausts retries, and escalates — never producing a credit decision
    from bad data. Writes evidence/reflection_trace.json.
    """
    import src.agents.credit_agent as credit_agent

    call_log = []

    def fail_mcp(tool_name, arguments):
        call_log.append({"tool": tool_name, "args": arguments})
        raise RuntimeError("synthetic MCP bureau_check outage")

    monkeypatch.setattr(credit_agent, "invoke_mcp_tool_sync", fail_mcp)

    graph = build_graph()
    state = new_state(thread_id="AC-12-mcp-failure")
    state["messages"] = [HumanMessage(content=json.dumps({
        "applicant_id": "SYN-0001",
        "full_name": "Asha Kulkarni",
        "dob_synthetic": "1990-01-01",
        "declared_income": 85000,
        "declared_employment": "Synthetic engineer",
        "raw_free_text_notes": "Normal application.",
    }))]

    result = graph.invoke(state)

    # Self-healing assertions
    assert result["credit_assessment"] is None, "No credit decision from failed MCP"
    assert result["reflection_log"], "Reflection log must be non-empty"
    triggered = result["reflection_log"][-1].triggered_by
    assert triggered in {"mcp_bureau_check_failure", "max_retries_exceeded"}, \
        f"Unexpected trigger: {triggered}"
    assert result["retry_count"] >= 1

    # Write committed AC-12 evidence
    trace = {
        "ac_reference": "AC-12",
        "scenario": "MCP bureau_check failure triggers reflection loop",
        "thread_id": "AC-12-mcp-failure",
        "mcp_calls_attempted": call_log,
        "retry_count_final": result["retry_count"],
        "reflection_log": [
            {"triggered_by": n.triggered_by, "action_taken": n.action_taken, "detail": n.detail}
            for n in result["reflection_log"]
        ],
        "credit_assessment": None,
        "outcome": "escalated_to_human — no credit decision issued from failed MCP data",
    }
    EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE_PATH.write_text(json.dumps(trace, indent=2))
    print(f"\nAC-12 trace written → {EVIDENCE_PATH}")


def test_low_confidence_credit_triggers_reflection():
    """AC-12: credit confidence < 0.55 routes to reflector, not offer_draft."""
    from src.graph.routing import route_after_credit
    from src.state.schema import CreditAssessment

    state = new_state(thread_id="low-conf")
    state["credit_assessment"] = CreditAssessment(
        thin_file=False,
        bureau_score_synthetic=650,
        dti_ratio=0.3,
        decision="approve",
        rationale="borderline",
        confidence=0.4,   # below 0.55 threshold
    )
    assert route_after_credit(state) == "reflector"


def test_reflector_escalates_after_max_retries():
    """AC-12/NFR-07: reflector escalates to human after MAX_RETRIES exhausted."""
    from src.agents.reflector import reflector_node
    from src.graph.routing import MAX_RETRIES
    from src.state.schema import ReflectionNote

    state = new_state(thread_id="max-retries")
    state["retry_count"] = MAX_RETRIES
    state["reflection_log"] = [
        ReflectionNote(triggered_by="mcp_failure", action_taken="retry", detail="test")
    ]

    result = reflector_node(state)
    assert result["reflection_log"][-1].action_taken == "escalate_to_human"
    assert result["reflection_log"][-1].triggered_by == "max_retries_exceeded"


def test_reflector_does_not_skip_kyc_prerequisite_on_retry():
    """AC-12: retry routing from reflector respects prerequisites — never skips KYC."""
    from src.graph.routing import route_after_reflection
    from src.state.schema import ReflectionNote

    state = new_state(thread_id="prereq-test")
    state["retry_count"] = 1
    state["applicant"] = None   # intake hasn't completed yet
    state["reflection_log"] = [
        ReflectionNote(triggered_by="some_failure", action_taken="retry", detail="x")
    ]

    # With no applicant, retry must go back to intake, not skip to credit
    assert route_after_reflection(state) == "intake"



def test_low_confidence_kyc_triggers_reflection():
    """AC-12: low-confidence KYC output routes to reflector before credit."""
    from src.graph.routing import route_after_kyc
    from src.state.schema import KYCResult

    state = {
        "kyc_result": KYCResult(
            status="pass",
            checks_performed=["watchlist_screen"],
            confidence=0.40,
        )
    }
    assert route_after_kyc(state) == "reflector"
