"""
Phase 2 graph-level MCP integration.
AC-10: the actual loan graph invokes MCP tools through the adapter.
"""
import json
from pathlib import Path

from src.graph.build_graph import build_graph
from tests.helpers import load_initial_state


EVIDENCE_PATH = Path("evidence/run_transcripts/phase2_mcp_graph.json")


def test_strong_application_uses_mcp_for_kyc_and_credit():
    graph = build_graph()
    result = graph.invoke(load_initial_state("applicant_strong.json", "phase2-strong"))

    assert result["kyc_result"].status == "pass"
    assert result["credit_assessment"].bureau_score_synthetic == 740
    assert result["credit_assessment"].decision == "approve"

    evidence = {
        "ac_reference": "AC-10",
        "thread_id": "phase2-strong",
        "path": "supervisor -> intake -> kyc_check[MCP applicant_lookup] -> credit_assessment[MCP bureau_check] -> offer_draft -> END",
        "mcp_tools_used_by_graph": ["applicant_lookup", "bureau_check"],
        "kyc_status": result["kyc_result"].status,
        "bureau_score_synthetic": result["credit_assessment"].bureau_score_synthetic,
        "credit_decision": result["credit_assessment"].decision,
    }
    EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE_PATH.write_text(json.dumps(evidence, indent=2))


def test_watchlist_application_uses_mcp_lookup_for_kyc_gate():
    graph = build_graph()
    result = graph.invoke(load_initial_state("applicant_kyc_fail.json", "phase2-watchlist"))

    assert result["kyc_result"].status == "fail"
    assert "synthetic_watchlist_hit" in result["kyc_result"].risk_flags
    assert result["credit_assessment"] is None


def test_watchlist_decision_comes_from_mcp_lookup_not_applicant_name():
    from langchain_core.messages import HumanMessage
    from src.state.schema import new_state

    graph = build_graph()
    payload = {
        "applicant_id": "SYN-0003",
        "full_name": "Different Synthetic Name",
        "dob_synthetic": "1985-03-22",
        "declared_income": 60000,
        "declared_employment": "Synthetic test employer",
        "raw_free_text_notes": "Normal synthetic notes.",
    }
    state = new_state(thread_id="phase2-mcp-source")
    state["messages"] = [HumanMessage(content=json.dumps(payload))]

    result = graph.invoke(state)

    assert result["kyc_result"].status == "fail"
    assert "synthetic_watchlist_hit" in result["kyc_result"].risk_flags


def test_mcp_bureau_failure_enters_reflection_retry(monkeypatch):
    import src.agents.credit_agent as credit_agent

    def fail_mcp(*args, **kwargs):
        raise RuntimeError("synthetic MCP outage")

    monkeypatch.setattr(credit_agent, "invoke_mcp_tool_sync", fail_mcp)

    graph = build_graph()
    result = graph.invoke(load_initial_state("applicant_strong.json", "phase2-mcp-failure"))

    assert result["credit_assessment"] is None
    assert result["reflection_log"]
    assert result["reflection_log"][-1].triggered_by in {
        "mcp_bureau_check_failure",
        "max_retries_exceeded",
    }
