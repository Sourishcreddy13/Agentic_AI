"""AC-02: supervisor is a genuine resume-aware router, not a pass-through.

Covers the fix in src/graph/supervisor.py + src/graph/routing.py +
build_graph.py: `next_node` computed by the supervisor is now consulted by a
real conditional edge (route_after_supervisor), instead of being discarded
by a static add_edge("supervisor", "intake"). Previously every invocation --
including a resumed thread with partial or complete progress already
checkpointed -- unconditionally restarted at intake.
"""
from __future__ import annotations

import json

from langchain_core.messages import HumanMessage

from src.graph.build_graph import build_graph
from src.graph.routing import route_after_supervisor
from src.graph.supervisor import supervisor
from src.memory.checkpointer import get_checkpointer
from src.state.schema import CreditAssessment, KYCResult, OfferDraft, new_state
from tests.helpers import load_initial_state


# --------------------------------------------------------------------------
# Unit level: supervisor() dispatch decision for each stage of completion
# --------------------------------------------------------------------------

def test_supervisor_dispatches_to_intake_when_no_applicant():
    state = new_state(thread_id="sup-1")
    assert state["applicant"] is None
    assert supervisor(state) == {"next_node": "intake"}


def test_supervisor_dispatches_to_kyc_when_applicant_set_but_no_kyc_result():
    state = new_state(thread_id="sup-2")
    state["applicant"] = object()  # presence is all supervisor() checks
    assert state["kyc_result"] is None
    assert supervisor(state) == {"next_node": "kyc_check"}


def test_supervisor_dispatches_to_credit_when_kyc_passed_but_no_credit_decision():
    state = new_state(thread_id="sup-3")
    state["applicant"] = object()
    state["kyc_result"] = KYCResult(status="pass", checks_performed=["x"], confidence=0.9)
    assert state["credit_assessment"] is None
    assert supervisor(state) == {"next_node": "credit_assessment"}


def test_supervisor_dispatches_to_offer_when_credit_decided_but_no_offer():
    state = new_state(thread_id="sup-4")
    state["applicant"] = object()
    state["kyc_result"] = KYCResult(status="pass", checks_performed=["x"], confidence=0.9)
    state["credit_assessment"] = CreditAssessment(
        thin_file=False, bureau_score_synthetic=700, dti_ratio=0.2,
        decision="approve", rationale="r", confidence=0.9,
    )
    assert state["offer"] is None
    assert supervisor(state) == {"next_node": "offer_draft"}


def test_supervisor_dispatches_to_end_when_everything_already_decided():
    state = new_state(thread_id="sup-5")
    state["applicant"] = object()
    state["kyc_result"] = KYCResult(status="pass", checks_performed=["x"], confidence=0.9)
    state["credit_assessment"] = CreditAssessment(
        thin_file=False, bureau_score_synthetic=700, dti_ratio=0.2,
        decision="approve", rationale="r", confidence=0.9,
    )
    state["offer"] = OfferDraft(
        principal=1000, apr=9.0, term_months=12, conditions=[], is_indicative=True,
    )
    assert supervisor(state) == {"next_node": "END"}


def test_route_after_supervisor_follows_computed_next_node():
    state = new_state(thread_id="sup-route")
    state["next_node"] = "credit_assessment"
    assert route_after_supervisor(state) == "credit_assessment"


def test_route_after_supervisor_defaults_to_intake_when_next_node_missing():
    state = new_state(thread_id="sup-route-default")
    assert state.get("next_node") is None
    assert route_after_supervisor(state) == "intake"


# --------------------------------------------------------------------------
# Integration level: the real conditional edge actually short-circuits a
# completed thread to END instead of unconditionally restarting at intake.
# --------------------------------------------------------------------------

def test_completed_thread_resumes_to_end_without_rerunning_intake(tmp_path):
    db_path = tmp_path / "resume_checkpoints.sqlite"
    thread_id = "resume-to-end"

    with get_checkpointer(db_path) as checkpointer:
        graph = build_graph(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": thread_id, "memory_enabled": False}}

        state = load_initial_state("applicant_strong.json")
        state["thread_id"] = thread_id
        graph_path_first: list[str] = []
        for update in graph.stream(state, config=config, stream_mode="updates"):
            graph_path_first.extend(update.keys())

        first_values = graph.get_state(config).values
        assert first_values["applicant"] is not None
        assert first_values["offer"] is not None
        assert "intake" in graph_path_first

        # A follow-up "message" arrives on the same, already-completed thread.
        # It is deliberately NOT a full re-submittable application payload —
        # if the supervisor still (incorrectly) routed to intake, intake_node
        # would fail to parse it as JSON and the run would end in an
        # escalation instead of a clean no-op resume to END.
        graph_path_second: list[str] = []
        for update in graph.stream(
            {"messages": [HumanMessage(content="Any update on my application?")]},
            config=config,
            stream_mode="updates",
        ):
            graph_path_second.extend(update.keys())

        second_values = graph.get_state(config).values

    # The supervisor itself always runs (it's the entry point / router), but
    # none of the actual worker nodes should re-execute for an already
    # completed application.
    worker_nodes_rerun = [n for n in graph_path_second if n != "supervisor"]
    assert worker_nodes_rerun == [], (
        f"expected the completed thread to resume straight to END with no "
        f"worker nodes re-run, but these nodes executed: {worker_nodes_rerun}"
    )
    assert graph_path_second == ["supervisor"]
    # The already-decided fields are untouched by the follow-up message.
    assert second_values["offer"] is not None
    assert second_values["offer"].principal == first_values["offer"].principal
    assert second_values["reflection_log"] == first_values["reflection_log"]
