"""Subprocess worker used by AC-05 and AC-07 persistence tests."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from langchain_core.messages import HumanMessage

from src.graph.build_graph import build_graph
from src.state.schema import MemoryFact, new_state
from src.memory import runtime


def _patch_deterministic_models():
    from tests.conftest import fake_structured
    for module_name in (
        "src.agents.intake_agent",
        "src.agents.kyc_agent",
        "src.agents.credit_agent",
        "src.agents.offer_agent",
    ):
        module = __import__(module_name, fromlist=["invoke_structured_with_fallback"])
        module.invoke_structured_with_fallback = fake_structured


def _strong_state(thread_id: str, user_id: str):
    path = Path(__file__).resolve().parent.parent / "sample_inputs" / "applicant_strong.json"
    state = new_state(thread_id=thread_id, user_id=user_id)
    state["messages"] = [HumanMessage(content=path.read_text())]
    return state


def checkpoint_pause(db_path: str, evidence: str):
    _patch_deterministic_models()
    state = _strong_state("AC05-TEST", "AC05-USER")
    with __import__("src.memory.checkpointer", fromlist=["get_checkpointer"]).get_checkpointer(db_path) as saver:
        graph = build_graph(checkpointer=saver, interrupt_before=["credit_assessment"])
        config = {"configurable": {"thread_id": "AC05-TEST", "memory_enabled": False}}
        result = graph.invoke(state, config)
        snapshot = graph.get_state(config)

    payload = {
        "thread_id": "AC05-TEST",
        "kyc_present": result["kyc_result"] is not None,
        "next_nodes": list(snapshot.next),
        "paused_before_credit": "credit_assessment" in snapshot.next,
    }
    Path(evidence).write_text(json.dumps(payload, indent=2))
    assert payload["kyc_present"]
    assert payload["paused_before_credit"]


def checkpoint_resume(db_path: str, evidence: str):
    _patch_deterministic_models()
    with __import__("src.memory.checkpointer", fromlist=["get_checkpointer"]).get_checkpointer(db_path) as saver:
        graph = build_graph(checkpointer=saver)
        config = {"configurable": {"thread_id": "AC05-TEST", "memory_enabled": False}}
        result = graph.invoke(None, config)

    payload = {
        "thread_id": "AC05-TEST",
        "kyc_present": result["kyc_result"] is not None,
        "offer_present": result["offer"] is not None,
        "resumed_credit_decision": getattr(result["credit_assessment"], "decision", None),
    }
    Path(evidence).write_text(json.dumps(payload, indent=2))
    assert payload["kyc_present"]
    assert payload["offer_present"]


def memory_write(db_path: str, evidence: str):
    _patch_deterministic_models()
    from src.agents import memory_agent

    fact = MemoryFact(
        fact_id="AC07-EMPLOYMENT",
        user_id="AC07-USER",
        fact_type="employment",
        value="Software Engineer (synthetic)",
        importance=0.55,
        session_ts="2026-09-05T00:00:00+00:00",
        thread_id="AC07-THREAD-A",
        usage_count=0,
    )
    memory_agent.extract_memory_facts = lambda state: [fact]

    state = _strong_state("AC07-THREAD-A", "AC07-USER")
    config = {
        "configurable": {
            "thread_id": "AC07-THREAD-A",
            "memory_enabled": True,
            "memory_store_path": db_path,
        }
    }
    result = build_graph().invoke(state, config)
    Path(evidence).write_text(json.dumps({
        "user_id": "AC07-USER",
        "thread_id": "AC07-THREAD-A",
        "facts": [fact.model_dump(mode="json")],
        "offer_present": result["offer"] is not None,
    }, indent=2))


def memory_read(db_path: str, evidence: str):
    _patch_deterministic_models()
    state = _strong_state("AC07-THREAD-B", "AC07-USER")
    config = {
        "configurable": {
            "thread_id": "AC07-THREAD-B",
            "memory_enabled": True,
            "memory_store_path": db_path,
        }
    }
    result = build_graph().invoke(state, config)
    recalled = result.get("long_term_memory_hits") or []
    Path(evidence).write_text(json.dumps({
        "user_id": "AC07-USER",
        "current_thread_id": "AC07-THREAD-B",
        "recalled": recalled,
        "source_thread": "AC07-THREAD-A",
    }, indent=2))
    assert any("Software Engineer" in fact for fact in recalled)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["checkpoint_pause", "checkpoint_resume", "memory_write", "memory_read"])
    parser.add_argument("db_path")
    parser.add_argument("evidence")
    args = parser.parse_args()

    Path(args.evidence).parent.mkdir(parents=True, exist_ok=True)
    if args.mode == "checkpoint_pause":
        checkpoint_pause(args.db_path, args.evidence)
    elif args.mode == "checkpoint_resume":
        checkpoint_resume(args.db_path, args.evidence)
    elif args.mode == "memory_write":
        memory_write(args.db_path, args.evidence)
    elif args.mode == "memory_read":
        memory_read(args.db_path, args.evidence)


if __name__ == "__main__":
    main()
