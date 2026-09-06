from __future__ import annotations

import json

import pytest
from langchain_core.messages import HumanMessage

from src.agents.intake_agent import intake_node
from src.agents.memory_agent import memory_consolidation_node
from src.memory import extraction
from src.state.schema import MemoryFact, new_state

pytestmark = pytest.mark.memory_integration


def test_ac06_same_session_memory_recalls_prior_turn(tmp_path, monkeypatch):
    """AC-06: a later turn in the same thread can recall a durable prior-turn fact."""
    store_path = str(tmp_path / "chroma_memory")
    user_id = "AC06-USER"
    thread_id = "AC06-THREAD"

    fact = MemoryFact(
        fact_id="AC06-EMPLOYMENT",
        user_id=user_id,
        fact_type="employment",
        value="Software Engineer (synthetic)",
        importance=0.40,
        session_ts="2026-09-05T00:00:00+00:00",
        thread_id=thread_id,
        usage_count=0,
    )

    monkeypatch.setattr(
        extraction,
        "extract_memory_facts",
        lambda state: [fact],
    )

    first_turn = new_state(thread_id=thread_id, user_id=user_id)
    first_turn["offer"] = None
    first_turn["messages"] = [
        HumanMessage(content="First turn completed with synthetic employment fact.")
    ]

    memory_consolidation_node(
        first_turn,
        {
            "configurable": {
                "thread_id": thread_id,
                "memory_enabled": True,
                "memory_store_path": store_path,
            }
        },
    )

    second_turn = new_state(thread_id=thread_id, user_id=user_id)
    second_turn["messages"] = [
        HumanMessage(
            content=json.dumps(
                {
                    "applicant_id": "SYN-AC06",
                    "full_name": "Synthetic Applicant",
                    "dob_synthetic": "1990-01-01",
                    "declared_income": 80000,
                    "declared_employment": "Software Engineer",
                }
            )
        )
    ]

    result = intake_node(
        second_turn,
        {
            "configurable": {
                "thread_id": thread_id,
                "memory_enabled": True,
                "memory_store_path": store_path,
            }
        },
    )

    hits = result.get("long_term_memory_hits") or []
    assert any("Software Engineer" in value for value in hits)
    assert result["next_node"] == "kyc_check"

    evidence_path = "evidence/ac06_same_session_memory.json"
    with open(evidence_path, "w", encoding="utf-8") as evidence_file:
        json.dump(
            {
                "ac_reference": "AC-06",
                "scenario": "same-session prior-turn semantic memory recall",
                "user_id": user_id,
                "thread_id": thread_id,
                "prior_turn_fact": {
                    "fact_type": fact.fact_type,
                    "value": fact.value,
                },
                "later_turn_recalled": True,
                "recalled_values": hits,
                "decision_authority_changed": False,
            },
            evidence_file,
            indent=2,
        )
