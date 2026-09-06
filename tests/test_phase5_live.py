"""Optional live Phase 5 verification.

Run with:
    PHASE5_LIVE=1 pytest -v tests/test_phase5_live.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from src.rag.agentic_rag import agentic_policy_lookup

pytestmark = pytest.mark.live


@pytest.mark.skipif(
    os.getenv("PHASE5_LIVE") != "1",
    reason="Set PHASE5_LIVE=1 to execute the Phase 5 live RAG test",
)
def test_live_agentic_policy_lookup_uses_discretionary_tool():
    result = agentic_policy_lookup(
        "The applicant has a standard income tier and a DTI ratio of 0.41. "
        "Determine whether the lending policy needs to be consulted before "
        "explaining the automated outcome, and retrieve the relevant policy "
        "clause if needed."
    )

    assert result.called is True
    assert result.results

    evidence_dir = Path("evidence/run_transcripts")
    evidence_dir.mkdir(parents=True, exist_ok=True)
    (evidence_dir / "phase5_agentic_rag.json").write_text(
        json.dumps(
            {
                "ac_reference": "AC-11",
                "scenario": "live agentic lending-policy retrieval",
                "provider": "gemini-primary",
                "tool": "lending_policy_search",
                "called": result.called,
                "status": result.status,
                "provider": result.provider,
                "queries": list(result.queries),
                "result_clause_ids": [
                    item.get("clause_id") for item in result.results if item.get("clause_id")
                ],
                "result_count": len(result.results),
                "result_snippets": [
                    {
                        "clause_id": item.get("clause_id"),
                        "policy_id": item.get("policy_id"),
                        "heading": item.get("heading"),
                        "score": item.get("score"),
                    }
                    for item in result.results
                ],
                "steps": result.steps,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
