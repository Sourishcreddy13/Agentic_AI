"""Deterministic test doubles and hermetic memory behavior for Phases 1–4."""
from __future__ import annotations
import os
import json
import re
from datetime import datetime, timezone

import pytest

from src.state.schema import ApplicantProfile, OfferDraft
from src.agents.credit_agent import CreditRationale
from src.agents.kyc_agent import KYCExplanation
from src.memory import runtime
from src.context.compress import ContextSummary
from src.rag.agentic_rag import PolicyLookupResult


def _last_human_text(messages):
    return str(messages[-1].content) if messages else ""


def fake_structured(schema, messages):
    text = "\n".join(str(m.content) for m in messages)

    if schema is ApplicantProfile:
        payload = json.loads(_last_human_text(messages))
        return ApplicantProfile.model_validate(payload)

    if schema is KYCExplanation:
        return KYCExplanation(
            rationale="Fake rationale based only on trusted MCP applicant facts.",
            confidence=0.95,
        )

    if schema is CreditRationale:
        decision_match = re.search(r"Decision:\s*([\w_]+)", text)
        decision = decision_match.group(1) if decision_match else "approve"
        return CreditRationale(
            rationale=(
                f"Fake rationale: policy gates reached decision={decision}. "
                "MCP bureau facts were used as the authoritative source."
            ),
            confidence=0.85,
        )

    if schema is ContextSummary:
        return ContextSummary(summary="Fake compressed context summary.")

    if schema is OfferDraft:
        income_match = re.search(r'"declared_income"\s*:\s*([0-9.]+)', text)
        decision_match = re.search(r'"decision"\s*:\s*"([^"]+)"', text)
        max_principal_match = re.search(r"max_principal\s*=\s*([0-9.]+)", text)

        income = float(income_match.group(1)) if income_match else 0.0
        decision = decision_match.group(1) if decision_match else "approve"
        max_principal = (
            float(max_principal_match.group(1))
            if max_principal_match
            else income * 3
        )

        if decision == "approve":
            return OfferDraft(
                principal=min(round(income * 3, 2), max_principal),
                apr=9.5,
                term_months=36,
                conditions=["Subject to final human underwriter review."],
                is_indicative=True,
            )

        return OfferDraft(
            principal=0,
            apr=0,
            term_months=0,
            conditions=[f"Not approved for indicative offer: {decision}."],
            is_indicative=True,
        )

    raise AssertionError(f"Unexpected schema in fake LLM: {schema}")


@pytest.fixture(autouse=True)
def deterministic_phase_models(monkeypatch, request):
    """Keep ordinary tests deterministic and memory-hermetic.

    Live-provider tests retain real model behavior. Memory integration tests
    retain real memory behavior. All other tests disable persistent memory so
    no test can accidentally mutate a shared Chroma collection.
    """
    is_live = request.node.get_closest_marker("live") is not None
    is_memory_test = request.node.get_closest_marker("memory_integration") is not None

    if not is_live:
        for module_name in (
            "src.agents.intake_agent",
            "src.agents.kyc_agent",
            "src.agents.credit_agent",
            "src.agents.offer_agent",
        ):
            module = __import__(module_name, fromlist=["invoke_structured_with_fallback"])
            monkeypatch.setattr(module, "invoke_structured_with_fallback", fake_structured)

    if not is_memory_test:
        monkeypatch.setattr(runtime, "memory_enabled", lambda config=None: False)

    # Keep ordinary tests hermetic: no real Gemini tool-selection or MCP policy retrieval.
    # Keep ordinary graph tests hermetic: the credit agent receives a
    # deterministic fake RAG result. The RAG implementation itself is tested
    # separately in tests/test_agentic_rag.py.
    if not is_live:
        fake_policy_lookup = (
            lambda task, max_steps=2: PolicyLookupResult(
                called=False,
                queries=(),
                results=(),
                steps=0,
            )
        )

        monkeypatch.setattr(
            "src.agents.credit_agent.agentic_policy_lookup",
            fake_policy_lookup,
        )

        monkeypatch.setattr(
            "src.context.compress.invoke_structured_with_fallback",
            fake_structured,
        )


def pytest_sessionfinish(session, exitstatus):
    """Write a safe aggregate test-run artifact; never include test payloads."""
    terminalreporter = session.config.pluginmanager.getplugin("terminalreporter")
    stats = getattr(terminalreporter, "stats", {}) if terminalreporter else {}

    def count(outcome):
        return len(stats.get(outcome, []))

    is_phase3_live = bool(os.environ.get("PHASE3_LIVE"))
    is_phase5_live = bool(os.environ.get("PHASE5_LIVE"))

    if is_phase3_live:
        artifact_name = "phase3_live_test_summary.json"
        command = "PHASE3_LIVE=1 pytest -q tests/test_phase3_live.py"
        scope = "Phase 3 live integration suite"
    elif is_phase5_live:
        artifact_name = "phase5_live_test_summary.json"
        command = "PHASE5_LIVE=1 pytest -q tests/test_phase5_live.py"
        scope = "Phase 5 live agentic-RAG suite"
    else:
        artifact_name = "test_run_summary.json"
        command = "pytest"
        scope = "pytest regression suite"

    payload = {
        "scope": scope,
        "command": command,
        "test_count_collected": getattr(session, "testscollected", 0),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "exit_status": exitstatus,
        "passed": count("passed"),
        "failed": count("failed"),
        "errors": count("error"),
        "skipped": count("skipped"),
        "xfailed": count("xfailed"),
        "xpassed": count("xpassed"),
    }

    from pathlib import Path

    evidence_dir = Path("evidence")
    evidence_dir.mkdir(parents=True, exist_ok=True)

    (evidence_dir / artifact_name).write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )