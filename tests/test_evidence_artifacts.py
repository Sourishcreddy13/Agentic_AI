from __future__ import annotations

import json
from pathlib import Path


EVIDENCE_DIR = Path(__file__).resolve().parent.parent / "evidence"


def test_required_committed_evidence_artifacts_exist_and_are_safe():
    required = [
        "checkpoint_pause.json",
        "checkpoint_resume.json",
        "memory_write.json",
        "cross_session_memory.log",
        "ac06_same_session_memory.json",
        "observability_redaction.json",
        "observability_runtime.jsonl",
        "acceptance_evidence_manifest.json",
    ]

    for name in required:
        path = EVIDENCE_DIR / name
        assert path.exists(), f"missing committed evidence artifact: {name}"
        text = path.read_text(encoding="utf-8")
        assert "GOOGLE_API_KEY" not in text
        assert "GROQ_API_KEY" not in text
        assert "@" not in text

        if name.endswith(".jsonl"):
            records = [json.loads(line) for line in text.splitlines() if line.strip()]
            assert records
        else:
            record = json.loads(text)
            assert record

    ac06 = json.loads((EVIDENCE_DIR / "ac06_same_session_memory.json").read_text())
    assert ac06["ac_reference"] == "AC-06"

    ac07 = json.loads((EVIDENCE_DIR / "cross_session_memory.log").read_text())
    assert ac07["ac_reference"] == "AC-07"

    obs = json.loads((EVIDENCE_DIR / "observability_redaction.json").read_text())
    assert obs["raw_prompt_logged"] is False
    assert obs["raw_message_logged"] is False
    runtime_log = (EVIDENCE_DIR / "observability_runtime.jsonl").read_text()
    assert "Synthetic Person" not in runtime_log
    assert "85000" not in runtime_log
    assert "synthetic@example.com" not in runtime_log
    assert "sha256:" in runtime_log

    manifest = json.loads((EVIDENCE_DIR / "acceptance_evidence_manifest.json").read_text())
    assert manifest["AC-05"]["artifact"] == "checkpoint_pause.json"
    assert manifest["AC-06"]["artifact"] == "ac06_same_session_memory.json"
    assert manifest["AC-07"]["artifact"] == "cross_session_memory.log"
    assert manifest["AC-11"]["test"] == "tests/test_phase5_live.py"

    summary = json.loads(
        (EVIDENCE_DIR / "test_run_summary.json").read_text()
    )

    assert summary["command"] == "pytest"
    assert "exit_status" in summary
