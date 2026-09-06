from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("langgraph")
pytest.importorskip("langgraph.checkpoint.sqlite")


def test_checkpoint_resume_across_processes(tmp_path):
    db_path = tmp_path / "checkpoints.sqlite"
    pause_evidence = tmp_path / "checkpoint_pause.json"
    resume_evidence = tmp_path / "checkpoint_resume.json"

    module = "tests.memory_process"

    subprocess.run(
        [sys.executable, "-m", module, "checkpoint_pause", str(db_path), str(pause_evidence)],
        check=True,
    )
    subprocess.run(
        [sys.executable, "-m", module, "checkpoint_resume", str(db_path), str(resume_evidence)],
        check=True,
    )

    paused = json.loads(pause_evidence.read_text())
    resumed = json.loads(resume_evidence.read_text())

    assert paused["paused_before_credit"] is True
    assert resumed["kyc_present"] is True
    assert resumed["offer_present"] is True

    evidence_dir = Path("evidence")
    evidence_dir.mkdir(parents=True, exist_ok=True)
    (evidence_dir / "checkpoint_pause.json").write_text(
        json.dumps({
            "ac_reference": "AC-05",
            "scenario": "SQLite checkpoint pause before credit assessment",
            **paused,
        }, indent=2),
        encoding="utf-8",
    )
    (evidence_dir / "checkpoint_resume.json").write_text(
        json.dumps({
            "ac_reference": "AC-05",
            "scenario": "SQLite checkpoint resume in a new process",
            **resumed,
        }, indent=2),
        encoding="utf-8",
    )
