from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("chromadb")
pytest.importorskip("sentence_transformers")

pytestmark = pytest.mark.memory_integration


def test_cross_session_memory_persists_between_processes(tmp_path):
    store_path = tmp_path / "chroma_memory"
    write_evidence = tmp_path / "memory_write.json"
    read_evidence = tmp_path / "cross_session_memory.log"

    module = "tests.memory_process"

    subprocess.run(
        [sys.executable, "-m", module, "memory_write", str(store_path), str(write_evidence)],
        check=True,
    )
    subprocess.run(
        [sys.executable, "-m", module, "memory_read", str(store_path), str(read_evidence)],
        check=True,
    )

    write_result = json.loads(write_evidence.read_text())
    read_result = json.loads(read_evidence.read_text())

    assert write_result["user_id"] == "AC07-USER"
    assert any("Software Engineer" in fact for fact in read_result["recalled"])
    assert read_result["source_thread"] != read_result["current_thread_id"]

    evidence_dir = Path("evidence")
    evidence_dir.mkdir(parents=True, exist_ok=True)
    (evidence_dir / "memory_write.json").write_text(
        json.dumps({"ac_reference": "AC-07", **write_result}, indent=2),
        encoding="utf-8",
    )
    (evidence_dir / "cross_session_memory.log").write_text(
        json.dumps({"ac_reference": "AC-07", **read_result}, indent=2),
        encoding="utf-8",
    )
