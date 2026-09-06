from __future__ import annotations

import json

def test_audit_log_redacts_pii_and_hashes_identifiers(monkeypatch, tmp_path):
    from src.observability import audit_log

    monkeypatch.setitem(audit_log.CONFIG["observability"], "log_file", str(tmp_path / "events.jsonl"))
    monkeypatch.setitem(audit_log.CONFIG["observability"], "enabled", True)

    audit_log.log_event(
        "test_event",
        user_id="SYN-0001",
        thread_id="THREAD-1",
        applicant_id="SYN-0001",
        full_name="Synthetic Person",
        declared_income=85000,
        email="synthetic@example.com",
        tool="bureau_check",
    )

    record = json.loads((tmp_path / "events.jsonl").read_text().strip())
    text = (tmp_path / "events.jsonl").read_text()
    assert record["user_id"].startswith("sha256:")
    assert record["thread_id"].startswith("sha256:")
    assert record["applicant_id"]["redacted"] is True
    assert "Synthetic Person" not in text
    assert "85000" not in text
    assert "synthetic@example.com" not in text


def test_audit_log_records_safe_runtime_fields(monkeypatch, tmp_path):
    from src.observability import audit_log

    monkeypatch.setitem(audit_log.CONFIG["observability"], "log_file", str(tmp_path / "events.jsonl"))
    monkeypatch.setitem(audit_log.CONFIG["observability"], "enabled", True)

    audit_log.log_event(
        "llm_invocation_completed",
        provider="gemini-primary",
        schema="CreditRationale",
        success=True,
    )

    record = json.loads((tmp_path / "events.jsonl").read_text().strip())
    assert record["event"] == "llm_invocation_completed"
    assert record["provider"] == "gemini-primary"
    assert record["schema"] == "CreditRationale"
