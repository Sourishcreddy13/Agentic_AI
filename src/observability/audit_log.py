"""Structured JSONL audit logging with aggressive PII redaction."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import CONFIG, PROJECT_ROOT

_SENSITIVE_KEY_RE = re.compile(
    r"(^|_)(full_name|dob(_synthetic)?|birth|declared_income|salary|employment|employer|"
    r"address|email|phone|mobile|ssn|pan|aadhaar|account|routing|card|notes|raw|"
    r"message|prompt|content|text|applicant_id|user_id)(_|$)",
    re.IGNORECASE,
)
_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d\s().-]{7,}\d)(?!\d)")


def _settings() -> dict[str, Any]:
    return CONFIG.get("observability", {}) or {}


def _log_path() -> Path:
    value = str(_settings().get("log_file", "data/logs/loan_origination.jsonl"))
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def fingerprint(value: Any) -> str:
    digest = hashlib.sha256(str(value).encode("utf-8")).hexdigest()
    return f"sha256:{digest[:16]}"


def _scrub_string(value: str) -> str:
    value = _EMAIL_RE.sub("[REDACTED_EMAIL]", value)
    value = _PHONE_RE.sub("[REDACTED_PHONE]", value)
    return value


def _sanitize(value: Any, key: str | None = None) -> Any:
    if key and _SENSITIVE_KEY_RE.search(key):
        return {"redacted": True, "fingerprint": fingerprint(value)}
    if isinstance(value, dict):
        return {str(k): _sanitize(v, str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    if isinstance(value, str):
        return _scrub_string(value)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return _scrub_string(str(value))


def _identifier(value: Any) -> str | None:
    if value is None:
        return None
    return fingerprint(value)


def log_event(event: str, *, user_id: Any = None, thread_id: Any = None, **fields: Any) -> None:
    """Write one sanitized structured event. Raw prompts/messages are never logged."""
    if not bool(_settings().get("enabled", True)):
        return

    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
    }
    if user_id is not None:
        record["user_id"] = _identifier(user_id)
    if thread_id is not None:
        record["thread_id"] = _identifier(thread_id)
    record.update({key: _sanitize(value, key) for key, value in fields.items()})

    path = _log_path()
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def summarize_state(state: dict[str, Any]) -> dict[str, Any]:
    """Produce a safe structural state summary for node-level tracing."""
    return {
        "has_applicant": state.get("applicant") is not None,
        "has_kyc": state.get("kyc_result") is not None,
        "has_credit": state.get("credit_assessment") is not None,
        "has_offer": state.get("offer") is not None,
        "message_count": len(state.get("messages") or []),
        "retry_count": state.get("retry_count", 0),
        "memory_hit_count": len(state.get("long_term_memory_hits") or []),
        "quarantined_count": len(state.get("quarantined_inputs") or []),
    }
