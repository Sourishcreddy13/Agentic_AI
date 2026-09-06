"""Shared PII-redaction primitives.

This used to be implemented twice — once in src/observability/audit_log.py
for the project's structured event log, and again, independently, in
cli.py for the CLI's own execution-log file. The two regexes had already
started to drift (audit_log.py: `dob(_synthetic)?`; cli.py: `dob|
dob_synthetic`) even though they were meant to express the same rule. This
module is the single place that rule (and the accompanying sanitize/
fingerprint helpers) is defined; both audit_log.py and cli.py now import
from here instead of keeping their own copies.

`sanitize()` combines what both prior copies did: field-name-based
redaction (a value under a sensitive key becomes a fingerprint, never the
raw value), Pydantic-model unwrapping, LangChain-message-like content
fingerprinting, and best-effort email/phone scrubbing of any remaining free
text — so nothing loses coverage by consolidating.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any

SENSITIVE_KEY_RE = re.compile(
    r"(^|_)(full_name|dob(_synthetic)?|birth|declared_income|salary|employment|employer|"
    r"address|email|phone|mobile|ssn|pan|aadhaar|account|routing|card|notes|raw|"
    r"message|prompt|content|text|applicant_id|user_id)(_|$)",
    re.IGNORECASE,
)
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d\s().-]{7,}\d)(?!\d)")


def fingerprint(value: Any) -> str:
    digest = hashlib.sha256(str(value).encode("utf-8")).hexdigest()
    return f"sha256:{digest[:16]}"


def scrub_string(value: str) -> str:
    value = EMAIL_RE.sub("[REDACTED_EMAIL]", value)
    value = PHONE_RE.sub("[REDACTED_PHONE]", value)
    return value


def sanitize(value: Any, key: str | None = None) -> Any:
    """Recursively redact a value for safe persistence in a log/evidence file.

    - A value under a sensitive-looking key is replaced with a fingerprint.
    - Pydantic models are unwrapped via model_dump() and sanitized field by
      field (so a nested sensitive field is still caught by key).
    - dict / list / tuple / set are recursed into.
    - A LangChain-message-like object (has .content) is reduced to its type
      plus a fingerprint of its content, never the raw content.
    - Plain strings are scrubbed for embedded emails/phone numbers even when
      their key isn't recognized as sensitive.
    - Everything else is stringified and scrubbed the same way.
    """
    if key and SENSITIVE_KEY_RE.search(key):
        return {"redacted": True, "fingerprint": fingerprint(value)}

    if hasattr(value, "model_dump"):
        return sanitize(value.model_dump(mode="json"), key)

    if isinstance(value, dict):
        return {str(k): sanitize(v, str(k)) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [sanitize(item) for item in value]

    if hasattr(value, "content") and not isinstance(value, (str, bytes)):
        return {
            "type": getattr(value, "type", value.__class__.__name__),
            "content_fingerprint": fingerprint(value.content),
        }

    if isinstance(value, str):
        return scrub_string(value)

    if isinstance(value, (int, float, bool)) or value is None:
        return value

    return scrub_string(str(value))
