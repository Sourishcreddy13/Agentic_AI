"""
Context isolation (NFR-03 / Context-Isolation Rule).

Applicant-submitted free text is untrusted. It is quarantined as data and is
never used as an instruction source. Raw applicant text is deliberately
excluded from compression/summarisation; downstream model calls receive only
trusted structured projections or explicit quarantine metadata.

Security boundary vs. compliance signal — important distinction
-----------------------------------------------------------------
`SUSPICIOUS_PATTERNS` below is a best-effort *compliance/audit* signal
only. Whether or not free text matches one of these patterns has **no**
effect on the actual isolation guarantee: `quarantine_applicant_text`
unconditionally wraps every non-empty `raw_free_text_notes` value in an
`<untrusted_applicant_input>` envelope and the intake/compression paths
never place that raw text into a model prompt, regardless of the match
result (see `intake_agent.py` and `build_safe_summary_projection` below).
A missed pattern therefore only means a real injection attempt goes
un-flagged for compliance review (no `ComplianceEvent` is raised) — it
does **not** create a path for injected text to reach a prompt. This list
is intentionally kept to literal, lowercase substrings (cheap,
deterministic, auditable) rather than a fuzzy/ML classifier, and is not
intended to be exhaustive; broaden it as new real-world phrasing patterns
are observed.
"""
from __future__ import annotations

import json
from typing import Any


SUSPICIOUS_PATTERNS = [
    "ignore previous",
    "ignore all previous",
    "ignore your instructions",
    "disregard your instructions",
    "disregard previous",
    "forget previous instructions",
    "forget your instructions",
    "override your instructions",
    "new instructions:",
    "system:",
    "assistant:",
    "you are now",
    "act as if you",
    "pretend you are",
]

# These are the only structured application fields permitted into the trusted
# summary projection. Free-text fields are explicitly excluded.
TRUSTED_APPLICATION_FIELDS = (
    "applicant_id",
    "full_name",
    "dob_synthetic",
    "declared_income",
    "declared_employment",
    "loan_amount",
    "loan_purpose",
    "term_months",
)


def quarantine_applicant_text(raw_text: str) -> tuple[str, bool]:
    """Return an explicit quarantine envelope plus a prompt-injection flag."""
    if not raw_text:
        return "", False

    lowered = raw_text.lower()
    suspicious = any(pattern in lowered for pattern in SUSPICIOUS_PATTERNS)
    wrapped = (
        "<untrusted_applicant_input>"
        f"{raw_text}"
        "</untrusted_applicant_input>"
    )
    return wrapped, suspicious


def _parse_json_object(content: Any) -> dict[str, Any] | None:
    if not isinstance(content, str):
        return None
    try:
        payload = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return None
    return payload if isinstance(payload, dict) else None


def project_message_for_summary(message: Any) -> dict[str, Any]:
    """
    Project one chat message into a summarisation-safe representation.

    Applicant human messages are reduced to trusted structured fields only.
    Any raw free text is represented by metadata, never copied into the
    summarisation prompt. Non-applicant messages retain their role and text.
    """
    message_type = getattr(message, "type", "message")
    content = getattr(message, "content", "")

    if message_type == "human":
        payload = _parse_json_object(content)
        if payload is not None:
            trusted = {
                key: payload[key]
                for key in TRUSTED_APPLICATION_FIELDS
                if key in payload
            }
            return {
                "type": "applicant_structured_data",
                "trusted_fields": trusted,
                "untrusted_free_text_present": bool(
                    str(payload.get("raw_free_text_notes") or "").strip()
                ),
            }

        # Unknown human text is untrusted by default. Keep only metadata.
        return {
            "type": "applicant_untrusted_text",
            "content_omitted": True,
        }

    return {
        "type": str(message_type),
        "content": str(content),
    }


def build_safe_summary_projection(messages: list[Any]) -> list[dict[str, Any]]:
    """Build the only representation permitted to enter context compression."""
    return [project_message_for_summary(message) for message in messages]
