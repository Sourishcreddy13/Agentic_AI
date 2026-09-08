"""Phase 3 intake worker: deterministic extraction with trusted-field guards.

Intake extraction is deterministic, not LLM-based. The applicant profile is
built directly from the trusted structured application fields — there is no
free-form extraction step, so there is nothing for an LLM to add. The profile
is still validated through the ApplicantProfile Pydantic schema at the
handoff boundary (AC-04), so a shape mismatch on new/unexpected input is
still caught and routed to reflection/retry rather than silently accepted.
"""
from __future__ import annotations

import json

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from pydantic import ValidationError

from src.context.quarantine import quarantine_applicant_text
from src.memory import runtime
from src.context.middleware import prepare_worker_context
from src.state.schema import ApplicantProfile, ComplianceEvent, LoanApplicationState, ReflectionNote
from src.observability.audit_log import log_event


TRUSTED_FIELDS = (
    "applicant_id",
    "full_name",
    "dob_synthetic",
    "declared_income",
    "declared_employment",
)


def _parse_application(raw: str) -> tuple[dict, str]:
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("Application payload must be a JSON object.")

    trusted = {field: payload[field] for field in TRUSTED_FIELDS if field in payload}
    notes = str(payload.get("raw_free_text_notes") or "")
    return trusted, notes


def intake_node(state: LoanApplicationState, config: RunnableConfig) -> dict:
    if not state["messages"]:
        return {
            "reflection_log": [
                ReflectionNote(
                    triggered_by="missing_application_payload",
                    action_taken="escalate_to_human",
                    detail="No application message found in state.",
                )
            ],
            "next_node": "reflector",
        }

    raw = str(state["messages"][-1].content)

    try:
        trusted_payload, raw_notes = _parse_application(raw)
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        return {
            "reflection_log": [
                ReflectionNote(
                    triggered_by="intake_validation_error",
                    action_taken="retry",
                    detail=str(exc)[:500],
                )
            ],
            "next_node": "reflector",
        }

    # Long-term memory is retrieved before downstream workers run. It is
    # contextual input only; Python policy functions never consume it.
    memory_hits: list[str] = []
    if runtime.memory_enabled(config) and trusted_payload.get("applicant_id"):
        try:
            from src.memory.long_term_store import get_memory_store
            store = get_memory_store(runtime.memory_store_path(config))
            query = (
                "prior loan application facts, employment, prior outcomes, "
                "and durable user preferences"
            )
            facts = store.search(state["user_id"], query, k=5)
            memory_hits = [fact.value for fact in facts]
        except Exception:
            memory_hits = []

    selected, compression = prepare_worker_context(state, "intake")

    # Deterministic extraction: the applicant profile IS the trusted
    # structured payload, validated against the schema. Applicant-submitted
    # free text remains untrusted and is never a source for ApplicantProfile
    # field values; it is quarantined separately below. Any shape mismatch
    # (e.g. new/unfamiliar test data) surfaces as a ValidationError here and
    # is routed to reflection/retry rather than propagating downstream.
    try:
        profile = ApplicantProfile.model_validate(trusted_payload)
    except (ValidationError, ValueError, TypeError) as exc:
        return {
            "reflection_log": [
                ReflectionNote(
                    triggered_by="intake_validation_error",
                    action_taken="retry",
                    detail=str(exc)[:500],
                )
            ],
            "next_node": "reflector",
        }

    updates: dict = {
        "compressed_summary": selected.get("compressed_summary"),
        "applicant": profile,
        "next_node": "kyc_check",
        "long_term_memory_hits": memory_hits,
        # "write" context engineering: a short, non-sensitive transcript
        # entry per stage. This keeps state["messages"] a genuine growing
        # record of the workflow (rather than a single static entry), which
        # is what makes long-thread compression (NFR-08) a reachable path
        # for a returning applicant with many prior submissions on the same
        # thread — see tests/test_multi_turn_compression.py.
        "messages": [AIMessage(content=f"Intake completed for applicant {profile.applicant_id}.")],
    }
    if raw_notes:
        wrapped_notes, suspicious = quarantine_applicant_text(raw_notes)
        updates["quarantined_inputs"] = [wrapped_notes]
        if suspicious:
            log_event(
                "applicant_text_injection_pattern_detected",
                user_id=state.get("user_id"),
                thread_id=state.get("thread_id"),
                action="quarantined_and_excluded_from_prompts",
            )
            # Compliance visibility: a detected injection attempt is recorded
            # as a ComplianceEvent (a channel reflector_node never reads),
            # not a ReflectionNote — so it is visible for audit/compliance
            # review without any risk of being misread as "the current
            # failure" by the reflector's retry/replan/escalate classifier.
            # next_node above is unchanged and the graph still continues to
            # kyc_check exactly as before — see
            # tests/test_routing.py::test_injection_in_free_text_does_not_trigger_reflection_bypass.
            updates["compliance_flags"] = [
                ComplianceEvent(
                    event_type="suspected_prompt_injection_in_free_text",
                    detail=(
                        "Applicant-submitted free text matched a known "
                        "prompt-injection pattern and was quarantined. "
                        "Recorded for compliance review; routing and "
                        "extraction were not affected."
                    ),
                )
            ]

    return updates
