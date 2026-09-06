"""Post-decision durable-memory extraction."""
from __future__ import annotations

from datetime import datetime, timezone

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.llm.gateway import invoke_structured_with_fallback
from src.state.schema import CreditAssessment, KYCResult, LoanApplicationState, MemoryFact


class MemoryCandidate(BaseModel):
    """LLM-extracted durable fact before deterministic session metadata is added."""

    fact_type: str
    value: str
    confidence: float = Field(ge=0, le=1)


class MemoryExtractionResult(BaseModel):
    facts: list[MemoryCandidate] = Field(default_factory=list)


EXTRACTION_SYSTEM_PROMPT = """You extract durable facts from a completed synthetic loan application.
Only extract facts explicitly present in the supplied session summary. Never invent facts.
Do not store raw PII, raw prompts, MCP responses, routing internals, or free-text applicant notes.
Allowed fact types: employment, declared_income_band, kyc_outcome, credit_outcome,
preferred_term, prior_application_count.
Return only the requested structured object."""


def _session_summary(state: LoanApplicationState) -> str:
    applicant = state.get("applicant")
    kyc = state.get("kyc_result")
    credit = state.get("credit_assessment")
    offer = state.get("offer")

    lines = [f"user_id={state['user_id']}", f"thread_id={state['thread_id']}"]
    if applicant is not None:
        income = applicant.declared_income
        if income < 50_000:
            income_band = "under_50k"
        elif income <= 75_000:
            income_band = "50k_to_75k"
        else:
            income_band = "over_75k"
        lines.extend([
            f"employment={applicant.declared_employment}",
            f"declared_income_band={income_band}",
        ])
    if isinstance(kyc, KYCResult):
        lines.append(f"kyc_outcome={kyc.status}")
    if isinstance(credit, CreditAssessment):
        lines.append(f"credit_outcome={credit.decision}")
    if offer is not None:
        lines.append(f"proposed_term={offer.term_months}")
    return "\n".join(lines)


def _build_facts(candidates: list[MemoryCandidate], state: LoanApplicationState) -> list[MemoryFact]:
    now = datetime.now(timezone.utc).isoformat()
    facts: list[MemoryFact] = []
    allowed = {
        "employment",
        "declared_income_band",
        "kyc_outcome",
        "credit_outcome",
        "preferred_term",
        "prior_application_count",
    }
    for candidate in candidates:
        if candidate.confidence < 0.70 or candidate.fact_type not in allowed:
            continue
        facts.append(
            MemoryFact(
                fact_id=str(__import__("uuid").uuid4()),
                user_id=state["user_id"],
                fact_type=candidate.fact_type,
                value=candidate.value,
                importance=0.40,
                session_ts=now,
                thread_id=state["thread_id"],
                usage_count=0,
            )
        )
    return facts


def _langmem_extract(state: LoanApplicationState) -> list[MemoryCandidate]:
    from langmem import create_memory_manager
    from src.llm.gateway import _gemini_model

    manager = create_memory_manager(
        _gemini_model(),
        schemas=[MemoryCandidate],
        instructions=EXTRACTION_SYSTEM_PROMPT,
        enable_inserts=True,
        enable_updates=False,
        enable_deletes=False,
    )

    extracted = manager.invoke({
        "messages": [{
            "role": "user",
            "content": _session_summary(state),
        }]
    })

    candidates: list[MemoryCandidate] = []
    for item in extracted or []:
        content = getattr(item, "content", None)
        if isinstance(content, MemoryCandidate):
            candidates.append(content)
        elif isinstance(item, MemoryCandidate):
            candidates.append(item)
    return candidates


def _fallback_extract(state: LoanApplicationState) -> list[MemoryCandidate]:
    result = invoke_structured_with_fallback(
        MemoryExtractionResult,
        [
            SystemMessage(content=EXTRACTION_SYSTEM_PROMPT),
            HumanMessage(content=_session_summary(state)),
        ],
    )
    result = MemoryExtractionResult.model_validate(result)
    return result.facts


def extract_memory_facts(state: LoanApplicationState) -> list[MemoryFact]:
    """Prefer LangMem extraction; fall back to the shared structured LLM gateway."""
    try:
        candidates = _langmem_extract(state)
    except Exception:
        candidates = _fallback_extract(state)
    return _build_facts(candidates, state)
