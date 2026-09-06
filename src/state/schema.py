"""
Typed state object for the loan-origination LangGraph (AC-01).

Root state is a TypedDict (cheap to checkpoint every superstep).
Domain payloads are Pydantic models, validated at node handoff
boundaries (AC-04) before being written back into state.

Identity model
--------------
user_id   → identifies the applicant across sessions (long-term memory key)
thread_id → identifies one application/conversation instance (checkpoint key)

A single user may have multiple threads:
  user_id="U-001" → thread-A (2025 application)
                  → thread-B (2026 re-application)
"""
from __future__ import annotations

from typing import TypedDict, Annotated, Literal, Optional
from operator import add

from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage


# --------------------------------------------------------------------------
# Pydantic sub-models  (validated at every node handoff boundary — AC-04)
# --------------------------------------------------------------------------

class ApplicantProfile(BaseModel):
    applicant_id: str
    full_name: str
    dob_synthetic: str
    declared_income: float = Field(ge=0)
    declared_employment: str
    # UNTRUSTED free text — quarantined before any prompt (NFR-03).
    raw_free_text_notes: str = ""


class KYCResult(BaseModel):
    status: Literal["pass", "fail", "manual_review"]
    checks_performed: list[str]
    risk_flags: list[str] = Field(default_factory=list)
    rationale: str = ""
    confidence: float = Field(ge=0, le=1)


class CreditAssessment(BaseModel):
    thin_file: bool
    bureau_score_synthetic: int = Field(ge=300, le=900)
    dti_ratio: float
    # decision is ALWAYS set by Python policy gates, never by the LLM directly.
    decision: Literal["approve", "decline", "manual_underwriting"]
    # rationale is the only field the LLM contributes.
    rationale: str
    confidence: float = Field(ge=0, le=1)


class OfferDraft(BaseModel):
    principal: float
    apr: float
    term_months: int
    conditions: list[str] = Field(default_factory=list)
    is_indicative: bool = True


class MemoryFact(BaseModel):
    """Durable semantic fact stored across application sessions."""

    fact_id: str
    user_id: str
    fact_type: Literal[
        "employment",
        "declared_income_band",
        "kyc_outcome",
        "credit_outcome",
        "preferred_term",
        "prior_application_count",
    ]
    value: str
    importance: float = Field(ge=0, le=1)
    session_ts: str
    thread_id: str
    usage_count: int = Field(default=0, ge=0)
    last_access_ts: str | None = None


class ReflectionNote(BaseModel):
    triggered_by: str
    # Explicit failure taxonomy — see src/agents/reflector.py
    action_taken: Literal["retry", "replan", "escalate_to_human"]
    detail: str


class ComplianceEvent(BaseModel):
    """A point-in-time compliance-relevant event, recorded purely for audit.

    This is deliberately a *separate* channel from ReflectionNote /
    reflection_log. reflection_log's last entry is what reflector_node reads
    to classify "the current failure" for the retry/replan/escalate control
    loop (src/agents/reflector.py); appending an unrelated audit note there
    would risk a later, unrelated node reading it as the most recent failure
    and misclassifying. ComplianceEvent records things that must be visible
    for audit/compliance (e.g. a detected prompt-injection attempt, or a
    KYC-fail referral) without ever being able to influence routing.
    """

    event_type: str
    detail: str = ""


# --------------------------------------------------------------------------
# Root graph state
# --------------------------------------------------------------------------

class LoanApplicationState(TypedDict):
    # --- identity ---
    # user_id: long-term memory key; survives across threads/sessions
    user_id: str
    # thread_id: checkpoint key; identifies one application instance
    thread_id: str

    # --- short-term / working memory (reducer appends) ---
    messages: Annotated[list[AnyMessage], add_messages]

    # --- domain payloads (Pydantic-validated at producing node) ---
    applicant: Optional[ApplicantProfile]
    kyc_result: Optional[KYCResult]
    credit_assessment: Optional[CreditAssessment]
    offer: Optional[OfferDraft]

    # --- routing / control ---
    next_node: Optional[str]
    retry_count: Annotated[int, add]
    reflection_log: Annotated[list[ReflectionNote], add]

    # --- context engineering ---
    quarantined_inputs: Annotated[list[str], add]
    compressed_summary: Optional[str]

    # --- long-term memory recall (written by memory layer in Phase 4) ---
    long_term_memory_hits: Optional[list[str]]

    # --- compliance audit trail (never consulted by routing — see ComplianceEvent) ---
    compliance_flags: Annotated[list[ComplianceEvent], add]


def new_state(thread_id: str, user_id: str = "default-user") -> LoanApplicationState:
    """
    Factory for a fresh, empty application state.

    user_id defaults to 'default-user' so existing tests that don't
    specify it continue to work without modification.
    """
    return LoanApplicationState(
        user_id=user_id,
        thread_id=thread_id,
        messages=[],
        applicant=None,
        kyc_result=None,
        credit_assessment=None,
        offer=None,
        next_node=None,
        retry_count=0,
        reflection_log=[],
        quarantined_inputs=[],
        compressed_summary=None,
        long_term_memory_hits=None,
        compliance_flags=[],
    )
