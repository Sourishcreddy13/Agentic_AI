"""
Credit-assessment worker (AC-02, AC-04).

Architectural principle
-----------------------
Python owns:  hard eligibility gates, DTI threshold, thin-file flag,
              offer decision, routing signals.
Gemini owns:  rationale text, confidence score for the reasoning.

The LLM never sets `decision`. Python reads bureau facts from the MCP
tool and applies deterministic policy gates. The LLM then explains the
decision in `rationale`.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.llm.gateway import invoke_structured_with_fallback
from src.mcp_client import invoke_mcp_tool_sync
from src.state.schema import LoanApplicationState, CreditAssessment, ReflectionNote
from src.context.middleware import prepare_worker_context
from src.observability.audit_log import log_event
from src.rag.agentic_rag import agentic_policy_lookup


# --------------------------------------------------------------------------
# Internal: what the LLM contributes (rationale only, never the decision)
# --------------------------------------------------------------------------

class CreditRationale(BaseModel):
    """LLM output for credit reasoning. Decision is NOT set here."""
    rationale: str
    confidence: float = Field(ge=0, le=1)


class BureauCheckResult(BaseModel):
    applicant_id: str
    found_in_bureau: bool
    synthetic_score: int = Field(ge=300, le=900)
    delinquencies_24m: int = Field(ge=0)
    thin_file: bool
    dti_estimate: float = Field(ge=0)


# --------------------------------------------------------------------------
# Deterministic policy gates (Python — not LLM)
# --------------------------------------------------------------------------

def _apply_policy_gates(
    bureau: BureauCheckResult,
    annual_income: float,
) -> tuple[str, float]:
    """Apply the synthetic DTI/score policy deterministically."""
    # THIN-001: thin-file cases are manual, except DTI > 0.60 is an explicit
    # decline condition for thin-file applicants.
    if bureau.thin_file:
        if bureau.dti_estimate > 0.60:
            return "decline", 1.0
        return "manual_underwriting", 1.0

    # PRICE-001: scores below 500 are not automatically approvable.
    if bureau.synthetic_score < 500:
        return "manual_underwriting", 1.0

    # PRICE-001: 500–599 is manual review.
    if bureau.synthetic_score < 600:
        return "manual_underwriting", 1.0

    # DTI-001: >0.50 always goes to credit officer.
    if bureau.dti_estimate > 0.50:
        return "manual_underwriting", 1.0

    # DTI-001: standard cap 0.40; premium (>75k pa) cap 0.50.
    max_dti = 0.50 if annual_income > 75_000 else 0.40
    if bureau.dti_estimate > max_dti:
        return "decline", 1.0

    return "approve", 0.85


# --------------------------------------------------------------------------
# System prompt — LLM is asked for explanation only, not decision
# --------------------------------------------------------------------------

RATIONALE_SYSTEM_PROMPT = """You are a credit-assessment specialist producing \
a plain-English rationale for a lending decision that has already been made \
by automated policy gates.

The decision is final. Your role is to explain it clearly using the bureau \
facts provided. Do not suggest reversing or overriding the decision.
Return only the requested structured rationale."""


def credit_assessment_node(state: LoanApplicationState) -> dict:
    applicant = state["applicant"]
    assert applicant is not None, "credit_assessment_node reached without an applicant profile"

    # Step 1 — Fetch bureau facts from MCP (deterministic tool)
    try:
        raw = invoke_mcp_tool_sync(
            "bureau_check",
            {
                "applicant_id": applicant.applicant_id,
                "declared_income": applicant.declared_income,
            },
        )
        bureau = BureauCheckResult.model_validate(raw)
    except Exception as exc:
        note = ReflectionNote(
            triggered_by="mcp_bureau_check_failure",
            action_taken="retry",
            detail=f"bureau_check MCP call failed: {str(exc)[:240]}",
        )
        return {"reflection_log": [note]}

    # Step 2 — Python applies hard policy gates (no LLM involved)
    decision, base_confidence = _apply_policy_gates(bureau, applicant.declared_income)

    # Step 3 — Agentic policy retrieval is discretionary. It supplies context only.
    selected, _compression = prepare_worker_context(state, "credit")
    rag_task = (
        f"Current credit decision context: decision={decision}; "
        f"annual_income={applicant.declared_income}; "
        f"bureau_score={bureau.synthetic_score}; dti={bureau.dti_estimate}. "
        "Determine whether lending-policy or eligibility text is needed to explain "
        "this decision. If needed, retrieve the relevant policy clause."
    )
    policy_lookup = agentic_policy_lookup(rag_task)
    log_event(
        "credit_policy_context_ready",
        user_id=state.get("user_id"),
        thread_id=state.get("thread_id"),
        rag_status=policy_lookup.status,
        rag_provider=policy_lookup.provider,
        rag_result_count=len(policy_lookup.results),
        rag_error_type=policy_lookup.error_type,
    )

    # Step 4 — LLM generates rationale and refines confidence (explanation only).
    try:
        if policy_lookup.status == "retrieved":
            policy_text = "\n".join(
                f"{item.get('clause_id', 'policy')}: {item.get('snippet', '')}"
                for item in policy_lookup.results
            )
        elif policy_lookup.status == "failed":
            policy_text = (
                "Policy retrieval was unavailable; continue using the "
                "deterministic Python policy decision."
            )
        else:
            policy_text = "No additional policy lookup was requested."
        context_summary = selected.get("compressed_summary") or "No prior conversation summary."
        llm_out = invoke_structured_with_fallback(
            CreditRationale,
            [
                SystemMessage(content=RATIONALE_SYSTEM_PROMPT),
                HumanMessage(content=(
                    f"Decision: {decision}\n"
                    f"Bureau facts:\n{bureau.model_dump_json()}\n"
                    f"Prior-context summary: {context_summary}\n"
                    f"Agentic policy lookup result:\n{policy_text}"
                )),
            ],
        )
        rationale_obj = CreditRationale.model_validate(llm_out)
        # For approve: use LLM's confidence; for hard gates: keep deterministic 1.0
        final_confidence = rationale_obj.confidence if decision == "approve" else base_confidence
        rationale = rationale_obj.rationale
    except Exception as exc:
        # LLM failure for rationale is non-fatal — use a fallback explanation
        note = ReflectionNote(
            triggered_by="llm_rationale_failure",
            action_taken="retry",
            detail=f"Rationale generation failed: {str(exc)[:240]}",
        )
        # Stub-vs-escalate is decided from how many times *rationale
        # generation specifically* has already failed in this run, not the
        # graph-wide retry_count. Using the global counter here previously
        # meant an unrelated earlier retry (e.g. a transient KYC hiccup)
        # silently consumed this node's own one-stub-then-escalate budget.
        rationale_failure_count = sum(
            1
            for prior_note in state.get("reflection_log", [])
            if prior_note.triggered_by == "llm_rationale_failure"
        )
        if rationale_failure_count == 0:
            rationale = f"Decision: {decision}. Bureau facts used (rationale unavailable)."
            final_confidence = base_confidence
        else:
            return {"reflection_log": [note]}

    # Step 4 — Assemble and validate the full CreditAssessment
    try:
        assessment = CreditAssessment.model_validate({
            "thin_file": bureau.thin_file,
            "bureau_score_synthetic": bureau.synthetic_score,
            "dti_ratio": bureau.dti_estimate,
            "decision": decision,           # from Python gate — not LLM
            "rationale": rationale,         # from LLM
            "confidence": final_confidence,
        })
    except Exception as exc:
        note = ReflectionNote(
            triggered_by="credit_assembly_error",
            action_taken="retry",
            detail=f"CreditAssessment validation failed: {str(exc)[:240]}",
        )
        return {"reflection_log": [note]}

    return {
        "credit_assessment": assessment,
        "next_node": "offer_draft",
        "messages": [
            AIMessage(
                content=(
                    f"Credit decision: {decision} "
                    f"(score={bureau.synthetic_score}, dti={bureau.dti_estimate})."
                )
            )
        ],
    }
