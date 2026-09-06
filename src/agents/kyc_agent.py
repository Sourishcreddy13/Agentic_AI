"""Phase 3 KYC worker: MCP facts + deterministic status gate + LLM explanation."""
from __future__ import annotations

from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage

from src.llm.gateway import invoke_structured_with_fallback
from src.context.middleware import prepare_worker_context
from src.mcp_client import invoke_mcp_tool_sync
from src.state.schema import LoanApplicationState, KYCResult, ReflectionNote


class ApplicantLookupResult(BaseModel):
    applicant_id: str
    found: bool
    prior_applications: int = Field(ge=0)
    synthetic_watchlist_hit: bool


class KYCExplanation(BaseModel):
    """LLM contribution for KYC; it cannot set the final status."""
    rationale: str
    confidence: float = Field(ge=0, le=1)


KYC_SYSTEM_PROMPT = """You are a KYC specialist in a synthetic loan-origination workflow.
The KYC outcome has already been determined by trusted Python policy gates from
the MCP applicant lookup result. You must not change or override the outcome.
Provide a concise rationale using only the supplied MCP facts. Never infer a
watchlist hit from applicant name or free text. Return only structured rationale
and confidence."""


def _apply_kyc_policy(lookup: ApplicantLookupResult) -> tuple[str, list[str]]:
    if lookup.synthetic_watchlist_hit:
        return "fail", ["synthetic_watchlist_hit"]
    if not lookup.found:
        return "manual_review", ["applicant_not_found"]
    return "pass", []


def kyc_check_node(state: LoanApplicationState) -> dict:
    applicant = state["applicant"]
    assert applicant is not None, "kyc_check_node reached without an applicant profile"

    try:
        raw = invoke_mcp_tool_sync(
            "applicant_lookup",
            {"applicant_id": applicant.applicant_id},
        )
        lookup = ApplicantLookupResult.model_validate(raw)
    except Exception as exc:
        return {"reflection_log": [ReflectionNote(
            triggered_by="mcp_applicant_lookup_failure",
            action_taken="retry",
            detail=f"applicant_lookup MCP call failed: {str(exc)[:240]}",
        )]}

    status, risk_flags = _apply_kyc_policy(lookup)
    checks_performed = ["synthetic_history_lookup", "watchlist_screen"]

    selected, _compression = prepare_worker_context(state, "kyc")
    try:
        explanation = invoke_structured_with_fallback(
            KYCExplanation,
            [
                SystemMessage(content=KYC_SYSTEM_PROMPT),
                HumanMessage(content=(
                    f"KYC status already determined by policy: {status}\n"
                    f"Applicant identifier: {applicant.applicant_id}\n"
                    "MCP applicant lookup result:\n"
                    + lookup.model_dump_json()
                    + f"\nPrior-context summary: {selected.get('compressed_summary') or 'None'}"
                )),
            ],
        )
        explanation = KYCExplanation.model_validate(explanation)

        validated = KYCResult.model_validate({
            "status": status,
            "checks_performed": checks_performed,
            "risk_flags": risk_flags,
            "rationale": explanation.rationale,
            "confidence": explanation.confidence,
        })
    except Exception as exc:
        return {"reflection_log": [ReflectionNote(
            triggered_by="llm_kyc_failure",
            action_taken="retry",
            detail=f"KYC explanation generation failed: {str(exc)[:240]}",
        )]}

    return {"kyc_result": validated, "next_node": "credit_assessment", "compressed_summary": selected.get("compressed_summary")}
