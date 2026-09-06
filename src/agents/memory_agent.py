"""Phase 4 terminal memory-consolidation node."""
from __future__ import annotations

from langchain_core.runnables import RunnableConfig

from src.memory import runtime
from src.state.schema import LoanApplicationState
from src.observability.audit_log import log_event


def memory_consolidation_node(
    state: LoanApplicationState,
    config: RunnableConfig) -> dict:
    """Extract and persist durable facts only after the loan decision is complete."""
    if not runtime.memory_enabled(config):
        log_event("memory_consolidation_skipped", user_id=state.get("user_id"), thread_id=state.get("thread_id"), reason="disabled")
        return {}

    user_id = state["user_id"]
    if not user_id:
        log_event("memory_consolidation_skipped", thread_id=state.get("thread_id"), reason="missing_user_id")
        return {}

    try:
        from src.memory.extraction import extract_memory_facts
        from src.memory.long_term_store import ChromaMemoryStore
        facts = extract_memory_facts(state)
        if not facts:
            log_event("memory_consolidation_completed", user_id=user_id, thread_id=state.get("thread_id"), fact_count=0)
            return {}
        ChromaMemoryStore(runtime.memory_store_path(config)).upsert_facts(facts)
        log_event(
            "memory_consolidation_completed",
            user_id=user_id,
            thread_id=state.get("thread_id"),
            fact_count=len(facts),
            fact_types=sorted({fact.fact_type for fact in facts}),
        )
    except Exception as exc:
        # Memory failure must never alter an already-completed loan decision.
        log_event(
            "memory_consolidation_failed",
            user_id=user_id,
            thread_id=state.get("thread_id"),
            error_type=type(exc).__name__,
        )
        return {}

    return {}
