"""
Builds the LangGraph StateGraph (AC-01, AC-02, AC-03).

Phase 4:
- KYC and credit workers invoke MCP tools through langchain-mcp-adapters.
- Worker reasoning uses Gemini as the primary LLM with an approved Groq
  fallback (see docs/deviations.md for the administrator-approved scope).
- SqliteSaver provides per-thread checkpoint persistence.
- Long-term memory is read at intake and consolidated after offer_draft.
- memory_enabled is supplied per invocation through LangGraph configurable state.

Supervisor dispatch (AC-02): "supervisor" is wired with a real conditional
edge (route_after_supervisor), not a static add_edge. The supervisor node
inspects state to decide the entry hop, and that decision is what the graph
actually follows — including resuming straight into the correct in-flight
stage for a thread that already has partial progress.
"""
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig

import inspect

from src.observability.audit_log import log_event, summarize_state

from src.state.schema import LoanApplicationState
from src.graph.supervisor import supervisor
from src.graph.routing import (
    route_after_supervisor,
    route_after_intake,
    route_after_kyc,
    route_after_credit,
    route_after_reflection,
)
from src.agents.intake_agent import intake_node
from src.agents.kyc_agent import kyc_check_node
from src.agents.credit_agent import credit_assessment_node
from src.agents.offer_agent import offer_draft_node
from src.agents.memory_agent import memory_consolidation_node
from src.agents.reflector import reflector_node


def _observed_node(name, node):
    """Wrap a node with privacy-safe lifecycle logging."""
    accepts_config = "config" in inspect.signature(node).parameters

    def wrapped(state, config: RunnableConfig):
        log_event(
            "node_started",
            user_id=state.get("user_id"),
            thread_id=state.get("thread_id"),
            node=name,
            state=summarize_state(state),
        )
        try:
            result = node(state, config) if accepts_config else node(state)
            log_event(
                "node_completed",
                user_id=state.get("user_id"),
                thread_id=state.get("thread_id"),
                node=name,
                output_fields=sorted(result.keys()) if isinstance(result, dict) else [],
            )
            return result
        except Exception as exc:
            log_event(
                "node_failed",
                user_id=state.get("user_id"),
                thread_id=state.get("thread_id"),
                node=name,
                error_type=type(exc).__name__,
            )
            raise

    return wrapped


def build_graph(checkpointer=None, interrupt_before=None):
    graph = StateGraph(LoanApplicationState)

    graph.add_node("supervisor", _observed_node("supervisor", supervisor))
    graph.add_node("intake", _observed_node("intake", intake_node))
    graph.add_node("kyc_check", _observed_node("kyc_check", kyc_check_node))
    graph.add_node("credit_assessment", _observed_node("credit_assessment", credit_assessment_node))
    graph.add_node("offer_draft", _observed_node("offer_draft", offer_draft_node))
    graph.add_node("reflector", _observed_node("reflector", reflector_node))

    graph.set_entry_point("supervisor")
    graph.add_conditional_edges(
        "supervisor", route_after_supervisor,
        {
            "intake": "intake",
            "kyc_check": "kyc_check",
            "credit_assessment": "credit_assessment",
            "offer_draft": "offer_draft",
            "END": END,
        },
    )

    graph.add_conditional_edges(
        "intake", route_after_intake,
        {"kyc_check": "kyc_check", "reflector": "reflector"},
    )
    graph.add_conditional_edges(
        "kyc_check", route_after_kyc,
        {
            "credit_assessment": "credit_assessment",
            "offer_draft": "offer_draft",
            "reflector": "reflector",
        },
    )
    graph.add_conditional_edges(
        "credit_assessment", route_after_credit,
        {"offer_draft": "offer_draft", "reflector": "reflector"},
    )
    graph.add_node("memory_consolidation", _observed_node("memory_consolidation", memory_consolidation_node))
    graph.add_edge("offer_draft", "memory_consolidation")
    graph.add_edge("memory_consolidation", END)

    graph.add_conditional_edges(
        "reflector", route_after_reflection,
        {
            "intake": "intake",
            "credit_assessment": "credit_assessment",
            "kyc_check": "kyc_check",
            "END": END,
        },
    )

    return graph.compile(checkpointer=checkpointer, interrupt_before=interrupt_before)
