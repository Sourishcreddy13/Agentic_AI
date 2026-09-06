"""Streamlit execution console for the loan-origination LangGraph.

Run with:
    streamlit run app.py

The UI is deliberately a thin presentation layer over the existing LangGraph.
It does not duplicate lending policy, routing, MCP, RAG, memory, or reflection
logic. Every application run gets a fresh UI execution state and a fresh
application thread id unless the user explicitly supplies one.
"""
from __future__ import annotations

import json
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.messages import HumanMessage

from src.graph.build_graph import build_graph
from src.memory.checkpointer import get_checkpointer
from src.observability.audit_log import log_event
from src.state.schema import new_state


NODE_LABELS = {
    "supervisor": "Supervisor",
    "intake": "Intake",
    "kyc_check": "KYC Check",
    "credit_assessment": "Credit Assessment",
    "offer_draft": "Offer Draft",
    "memory_consolidation": "Memory Consolidation",
    "reflector": "Reflection / Recovery",
    "END": "END",
}

NODE_ORDER = [
    "supervisor",
    "intake",
    "kyc_check",
    "credit_assessment",
    "offer_draft",
    "memory_consolidation",
    "reflector",
    "END",
]

EDGES = [
    ("supervisor", "intake"),
    ("intake", "kyc_check"),
    ("intake", "reflector"),
    ("kyc_check", "credit_assessment"),
    ("kyc_check", "offer_draft"),
    ("kyc_check", "reflector"),
    ("credit_assessment", "offer_draft"),
    ("credit_assessment", "reflector"),
    ("offer_draft", "memory_consolidation"),
    ("memory_consolidation", "END"),
    ("reflector", "intake"),
    ("reflector", "kyc_check"),
    ("reflector", "credit_assessment"),
    ("reflector", "END"),
]

DEFAULTS = {
    "run_id": None,
    "node_status": {},
    "node_errors": {},
    "node_inputs": {},
    "node_outputs": {},
    "execution_events": [],
    "visited_nodes": [],
    "executed_edges": [],
    "final_state": None,
    "input_payload": None,
    "run_started": None,
    "run_finished": None,
    "run_error": None,
}

STATUS_META = {
    "pending": ("#E5E7EB", "#374151", "#F9FAFB"),
    "running": ("#BFDBFE", "#1D4ED8", "#EFF6FF"),
    "completed": ("#BBF7D0", "#166534", "#F0FDF4"),
    "failed": ("#FECACA", "#B91C1C", "#FEF2F2"),
    "skipped": ("#FDE68A", "#92400E", "#FFFBEB"),
    "not_reached": ("#E5E7EB", "#6B7280", "#F9FAFB"),
}


def _init_session() -> None:
    for key, value in DEFAULTS.items():
        st.session_state.setdefault(key, value.copy() if isinstance(value, (dict, list)) else value)


def _new_run_id() -> str:
    return f"RUN-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6].upper()}"


def _reset_run() -> None:
    for key, value in DEFAULTS.items():
        st.session_state[key] = value.copy() if isinstance(value, (dict, list)) else value
    st.session_state.run_id = _new_run_id()
    st.session_state.node_status = {node: "pending" for node in NODE_ORDER}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _short_time() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _safe_json(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_safe_json(v) for v in value]
    if hasattr(value, "model_dump"):
        return _safe_json(value.model_dump(mode="json"))
    if hasattr(value, "dict"):
        return _safe_json(value.dict())
    if hasattr(value, "content") and not isinstance(value, str):
        return {"type": type(value).__name__, "content": _safe_json(value.content)}
    return str(value)


def _state_view(state: dict[str, Any] | None) -> dict[str, Any]:
    """Return a useful UI representation of the current graph state."""
    if not state:
        return {}

    applicant = state.get("applicant")
    kyc = state.get("kyc_result")
    credit = state.get("credit_assessment")
    offer = state.get("offer")
    reflection_log = state.get("reflection_log") or []
    memory_hits = state.get("long_term_memory_hits") or []
    quarantined = state.get("quarantined_inputs") or []

    return {
        "identity": {
            "user_id": state.get("user_id"),
            "thread_id": state.get("thread_id"),
        },
        "applicant": _safe_json(applicant),
        "kyc": _safe_json(kyc),
        "credit": _safe_json(credit),
        "offer": _safe_json(offer),
        "routing": {
            "next_node": state.get("next_node"),
            "retry_count": state.get("retry_count", 0),
        },
        "context": {
            "compressed_summary_present": bool(state.get("compressed_summary")),
            "memory_hits": len(memory_hits),
            "quarantined_inputs": len(quarantined),
        },
        "reflection": _safe_json(reflection_log),
    }


def _node_input_view(state: dict[str, Any]) -> dict[str, Any]:
    view = _state_view(state)
    # Keep the raw message contents out of the stage-level log. The explicit
    # application input is displayed separately and the project audit logger
    # remains responsible for sensitive-data redaction.
    view.pop("messages", None)
    return view


def _node_output_view(update: Any) -> Any:
    if not isinstance(update, dict):
        return _safe_json(update)
    return _safe_json(update)


def _build_state(
    applicant_id: str,
    full_name: str,
    dob_synthetic: str,
    declared_income: float,
    declared_employment: str,
    raw_free_text_notes: str,
    thread_id: str,
    user_id: str,
) -> dict[str, Any]:
    payload = {
        "applicant_id": applicant_id.strip(),
        "full_name": full_name.strip(),
        "dob_synthetic": dob_synthetic.strip(),
        "declared_income": float(declared_income),
        "declared_employment": declared_employment.strip(),
        "raw_free_text_notes": raw_free_text_notes,
    }

    state = new_state(thread_id=thread_id.strip(), user_id=user_id.strip())
    state["messages"] = [
        HumanMessage(content=json.dumps(payload, ensure_ascii=False))
    ]
    return state


def _edge_exists(source: str, target: str) -> bool:
    return (source, target) in EDGES


def _record_event(
    *,
    event_type: str,
    node: str | None = None,
    status: str | None = None,
    input_data: Any = None,
    output_data: Any = None,
    error: str | None = None,
    duration_ms: float | None = None,
    detail: str | None = None,
) -> None:
    st.session_state.execution_events.append(
        {
            "timestamp": _now(),
            "clock": _short_time(),
            "run_id": st.session_state.run_id,
            "event": event_type,
            "node": node,
            "status": status,
            "input": _safe_json(input_data),
            "output": _safe_json(output_data),
            "error": error,
            "duration_ms": duration_ms,
            "detail": detail,
        }
    )


def _mark_running(node: str, input_state: dict[str, Any]) -> None:
    previous = st.session_state.visited_nodes[-1] if st.session_state.visited_nodes else None
    if previous and previous != node and _edge_exists(previous, node):
        edge = (previous, node)
        if edge not in st.session_state.executed_edges:
            st.session_state.executed_edges.append(edge)
            _record_event(
                event_type="graph_transition",
                node=node,
                status="traversed",
                detail=f"{NODE_LABELS[previous]} → {NODE_LABELS[node]}",
            )

    st.session_state.node_status[node] = "running"
    if node not in st.session_state.visited_nodes:
        st.session_state.visited_nodes.append(node)
    st.session_state.node_inputs[node] = _node_input_view(input_state)
    _record_event(
        event_type="node_started",
        node=node,
        status="running",
        input_data=st.session_state.node_inputs[node],
    )


def _mark_completed(node: str, update: Any, duration_ms: float) -> None:
    st.session_state.node_status[node] = "completed"
    st.session_state.node_outputs[node] = _node_output_view(update)
    _record_event(
        event_type="node_completed",
        node=node,
        status="completed",
        output_data=st.session_state.node_outputs[node],
        duration_ms=round(duration_ms, 2),
    )


def _mark_failed(node: str, exc: Exception) -> None:
    error_text = f"{type(exc).__name__}: {exc}"
    st.session_state.node_status[node] = "failed"
    st.session_state.node_errors[node] = error_text
    _record_event(
        event_type="node_failed",
        node=node,
        status="failed",
        error=error_text,
    )


def _svg_graph() -> str:
    statuses = st.session_state.node_status
    executed_edges = set(tuple(edge) for edge in st.session_state.executed_edges)
    visited = set(st.session_state.visited_nodes)

    positions = {
        "supervisor": (70, 180),
        "intake": (280, 180),
        "kyc_check": (490, 80),
        "credit_assessment": (700, 80),
        "offer_draft": (910, 80),
        "memory_consolidation": (1120, 80),
        "reflector": (700, 280),
        "END": (1330, 80),
    }
    sizes = {node: (155, 62) for node in NODE_ORDER}
    sizes["memory_consolidation"] = (190, 62)

    svg = [
        '<svg viewBox="0 0 1510 390" width="100%" role="img" aria-label="Loan origination agent graph">',
        '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="#94A3B8"/></marker>',
        '<marker id="arrowActive" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="#111827"/></marker></defs>',
        '<rect x="0" y="0" width="1510" height="390" rx="22" fill="#F8FAFC"/>',
    ]

    for source, target in EDGES:
        x1, y1 = positions[source]
        x2, y2 = positions[target]
        w1, h1 = sizes[source]
        w2, h2 = sizes[target]
        sx = x1 + w1
        sy = y1 + h1 / 2
        tx = x2
        ty = y2 + h2 / 2
        active = (source, target) in executed_edges
        stroke = "#111827" if active else "#CBD5E1"
        width = 3 if active else 1.5
        marker = "arrowActive" if active else "arrow"

        if abs(sy - ty) < 12:
            path = f'M {sx} {sy} L {tx} {ty}'
        else:
            midx = (sx + tx) / 2
            path = f'M {sx} {sy} C {midx} {sy}, {midx} {ty}, {tx} {ty}'

        svg.append(
            f'<path d="{path}" fill="none" stroke="{stroke}" stroke-width="{width}" marker-end="url(#{marker})"/>'
        )

    for node in NODE_ORDER:
        x, y = positions[node]
        w, h = sizes[node]
        status = statuses.get(node, "pending")
        fill, text, soft = STATUS_META.get(status, STATUS_META["pending"])
        if node in visited and status == "pending":
            status = "not_reached"
            fill, text, soft = STATUS_META[status]
        label = NODE_LABELS[node]
        svg.append(
            f'<g><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="16" fill="{soft}" stroke="{text}" stroke-width="2.5"/>'
            f'<circle cx="{x + 22}" cy="{y + 22}" r="7" fill="{fill}" stroke="{text}" stroke-width="1.5"/>'
            f'<text x="{x + 38}" y="{y + 27}" font-family="Arial" font-size="13" font-weight="700" fill="#0F172A">{label}</text>'
            f'<text x="{x + 38}" y="{y + 47}" font-family="Arial" font-size="10" font-weight="600" fill="{text}">{status.upper()}</text></g>'
        )

    svg.append('</svg>')
    return "".join(svg)


def _render_graph(placeholder=None) -> None:
    target = placeholder if placeholder is not None else st
    target.markdown(_svg_graph(), unsafe_allow_html=True)


def _render_status() -> None:
    statuses = st.session_state.node_status
    completed = sum(status == "completed" for status in statuses.values())
    failed = sum(status == "failed" for status in statuses.values())
    running = sum(status == "running" for status in statuses.values())
    reached = len([n for n in st.session_state.visited_nodes if n != "END"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stages reached", reached)
    c2.metric("Completed", completed)
    c3.metric("Errors", failed)
    c4.metric("Retries", st.session_state.get("final_state", {}).get("retry_count", 0) if isinstance(st.session_state.get("final_state"), dict) else 0)


def _render_application_input() -> None:
    st.subheader("Application input")
    payload = st.session_state.get("input_payload")
    if payload:
        display = dict(payload)
        if display.get("raw_free_text_notes"):
            display["raw_free_text_notes"] = {
                "value": display["raw_free_text_notes"],
                "trust": "UNTRUSTED / QUARANTINED",
            }
        st.json(display)
    else:
        st.info("Submit an application to begin a new run.")


def _render_timeline() -> None:
    events = st.session_state.execution_events
    if not events:
        st.info("No execution events yet.")
        return

    st.subheader("End-to-end execution log")
    for index, event in enumerate(reversed(events), start=1):
        label = event.get("node") or event.get("event") or "event"
        status = event.get("status") or "info"
        duration = event.get("duration_ms")
        suffix = f" · {duration:.0f} ms" if isinstance(duration, (int, float)) else ""
        title = f"{event.get('clock', '')} · {label} · {status.upper()}{suffix}"

        with st.expander(title, expanded=(index <= 3)):
            if event.get("detail"):
                st.write(event["detail"])
            if event.get("error"):
                st.error(event["error"])
            if event.get("input") not in (None, {}):
                st.markdown("**Input / state before stage**")
                st.json(event["input"])
            if event.get("output") not in (None, {}):
                st.markdown("**Output / state contribution**")
                st.json(event["output"])


def _render_stage_details() -> None:
    st.subheader("Stage details")
    visited = st.session_state.visited_nodes
    if not visited:
        st.info("Stage-level input/output will appear here as the graph executes.")
        return

    for node in visited:
        status = st.session_state.node_status.get(node, "pending")
        with st.expander(f"{NODE_LABELS[node]} · {status.upper()}", expanded=False):
            if node in st.session_state.node_inputs:
                st.markdown("**Input**")
                st.json(st.session_state.node_inputs[node])
            if node in st.session_state.node_outputs:
                st.markdown("**Output**")
                st.json(st.session_state.node_outputs[node])
            if node in st.session_state.node_errors:
                st.markdown("**Error**")
                st.error(st.session_state.node_errors[node])


def _render_outcome() -> None:
    state = st.session_state.get("final_state")
    if not state:
        return

    applicant = state.get("applicant")
    kyc = state.get("kyc_result")
    credit = state.get("credit_assessment")
    offer = state.get("offer")

    st.subheader("Final outcome")

    decision = getattr(credit, "decision", None)
    kyc_status = getattr(kyc, "status", None)

    if decision == "approve":
        st.success("Indicative offer generated")
    elif decision in {"decline", "manual_underwriting"}:
        st.warning(f"Credit outcome: {decision.replace('_', ' ').title()}")
    elif kyc_status:
        st.warning(f"KYC outcome: {kyc_status.replace('_', ' ').title()}")

    summary = {
        "applicant_id": getattr(applicant, "applicant_id", None),
        "kyc_status": kyc_status,
        "credit_decision": decision,
        "credit_confidence": getattr(credit, "confidence", None),
        "thin_file": getattr(credit, "thin_file", None),
        "offer_principal": getattr(offer, "principal", None),
        "offer_apr": getattr(offer, "apr", None),
        "offer_term_months": getattr(offer, "term_months", None),
        "next_node": state.get("next_node"),
        "retry_count": state.get("retry_count", 0),
    }
    st.json(_safe_json(summary))

    reflection_log = state.get("reflection_log") or []
    if reflection_log:
        st.markdown("**Recovery / reflection**")
        for note in reflection_log:
            st.info(
                f"{note.triggered_by} → {note.action_taken}: {note.detail}"
            )


def _execute_application(state: dict[str, Any]) -> None:
    _reset_run()
    st.session_state.input_payload = json.loads(state["messages"][0].content)
    st.session_state.run_started = _now()

    graph_placeholder = st.empty()
    status_placeholder = st.empty()
    timeline_placeholder = st.empty()
    stage_placeholder = st.empty()
    outcome_placeholder = st.empty()

    _record_event(
        event_type="application_started",
        status="started",
        input_data=st.session_state.input_payload,
        detail="Loan application submitted to the LangGraph execution engine.",
    )

    try:
        with get_checkpointer() as checkpointer:
            graph = build_graph(checkpointer=checkpointer)
            config = {
                "configurable": {
                    "thread_id": state["thread_id"],
                    "memory_enabled": True,
                }
            }

            current_state = dict(state)
            final_state = None
            node_started_at: dict[str, float] = {}

            _render_graph(graph_placeholder)

            for update in graph.stream(
                state,
                config=config,
                stream_mode="updates",
            ):
                for node_name, node_update in update.items():
                    if node_name not in NODE_LABELS:
                        continue

                    node_started_at[node_name] = time.perf_counter()
                    _mark_running(node_name, current_state)

                    # Merge the node contribution into our local observable state.
                    if isinstance(node_update, dict):
                        current_state.update(node_update)
                        final_state = dict(current_state)

                    elapsed_ms = (
                        time.perf_counter() - node_started_at[node_name]
                    ) * 1000
                    _mark_completed(node_name, node_update, elapsed_ms)

                    # Explicitly record END as a completed terminal stage when yielded.
                    if node_name == "END":
                        st.session_state.node_status["END"] = "completed"

                    _render_graph(graph_placeholder)
                    with status_placeholder.container():
                        _render_status()
                    with timeline_placeholder.container():
                        _render_timeline()
                    with stage_placeholder.container():
                        _render_stage_details()

                    time.sleep(0.10)

            try:
                snapshot = graph.get_state(config)
                if snapshot and getattr(snapshot, "values", None):
                    final_state = dict(snapshot.values)
            except Exception as exc:
                _record_event(
                    event_type="checkpoint_snapshot_warning",
                    status="warning",
                    error=f"{type(exc).__name__}: {exc}",
                    detail="Graph execution completed, but the final checkpoint snapshot could not be read.",
                )
                log_event(
                    "streamlit_state_snapshot_failed",
                    thread_id=state.get("thread_id"),
                    error_type=type(exc).__name__,
                )

            # Mark declared graph nodes that were never reached.
            for node in NODE_ORDER:
                if node == "END":
                    continue
                if node not in st.session_state.visited_nodes:
                    st.session_state.node_status[node] = "not_reached"

            if "END" not in st.session_state.visited_nodes:
                st.session_state.node_status["END"] = "completed"
                if st.session_state.visited_nodes:
                    last_node = st.session_state.visited_nodes[-1]
                    if last_node != "END" and _edge_exists(last_node, "END"):
                        edge = (last_node, "END")
                        if edge not in st.session_state.executed_edges:
                            st.session_state.executed_edges.append(edge)

            st.session_state.final_state = final_state or current_state or state
            st.session_state.run_finished = _now()
            _record_event(
                event_type="application_completed",
                status="completed",
                output_data=_state_view(st.session_state.final_state),
                detail="LangGraph execution completed and final checkpoint state captured.",
            )

            _render_graph(graph_placeholder)
            with status_placeholder.container():
                _render_status()
            with timeline_placeholder.container():
                _render_timeline()
            with stage_placeholder.container():
                _render_stage_details()
            with outcome_placeholder.container():
                _render_outcome()

    except Exception as exc:
        error_text = f"{type(exc).__name__}: {exc}"
        st.session_state.run_error = error_text

        # If the graph failed before a node name could be identified, preserve
        # everything already completed and log the application-level failure.
        _record_event(
            event_type="application_failed",
            node="application",
            status="failed",
            error=error_text,
            detail="Unexpected exception escaped the LangGraph execution boundary.",
        )
        log_event(
            "streamlit_application_failed",
            user_id=state.get("user_id"),
            thread_id=state.get("thread_id"),
            error_type=type(exc).__name__,
        )
        traceback.print_exc()

        for node in NODE_ORDER:
            if node not in st.session_state.visited_nodes:
                st.session_state.node_status[node] = "not_reached"

        _render_graph(graph_placeholder)
        with status_placeholder.container():
            _render_status()
        with timeline_placeholder.container():
            _render_timeline()
        with stage_placeholder.container():
            _render_stage_details()

        st.error(
            "The graph execution encountered an unexpected error. "
            "Completed stages and the full execution log remain available below."
        )
        st.code(error_text, language="text")


def _render_sidebar() -> tuple[dict[str, Any] | None, bool, bool]:
    with st.sidebar:
        st.header("Synthetic application")

        applicant_id = st.text_input("Applicant ID", value="SYN-0001")
        full_name = st.text_input("Full name", value="Asha Kulkarni")
        dob_synthetic = st.text_input("DOB (synthetic)", value="1990-01-01")
        declared_income = st.number_input(
            "Declared annual income",
            min_value=0.0,
            value=85000.0,
            step=5000.0,
        )
        declared_employment = st.text_input(
            "Declared employment",
            value="Software Engineer, synthetic employer Acme Corp",
        )
        raw_free_text_notes = st.text_area(
            "Applicant notes (untrusted)",
            value="I'd like a loan to renovate my kitchen next spring.",
            height=120,
            help="This field is treated as untrusted applicant content and is quarantined by the application.",
        )

        user_id = st.text_input("User ID", value="demo-user")
        thread_id = st.text_input(
            "Application / thread ID",
            value=f"streamlit-{uuid.uuid4().hex[:8]}",
        )

        run_application = st.button(
            "Run complete application",
            type="primary",
            use_container_width=True,
        )
        new_application = st.button(
            "Start new application",
            use_container_width=True,
        )

        st.divider()
        st.markdown("**Execution model**")
        st.caption(
            "Each run resets the visual graph and execution console. "
            "The underlying LangGraph, MCP, RAG, memory, checkpointing, and reflection behavior is unchanged."
        )

        if new_application:
            _reset_run()
            st.rerun()

        if run_application:
            try:
                state = _build_state(
                    applicant_id=applicant_id,
                    full_name=full_name,
                    dob_synthetic=dob_synthetic,
                    declared_income=declared_income,
                    declared_employment=declared_employment,
                    raw_free_text_notes=raw_free_text_notes,
                    thread_id=thread_id,
                    user_id=user_id,
                )
                return state, True, False
            except Exception as exc:
                st.error(
                    f"Invalid application input: {type(exc).__name__}: {exc}"
                )
                return None, False, False

    return None, False, False


def main() -> None:
    st.set_page_config(
        page_title="Loan Origination Copilot",
        page_icon="🏦",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    _init_session()

    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
        .run-chip {display:inline-block; padding:4px 10px; border-radius:999px;
                   background:#F1F5F9; color:#334155; font-size:12px; font-weight:700;}
        .section-card {padding:1rem 1.1rem; border:1px solid #E2E8F0;
                       border-radius:14px; background:#FFFFFF;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Loan Origination Copilot")
    st.caption(
        "LangGraph multi-agent workflow with deterministic policy gates, MCP, agentic RAG, memory, checkpointing, and self-healing."
    )

    if not st.session_state.run_id:
        st.session_state.run_id = _new_run_id()

    st.markdown(
        f'<span class="run-chip">Run ID: {st.session_state.run_id}</span>',
        unsafe_allow_html=True,
    )

    state, run_requested, _ = _render_sidebar()
    if run_requested and state is not None:
        _execute_application(state)

    # Main screen intentionally re-renders current session state only.
    # A new application starts from a clean graph and event history.
    left, right = st.columns([2.15, 1])

    with left:
        st.subheader("Live workflow graph")
        _render_graph()
        with st.container(border=True):
            _render_timeline()

    with right:
        with st.container(border=True):
            _render_application_input()
        with st.container(border=True):
            _render_outcome()

    st.divider()
    st.subheader("Execution state")
    _render_stage_details()

    if st.session_state.run_error:
        st.subheader("Error summary")
        st.error(st.session_state.run_error)


if __name__ == "__main__":
    main()
