"""Streamlit UI for the Loan Origination Copilot.

Run:
    streamlit run app.py

The UI is a presentation layer over the existing LangGraph implementation.
It does not duplicate lending policy, MCP, RAG, memory, routing, or reflection
logic.
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


# ---------------------------------------------------------------------------
# Graph metadata
# ---------------------------------------------------------------------------

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

NODE_DESCRIPTIONS = {
    "supervisor": "Pure Python entry router. Selects the first workflow hop.",
    "intake": "Deterministic Python extraction into ApplicantProfile from trusted structured fields (no LLM call).",
    "kyc_check": "MCP applicant facts + deterministic KYC gate + LLM explanation.",
    "credit_assessment": "MCP bureau facts + deterministic credit gates + optional agentic RAG.",
    "offer_draft": "Deterministic Python pricing-tier lookup (no LLM call).",
    "memory_consolidation": "Persists durable synthetic memory facts after the decision.",
    "reflector": "Classifies failures and routes bounded retry, replan, or escalation.",
    "END": "Terminal state for the application workflow.",
}

NODE_AUTHORITY = {
    "supervisor": "Deterministic Python",
    "intake": "Deterministic Python + Pydantic validation",
    "kyc_check": "MCP facts + Python gate + LLM explanation",
    "credit_assessment": "MCP facts + Python gate + LLM rationale",
    "offer_draft": "Deterministic Python (pricing-tier lookup)",
    "memory_consolidation": "Python / memory layer",
    "reflector": "Deterministic Python",
    "END": "Terminal",
}

NODE_TOOLS = {
    "supervisor": [],
    "intake": [],
    "kyc_check": ["applicant_lookup"],
    "credit_assessment": ["bureau_check", "lending_policy_search (optional/agentic)"],
    "offer_draft": [],
    "memory_consolidation": ["Chroma semantic memory"],
    "reflector": [],
    "END": [],
}

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
    "current_layer": None,
}

STATUS_META = {
    "pending": ("#343A40", "#9CA3AF", "#14181D"),
    "running": ("#60A5FA", "#BFDBFE", "#172554"),
    "completed": ("#4ADE80", "#BBF7D0", "#052E16"),
    "failed": ("#F87171", "#FECACA", "#450A0A"),
    "not_reached": ("#374151", "#6B7280", "#111827"),
    "recovered": ("#FBBF24", "#FDE68A", "#451A03"),
}


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def _init_session() -> None:
    for key, value in DEFAULTS.items():
        if key not in st.session_state:
            if isinstance(value, (dict, list)):
                st.session_state[key] = value.copy()
            else:
                st.session_state[key] = value

    if not st.session_state.run_id:
        st.session_state.run_id = _new_run_id()

    if not st.session_state.node_status:
        st.session_state.node_status = {
            node: "pending" for node in NODE_LABELS
        }


def _new_run_id() -> str:
    return (
        f"RUN-{datetime.now().strftime('%Y%m%d-%H%M%S')}-"
        f"{uuid.uuid4().hex[:6].upper()}"
    )


def _reset_run() -> None:
    for key, value in DEFAULTS.items():
        if isinstance(value, (dict, list)):
            st.session_state[key] = value.copy()
        else:
            st.session_state[key] = value

    st.session_state.run_id = _new_run_id()
    st.session_state.node_status = {
        node: "pending" for node in NODE_LABELS
    }


# ---------------------------------------------------------------------------
# Serialization / views
# ---------------------------------------------------------------------------

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
        return {
            "type": type(value).__name__,
            "content": _safe_json(value.content),
        }

    return str(value)


def _state_view(state: dict[str, Any] | None) -> dict[str, Any]:
    if not state:
        return {}

    applicant = state.get("applicant")
    kyc = state.get("kyc_result")
    credit = state.get("credit_assessment")
    offer = state.get("offer")

    return {
        "applicant": _safe_json(applicant),
        "kyc": _safe_json(kyc),
        "credit": _safe_json(credit),
        "offer": _safe_json(offer),
        "routing": {
            "next_node": state.get("next_node"),
            "retry_count": state.get("retry_count", 0),
        },
        "memory": {
            "hits": len(state.get("long_term_memory_hits") or []),
            "compressed_summary_present": bool(
                state.get("compressed_summary")
            ),
        },
        "quarantine": {
            "items": len(state.get("quarantined_inputs") or []),
        },
        "reflection": _safe_json(state.get("reflection_log") or []),
    }


def _node_input_view(state: dict[str, Any]) -> dict[str, Any]:
    view = _state_view(state)
    return view


# ---------------------------------------------------------------------------
# Execution events
# ---------------------------------------------------------------------------

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
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "clock": datetime.now().strftime("%H:%M:%S"),
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


def _edge_exists(source: str, target: str) -> bool:
    return (source, target) in EDGES


def _mark_running(node: str, input_state: dict[str, Any]) -> None:
    previous = (
        st.session_state.visited_nodes[-1]
        if st.session_state.visited_nodes
        else None
    )

    if previous and previous != node and _edge_exists(previous, node):
        edge = (previous, node)
        if edge not in st.session_state.executed_edges:
            st.session_state.executed_edges.append(edge)
            _record_event(
                event_type="graph_transition",
                node=node,
                status="traversed",
                detail=(
                    f"{NODE_LABELS[previous]} → {NODE_LABELS[node]}"
                ),
            )

    st.session_state.node_status[node] = "running"
    st.session_state.current_layer = NODE_LABELS[node]

    if node not in st.session_state.visited_nodes:
        st.session_state.visited_nodes.append(node)

    st.session_state.node_inputs[node] = _node_input_view(input_state)

    _record_event(
        event_type="node_started",
        node=node,
        status="running",
        input_data=st.session_state.node_inputs[node],
    )


def _mark_completed(
    node: str,
    update: Any,
    duration_ms: float,
) -> None:
    st.session_state.node_status[node] = "completed"
    st.session_state.node_outputs[node] = _safe_json(update)

    _record_event(
        event_type="node_completed",
        node=node,
        status="completed",
        output_data=st.session_state.node_outputs[node],
        duration_ms=round(duration_ms, 2),
    )


# ---------------------------------------------------------------------------
# Input / graph construction
# ---------------------------------------------------------------------------

def _build_state(
    *,
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

    if not payload["applicant_id"]:
        raise ValueError("Applicant ID is required.")

    if not payload["full_name"]:
        raise ValueError("Full name is required.")

    if not thread_id.strip():
        raise ValueError("Application / thread ID is required.")

    if not user_id.strip():
        raise ValueError("User ID is required.")

    state = new_state(
        thread_id=thread_id.strip(),
        user_id=user_id.strip(),
    )

    state["messages"] = [
        HumanMessage(
            content=json.dumps(
                payload,
                ensure_ascii=False,
            )
        )
    ]

    return state


# ---------------------------------------------------------------------------
# Graph rendering
# ---------------------------------------------------------------------------

def _svg_graph() -> str:
    statuses = st.session_state.node_status
    visited = set(st.session_state.visited_nodes)
    executed_edges = set(
        tuple(edge)
        for edge in st.session_state.executed_edges
    )

    positions = {
        "supervisor": (50, 140),
        "intake": (250, 140),
        "kyc_check": (455, 140),
        "credit_assessment": (675, 140),
        "offer_draft": (905, 140),
        "memory_consolidation": (1140, 140),
        "END": (1390, 140),
        "reflector": (675, 310),
    }

    sizes = {
        node: (145, 70)
        for node in NODE_LABELS
    }
    sizes["credit_assessment"] = (175, 70)
    sizes["memory_consolidation"] = (195, 70)
    sizes["reflector"] = (175, 70)

    svg = [
        '<svg viewBox="0 0 1580 430" '
        'width="100%" '
        'role="img" '
        'aria-label="Loan origination agent workflow">',
        "<defs>",
        '<filter id="glow" x="-20%" y="-20%" width="140%" height="140%">',
        '<feGaussianBlur stdDeviation="3" result="blur"/>',
        '<feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/>'
        "</feMerge>",
        "</filter>",
        '<marker id="arrow" markerWidth="9" markerHeight="9" '
        'refX="8" refY="4.5" orient="auto">',
        '<path d="M0,0 L9,4.5 L0,9 z" fill="#64748B"/>',
        "</marker>",
        '<marker id="arrowActive" markerWidth="9" markerHeight="9" '
        'refX="8" refY="4.5" orient="auto">',
        '<path d="M0,0 L9,4.5 L0,9 z" fill="#60A5FA"/>',
        "</marker>",
        "</defs>",
        '<rect x="0" y="0" width="1580" height="430" rx="24" '
        'fill="#090B10" stroke="#1F2937"/>',
    ]

    # Main lane label
    svg.append(
        '<text x="50" y="35" font-family="Arial" font-size="12" '
        'font-weight="700" fill="#94A3B8">PRIMARY DECISION PATH</text>'
    )

    # Reflection lane label
    svg.append(
        '<text x="610" y="395" font-family="Arial" font-size="11" '
        'font-weight="700" fill="#94A3B8">BOUNDED RECOVERY PATH</text>'
    )

    for source, target in EDGES:
        x1, y1 = positions[source]
        x2, y2 = positions[target]
        w1, h1 = sizes[source]
        w2, h2 = sizes[target]

        # Draw special recovery edges differently.
        if source == "reflector" or target == "reflector":
            sx = x1 + w1 / 2
            sy = y1
            tx = x2 + w2 / 2
            ty = y2 + h2
        else:
            sx = x1 + w1
            sy = y1 + h1 / 2
            tx = x2
            ty = y2 + h2 / 2

        active = (source, target) in executed_edges
        stroke = "#60A5FA" if active else "#334155"
        width = 3.5 if active else 1.5
        marker = "arrowActive" if active else "arrow"
        filter_attr = ' filter="url(#glow)"' if active else ""

        if source == "reflector" or target == "reflector":
            mid_y = (sy + ty) / 2
            path = (
                f"M {sx} {sy} C {sx} {mid_y}, "
                f"{tx} {mid_y}, {tx} {ty}"
            )
        elif abs(sy - ty) < 8:
            path = f"M {sx} {sy} L {tx} {ty}"
        else:
            mid_x = (sx + tx) / 2
            path = (
                f"M {sx} {sy} C {mid_x} {sy}, "
                f"{mid_x} {ty}, {tx} {ty}"
            )

        svg.append(
            f'<path d="{path}" fill="none" stroke="{stroke}" '
            f'stroke-width="{width}" marker-end="url(#{marker})"'
            f"{filter_attr}/>"
        )

    for node in NODE_LABELS:
        x, y = positions[node]
        w, h = sizes[node]

        status = statuses.get(node, "pending")
        if node not in visited and status == "pending":
            status = "pending"

        soft, text, _ = STATUS_META.get(
            status,
            STATUS_META["pending"],
        )

        if status == "running":
            fill = "#172554"
            border = "#3B82F6"
            dot = "#60A5FA"
        elif status == "completed":
            fill = "#052E16"
            border = "#22C55E"
            dot = "#4ADE80"
        elif status == "failed":
            fill = "#450A0A"
            border = "#EF4444"
            dot = "#F87171"
        elif status == "recovered":
            fill = "#451A03"
            border = "#F59E0B"
            dot = "#FBBF24"
        elif status == "not_reached":
            fill = "#111827"
            border = "#374151"
            dot = "#4B5563"
        else:
            fill = "#14181D"
            border = "#374151"
            dot = "#6B7280"

        if node == "credit_assessment":
            fill = "#0F172A"
            border = "#38BDF8" if status == "running" else border

        svg.append(
            f'<g>'
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="18" '
            f'fill="{fill}" stroke="{border}" stroke-width="2"/>'
            f'<circle cx="{x + 22}" cy="{y + 23}" r="7" '
            f'fill="{dot}"/>'
            f'<text x="{x + 38}" y="{y + 27}" '
            f'font-family="Arial" font-size="13" font-weight="700" '
            f'fill="#F8FAFC">{NODE_LABELS[node]}</text>'
            f'<text x="{x + 18}" y="{y + 53}" '
            f'font-family="Arial" font-size="10" font-weight="700" '
            f'letter-spacing="0.7" fill="{text}">{status.upper()}</text>'
            f'</g>'
        )

    svg.append("</svg>")
    return "".join(svg)


def _render_graph() -> None:
    st.markdown(
        _svg_graph(),
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Page 1: Workflow
# ---------------------------------------------------------------------------

def _render_input_form() -> dict[str, Any] | None:
    st.markdown("### Start an application")
    st.caption(
        "Enter the synthetic applicant data. The existing LangGraph "
        "handles the complete workflow."
    )

    with st.container(border=True):
        col1, col2 = st.columns(2)

        with col1:
            applicant_id = st.text_input(
                "Applicant ID",
                value="SYN-0001",
            )
            full_name = st.text_input(
                "Full name",
                value="Asha Kulkarni",
            )
            dob_synthetic = st.text_input(
                "DOB (synthetic)",
                value="1990-01-01",
            )

        with col2:
            declared_income = st.number_input(
                "Declared annual income",
                min_value=0.0,
                value=85000.0,
                step=5000.0,
            )
            declared_employment = st.text_input(
                "Declared employment",
                value="Software Engineer, synthetic employer",
            )
            user_id = st.text_input(
                "User ID",
                value="demo-user",
            )

        raw_free_text_notes = st.text_area(
            "Applicant notes",
            value="I'd like a loan to renovate my kitchen next spring.",
            height=100,
            help=(
                "Applicant-submitted free text is treated as untrusted "
                "content and quarantined by the context layer."
            ),
        )

        col_a, col_b = st.columns([1, 1])

        with col_a:
            start = st.button(
                "Start application",
                type="primary",
                width="stretch",
            )

        with col_b:
            new_run = st.button(
                "Clear / new application",
                width="stretch",
            )

        if new_run:
            _reset_run()
            st.rerun()

    if not start:
        return None

    thread_id = f"streamlit-{uuid.uuid4().hex[:10]}"

    try:
        return _build_state(
            applicant_id=applicant_id,
            full_name=full_name,
            dob_synthetic=dob_synthetic,
            declared_income=declared_income,
            declared_employment=declared_employment,
            raw_free_text_notes=raw_free_text_notes,
            thread_id=thread_id,
            user_id=user_id,
        )
    except Exception as exc:
        st.error(f"Invalid application input: {type(exc).__name__}: {exc}")
        return None


def _render_thinking() -> None:
    current = st.session_state.current_layer

    if not current:
        label = "Ready"
        detail = "Waiting for an application to start."
    else:
        label = f"Thinking · {current}"
        detail = NODE_DESCRIPTIONS.get(
            next(
                (
                    node
                    for node, name in NODE_LABELS.items()
                    if name == current
                ),
                "",
            ),
            "Executing the current workflow layer.",
        )

    st.markdown(
        f"""
        <div style="
            border:1px solid #1F2937;
            background:#0B0F14;
            border-radius:14px;
            padding:14px 16px;
            margin:10px 0 18px 0;
        ">
            <div style="
                color:#E5E7EB;
                font-size:15px;
                font-weight:700;
                margin-bottom:4px;
            ">{label}</div>
            <div style="
                color:#64748B;
                font-size:12px;
            ">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_current_credit_assessment() -> None:
    state = st.session_state.final_state

    if not state:
        return

    credit = state.get("credit_assessment")
    if not credit:
        return

    decision = getattr(credit, "decision", None)
    confidence = getattr(credit, "confidence", None)
    rationale = getattr(credit, "rationale", None)
    thin_file = getattr(credit, "thin_file", None)
    dti = getattr(credit, "dti", None)

    if decision == "approve":
        accent = "#22C55E"
        title = "Approved"
    elif decision == "manual_underwriting":
        accent = "#F59E0B"
        title = "Manual Underwriting"
    elif decision == "decline":
        accent = "#EF4444"
        title = "Declined"
    else:
        accent = "#60A5FA"
        title = "Credit Assessment"

    st.markdown(
        f"""
        <div style="
            border:1px solid #263241;
            border-left:4px solid {accent};
            background:#0B0F14;
            border-radius:16px;
            padding:18px 20px;
            margin-top:18px;
        ">
            <div style="color:#64748B;font-size:11px;
                        text-transform:uppercase;letter-spacing:1.2px;">
                Credit Assessment
            </div>
            <div style="color:#F8FAFC;font-size:25px;
                        font-weight:700;margin-top:5px;">
                {title}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Confidence",
        f"{confidence:.0%}" if isinstance(confidence, (int, float)) else "—",
    )
    c2.metric(
        "Thin-file",
        "Yes" if thin_file else "No",
    )
    c3.metric(
        "DTI",
        f"{dti:.3f}" if isinstance(dti, (int, float)) else "—",
    )

    if rationale:
        with st.expander("Credit rationale", expanded=True):
            st.write(rationale)


def _render_final_outcome() -> None:
    state = st.session_state.final_state

    if not state:
        return

    applicant = state.get("applicant")
    kyc = state.get("kyc_result")
    credit = state.get("credit_assessment")
    offer = state.get("offer")

    if credit:
        decision = getattr(credit, "decision", None)
    else:
        decision = None

    if decision == "approve":
        st.success("Indicative offer generated.")
    elif decision:
        st.warning(
            f"Credit outcome: {str(decision).replace('_', ' ').title()}"
        )
    elif kyc:
        status = getattr(kyc, "status", None)
        if status:
            st.warning(
                f"KYC outcome: {str(status).replace('_', ' ').title()}"
            )

    with st.container(border=True):
        st.markdown("### Final outcome")

        result = {
            "applicant_id": getattr(applicant, "applicant_id", None),
            "kyc_status": getattr(kyc, "status", None),
            "credit_decision": decision,
            "credit_confidence": getattr(credit, "confidence", None),
            "offer_principal": getattr(offer, "principal", None),
            "offer_apr": getattr(offer, "apr", None),
            "offer_term_months": getattr(offer, "term_months", None),
            "retry_count": state.get("retry_count", 0),
        }

        st.json(_safe_json(result))


def _execute_application(state: dict[str, Any]) -> None:
    _reset_run()

    st.session_state.input_payload = json.loads(
        state["messages"][0].content
    )
    st.session_state.run_started = datetime.now(
        timezone.utc
    ).isoformat()

    graph_placeholder = st.empty()
    thinking_placeholder = st.empty()
    status_placeholder = st.empty()
    credit_placeholder = st.empty()
    outcome_placeholder = st.empty()

    _record_event(
        event_type="application_started",
        status="started",
        input_data=st.session_state.input_payload,
        detail="Application submitted to LangGraph.",
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

            with graph_placeholder.container():
                _render_graph()

            for update in graph.stream(
                state,
                config=config,
                stream_mode="updates",
            ):
                for node_name, node_update in update.items():
                    if node_name not in NODE_LABELS:
                        continue

                    started = time.perf_counter()
                    _mark_running(node_name, current_state)

                    if isinstance(node_update, dict):
                        current_state.update(node_update)
                        final_state = dict(current_state)

                    elapsed_ms = (
                        time.perf_counter() - started
                    ) * 1000

                    _mark_completed(
                        node_name,
                        node_update,
                        elapsed_ms,
                    )

                    with graph_placeholder.container():
                        _render_graph()

                    with thinking_placeholder.container():
                        _render_thinking()

                    with status_placeholder.container():
                        reached = len(
                            [
                                node
                                for node in st.session_state.visited_nodes
                                if node != "END"
                            ]
                        )
                        completed = sum(
                            status == "completed"
                            for status in st.session_state.node_status.values()
                        )
                        errors = sum(
                            status == "failed"
                            for status in st.session_state.node_status.values()
                        )
                        st.caption(
                            f"Stages reached: {reached}  ·  "
                            f"Completed: {completed}  ·  "
                            f"Errors: {errors}  ·  "
                            f"Retries: "
                            f"{current_state.get('retry_count', 0)}"
                        )

                    if isinstance(final_state, dict):
                        st.session_state.final_state = final_state

                    with credit_placeholder.container():
                        _render_current_credit_assessment()

                    with outcome_placeholder.container():
                        _render_final_outcome()

                    time.sleep(0.08)

            try:
                snapshot = graph.get_state(config)
                if snapshot and getattr(snapshot, "values", None):
                    final_state = dict(snapshot.values)
            except Exception as exc:
                _record_event(
                    event_type="checkpoint_snapshot_warning",
                    status="warning",
                    error=f"{type(exc).__name__}: {exc}",
                    detail=(
                        "Execution completed, but the final checkpoint "
                        "snapshot could not be read."
                    ),
                )

            for node in NODE_LABELS:
                if (
                    node != "END"
                    and node not in st.session_state.visited_nodes
                ):
                    st.session_state.node_status[node] = "not_reached"

            if "END" not in st.session_state.visited_nodes:
                st.session_state.node_status["END"] = "completed"

            st.session_state.final_state = (
                final_state or current_state or state
            )
            st.session_state.run_finished = datetime.now(
                timezone.utc
            ).isoformat()

            _record_event(
                event_type="application_completed",
                status="completed",
                output_data=_state_view(
                    st.session_state.final_state
                ),
                detail="LangGraph execution completed.",
            )

            with graph_placeholder.container():
                _render_graph()

            with thinking_placeholder.container():
                _render_thinking()

            with credit_placeholder.container():
                _render_current_credit_assessment()

            with outcome_placeholder.container():
                _render_final_outcome()

    except Exception as exc:
        error_text = f"{type(exc).__name__}: {exc}"
        st.session_state.run_error = error_text

        _record_event(
            event_type="application_failed",
            node="application",
            status="failed",
            error=error_text,
            detail=(
                "Unexpected exception escaped the LangGraph "
                "execution boundary."
            ),
        )

        log_event(
            "streamlit_application_failed",
            user_id=state.get("user_id"),
            thread_id=state.get("thread_id"),
            error_type=type(exc).__name__,
        )

        for node in NODE_LABELS:
            if node not in st.session_state.visited_nodes:
                st.session_state.node_status[node] = "not_reached"

        with graph_placeholder.container():
            _render_graph()

        with thinking_placeholder.container():
            _render_thinking()

        st.error(
            "The workflow encountered an unexpected error. "
            "The completed stages and detailed logs remain available "
            "on the Execution Logs page."
        )


def _render_workflow_page() -> None:
    st.markdown(
        """
        <div style="margin-bottom:12px;">
            <div style="
                color:#64748B;
                font-size:11px;
                text-transform:uppercase;
                letter-spacing:1.4px;
            ">Loan Origination Copilot</div>
            <div style="
                color:#F8FAFC;
                font-size:32px;
                font-weight:700;
                margin-top:3px;
            ">Application Workflow</div>
            <div style="
                color:#94A3B8;
                font-size:13px;
                margin-top:4px;
            ">
                Execute the existing LangGraph and watch the active layer
                progress through the workflow.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    state = _render_input_form()

    if state is not None:
        _execute_application(state)

    st.divider()

    # Current execution identity
    c1, c2, c3 = st.columns(3)
    c1.markdown(
        f"**RUN**  `{st.session_state.run_id}`"
    )
    c2.markdown(
        f"**LAYER**  `{st.session_state.current_layer or 'Ready'}`"
    )
    c3.markdown(
        "**MODE**  `Existing LangGraph`"
    )

    st.markdown("### Agent workflow")
    _render_thinking()
    _render_graph()

    st.divider()

    _render_current_credit_assessment()

    if not st.session_state.final_state:
        st.info(
            "Credit assessment will appear here when the workflow reaches "
            "that stage."
        )

    _render_final_outcome()


# ---------------------------------------------------------------------------
# Page 2: Execution logs
# ---------------------------------------------------------------------------

def _render_logs_page() -> None:
    st.markdown("### Execution Logs")
    st.caption(
        "Complete end-to-end execution record for the current application."
    )

    events = st.session_state.execution_events

    if not events:
        st.info(
            "No execution events yet. Start an application from Workflow."
        )
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Events", len(events))
    c2.metric(
        "Nodes",
        len(st.session_state.visited_nodes),
    )
    c3.metric(
        "Transitions",
        len(st.session_state.executed_edges),
    )
    c4.metric(
        "Errors",
        sum(
            event.get("status") == "failed"
            for event in events
        ),
    )

    st.markdown("### Execution order")

    for index, event in enumerate(events, start=1):
        event_name = event.get("event", "event")
        node = event.get("node") or "system"
        status = event.get("status") or "info"
        clock = event.get("clock", "")
        duration = event.get("duration_ms")

        duration_text = (
            f" · {duration:.0f} ms"
            if isinstance(duration, (int, float))
            else ""
        )

        title = (
            f"{index:02d}  {clock}  ·  "
            f"{node}  ·  {event_name}  ·  "
            f"{status.upper()}{duration_text}"
        )

        with st.expander(title, expanded=(index == len(events))):
            if event.get("detail"):
                st.caption(event["detail"])

            if event.get("error"):
                st.error(event["error"])

            if event.get("input") not in (None, {}):
                st.markdown("**Input**")
                st.json(event["input"])

            if event.get("output") not in (None, {}):
                st.markdown("**Output**")
                st.json(event["output"])

    st.markdown("### Graph flow")

    if st.session_state.executed_edges:
        for source, target in st.session_state.executed_edges:
            st.markdown(
                f"`{NODE_LABELS[source]}` → "
                f"`{NODE_LABELS[target]}`"
            )

    st.markdown("### Raw execution record")

    downloadable = json.dumps(
        st.session_state.execution_events,
        indent=2,
        default=str,
    )

    st.download_button(
        "Download execution log",
        data=downloadable,
        file_name=f"{st.session_state.run_id}_execution.json",
        mime="application/json",
        width="content",
    )


# ---------------------------------------------------------------------------
# Page 3: Graph knowledge
# ---------------------------------------------------------------------------

def _render_graph_knowledge_page() -> None:
    st.markdown("### Graph Knowledge")
    st.caption(
        "Reference view of how the loan-origination graph is assembled, "
        "what each layer owns, and which tools it can use."
    )

    st.markdown("#### Node responsibilities")

    for node in NODE_LABELS:
        with st.container(border=True):
            col1, col2, col3 = st.columns([1.3, 1.2, 2.6])

            with col1:
                st.markdown(f"**{NODE_LABELS[node]}**")

            with col2:
                st.caption(NODE_AUTHORITY[node])

            with col3:
                st.write(NODE_DESCRIPTIONS[node])

            tools = NODE_TOOLS[node]
            if tools:
                st.caption(
                    "Tools / dependencies: "
                    + ", ".join(tools)
                )

    st.divider()

    st.markdown("#### Authority model")

    st.code(
        """LLM
  ├─ extraction
  ├─ rationale
  ├─ confidence
  ├─ constrained offer drafting
  └─ discretionary policy retrieval

Trusted MCP
  ├─ applicant facts
  ├─ bureau facts
  └─ lending-policy search

Deterministic Python
  ├─ KYC gate
  ├─ credit policy gate
  ├─ routing
  ├─ retry limits
  └─ hard offer constraints""",
        language="text",
    )

    st.markdown("#### MCP integration")

    mcp_rows = [
        {
            "Component": "applicant_lookup",
            "Type": "MCP tool",
            "Owner": "KYC",
            "Mode": "Deterministic",
        },
        {
            "Component": "bureau_check",
            "Type": "MCP tool",
            "Owner": "Credit",
            "Mode": "Deterministic",
        },
        {
            "Component": "lending_policy_search",
            "Type": "MCP tool",
            "Owner": "Credit / Offer",
            "Mode": "Agentic / optional",
        },
        {
            "Component": "policy://credit_policy_manual",
            "Type": "MCP resource",
            "Owner": "Policy reference",
            "Mode": "Readable resource",
        },
    ]

    st.dataframe(
        mcp_rows,
        width="stretch",
        hide_index=True,
    )

    st.markdown("#### Memory model")

    st.code(
        """thread_id
  → short-term checkpoint state
  → exact pause / resume

user_id
  → long-term Chroma semantic memory
  → durable facts across sessions

Policy:
  memory informs context
  current application facts + current MCP results
  remain authoritative for KYC / credit decisions""",
        language="text",
    )

    st.markdown("#### Context model")

    context_cols = st.columns(4)

    context_items = [
        ("WRITE", "Validated Pydantic objects enter graph state."),
        ("SELECT", "Workers receive task-specific context."),
        ("COMPRESS", "Long histories are summarized into bounded context."),
        ("ISOLATE", "Applicant free text is quarantined and excluded from model instruction context."),
    ]

    for column, (title, body) in zip(context_cols, context_items):
        with column:
            st.markdown(f"**{title}**")
            st.caption(body)

    st.markdown("#### Routing edges")

    edge_rows = [
        {
            "From": NODE_LABELS[source],
            "To": NODE_LABELS[target],
            "Type": (
                "Recovery"
                if source == "reflector"
                or target == "reflector"
                else "Workflow"
            ),
        }
        for source, target in EDGES
    ]

    st.dataframe(
        edge_rows,
        width="stretch",
        hide_index=True,
    )


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Loan Origination Copilot",
        page_icon="◈",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    _init_session()

    st.markdown(
        """
        <style>
        .stApp {
            background: #05070A;
            color: #E5E7EB;
        }

        [data-testid="stHeader"] {
            background: #05070A;
        }

        .block-container {
            max-width: 1500px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        div[data-testid="stMetric"] {
            background: #0B0F14;
            border: 1px solid #1F2937;
            border-radius: 14px;
            padding: 10px 14px;
        }

        div[data-testid="stExpander"] {
            background: #0B0F14;
            border: 1px solid #1F2937;
            border-radius: 12px;
        }

        .stTextInput input,
        .stNumberInput input,
        .stTextArea textarea {
            background: #0B0F14;
            color: #F8FAFC;
        }

        [data-testid="stTabs"] button {
            color: #94A3B8;
        }

        [data-testid="stTabs"] button[aria-selected="true"] {
            color: #F8FAFC;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="
            display:flex;
            justify-content:space-between;
            align-items:center;
            padding-bottom:12px;
            margin-bottom:8px;
            border-bottom:1px solid #111827;
        ">
            <div>
                <span style="
                    color:#F8FAFC;
                    font-size:18px;
                    font-weight:700;
                ">LOAN ORIGINATION COPILOT</span>
                <span style="
                    color:#475569;
                    margin-left:12px;
                    font-size:12px;
                ">Agent Execution Console</span>
            </div>
            <div style="
                color:#64748B;
                font-size:11px;
                letter-spacing:0.8px;
            ">
                {run_id}
            </div>
        </div>
        """.format(run_id=st.session_state.run_id),
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        ["Workflow", "Execution Logs", "Graph Knowledge"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if page == "Workflow":
        _render_workflow_page()
    elif page == "Execution Logs":
        _render_logs_page()
    else:
        _render_graph_knowledge_page()


if __name__ == "__main__":
    main()
