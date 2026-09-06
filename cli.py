"""Single-command CLI runner for the loan-origination LangGraph application.

Usage
-----
    python cli.py
    python cli.py --input sample_inputs/applicant_strong.json

The CLI drives the existing LangGraph workflow rather than duplicating any
agent, policy, MCP, RAG, memory, or reflection logic. It streams stage
updates, prints the end-to-end workflow, captures errors, and writes a
privacy-safe JSONL execution log.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage

from src.graph.build_graph import build_graph
from src.memory.checkpointer import get_checkpointer
from src.observability.audit_log import log_event
from src.state.schema import LoanApplicationState, new_state


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = PROJECT_ROOT / "sample_inputs" / "applicant_strong.json"
CLI_LOG_DIR = PROJECT_ROOT / "data" / "logs"
CLI_LOG_PATH = CLI_LOG_DIR / "cli_execution.jsonl"

_SENSITIVE_KEY_RE = re.compile(
    r"(^|_)(full_name|dob|dob_synthetic|birth|declared_income|salary|employment|"
    r"employer|address|email|phone|mobile|ssn|pan|aadhaar|account|routing|"
    r"card|notes|raw|message|prompt|content|text|applicant_id|user_id)(_|$)",
    re.IGNORECASE,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fingerprint(value: Any) -> str:
    digest = hashlib.sha256(str(value).encode("utf-8")).hexdigest()
    return f"sha256:{digest[:16]}"


def _sanitize(value: Any, key: str | None = None) -> Any:
    """Sanitize persisted CLI event data while retaining useful structure."""
    if key and _SENSITIVE_KEY_RE.search(key):
        return {
            "redacted": True,
            "fingerprint": _fingerprint(value),
        }

    if hasattr(value, "model_dump"):
        return _sanitize(value.model_dump(mode="json"), key)

    if isinstance(value, dict):
        return {
            str(k): _sanitize(v, str(k))
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [_sanitize(v) for v in value]

    if hasattr(value, "content"):
        return {
            "type": getattr(value, "type", value.__class__.__name__),
            "content_fingerprint": _fingerprint(value.content),
        }

    if isinstance(value, str):
        return value

    if isinstance(value, (int, float, bool)) or value is None:
        return value

    return str(value)


def _display(value: Any) -> Any:
    """Convert runtime values to terminal-friendly structures."""
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")

    if isinstance(value, dict):
        return {str(k): _display(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_display(v) for v in value]

    if hasattr(value, "content"):
        return {
            "type": getattr(value, "type", value.__class__.__name__),
            "content": str(value.content),
        }

    return value


def _write_cli_event(record: dict[str, Any]) -> None:
    """Persist a sanitized CLI event to JSONL."""
    CLI_LOG_DIR.mkdir(parents=True, exist_ok=True)

    safe_record = _sanitize(record)

    with CLI_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                safe_record,
                ensure_ascii=False,
                default=str,
                sort_keys=True,
            )
            + "\n"
        )


def _record_event(
    *,
    run_id: str,
    event: str,
    node: str | None = None,
    status: str | None = None,
    input_data: Any = None,
    output_data: Any = None,
    error: Exception | str | None = None,
    graph_path: list[str] | None = None,
    duration_ms: int | None = None,
) -> None:
    """Write the same execution event to the CLI and project audit logs."""
    record: dict[str, Any] = {
        "timestamp": _utc_now(),
        "run_id": run_id,
        "event": event,
    }

    if node is not None:
        record["node"] = node
    if status is not None:
        record["status"] = status
    if input_data is not None:
        record["input"] = input_data
    if output_data is not None:
        record["output"] = output_data
    if error is not None:
        record["error"] = {
            "type": type(error).__name__
            if isinstance(error, Exception)
            else "Error",
            "message": str(error)[:1000],
        }
    if graph_path is not None:
        record["graph_path"] = list(graph_path)
    if duration_ms is not None:
        record["duration_ms"] = duration_ms

    # Existing audit logger performs the project's canonical PII redaction.
    log_event(
        f"cli_{event}",
        thread_id=run_id,
        node=node,
        status=status,
        input=input_data,
        output=output_data,
        error=record.get("error"),
        graph_path=graph_path,
        duration_ms=duration_ms,
    )

    _write_cli_event(record)


def _load_application(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Application input file does not exist: {path}"
        )

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON application input: {path}: {exc}"
        ) from exc

    if not isinstance(payload, dict):
        raise ValueError("Application input must be a JSON object.")

    return payload


def _build_initial_state(
    payload: dict[str, Any],
    *,
    thread_id: str,
    user_id: str,
) -> LoanApplicationState:
    state = new_state(
        thread_id=thread_id,
        user_id=user_id,
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


def _print_stage(
    *,
    node: str,
    output: Any,
    elapsed_ms: int,
) -> None:
    print(f"✓ {node:<24} completed    {elapsed_ms:>6} ms")

    if output:
        rendered = json.dumps(
            _display(output),
            indent=2,
            ensure_ascii=False,
            default=str,
        )
        print("  OUTPUT")
        for line in rendered.splitlines():
            print(f"    {line}")


def run_application(
    input_path: Path,
    *,
    user_id: str,
    thread_id: str,
    memory_enabled: bool,
) -> int:
    payload = _load_application(input_path)
    run_id = (
        f"RUN-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-"
        f"{uuid.uuid4().hex[:8]}"
    )

    state = _build_initial_state(
        payload,
        thread_id=thread_id,
        user_id=user_id,
    )

    print("=" * 80)
    print("LOAN ORIGINATION COPILOT")
    print("=" * 80)
    print(f"Run ID    : {run_id}")
    print(f"Input     : {input_path}")
    print(f"User ID   : {user_id}")
    print(f"Thread ID : {thread_id}")
    print(f"Started   : {_utc_now()}")
    print("-" * 80)

    print("APPLICATION INPUT")
    print(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        )
    )
    print("-" * 80)
    print("GRAPH EXECUTION")

    _record_event(
        run_id=run_id,
        event="application_started",
        status="running",
        input_data=payload,
        graph_path=[],
    )

    config = {
        "configurable": {
            "thread_id": thread_id,
            "memory_enabled": memory_enabled,
        }
    }

    graph_path: list[str] = []
    last_state: dict[str, Any] = dict(state)
    previous_yield = time.perf_counter()
    exit_code = 0

    try:
        with get_checkpointer() as checkpointer:
            graph = build_graph(checkpointer=checkpointer)

            for update in graph.stream(
                state,
                config=config,
                stream_mode="updates",
            ):
                elapsed_ms = max(
                    0,
                    round((time.perf_counter() - previous_yield) * 1000),
                )
                previous_yield = time.perf_counter()

                if not isinstance(update, dict):
                    continue

                for node, node_output in update.items():
                    graph_path.append(str(node))

                    input_snapshot = {
                        "prior_graph_path": graph_path[:-1],
                        "available_state_fields": sorted(last_state.keys()),
                    }
                    output_snapshot = _display(node_output)

                    if isinstance(node_output, dict):
                        last_state.update(node_output)

                    _print_stage(
                        node=str(node),
                        output=output_snapshot,
                        elapsed_ms=elapsed_ms,
                    )

                    _record_event(
                        run_id=run_id,
                        event="node_completed",
                        node=str(node),
                        status="completed",
                        input_data=input_snapshot,
                        output_data=output_snapshot,
                        graph_path=graph_path,
                        duration_ms=elapsed_ms,
                    )

            snapshot = graph.get_state(config)
            if snapshot is not None and getattr(snapshot, "values", None):
                last_state = dict(snapshot.values)

    except Exception as exc:
        # Expected failures normally flow through the graph's reflector. This
        # boundary records any genuinely unhandled exception without hiding it.
        exit_code = 1

        print(
            f"✗ GRAPH EXECUTION FAILED [{type(exc).__name__}]: {exc}"
        )

        _record_event(
            run_id=run_id,
            event="application_failed",
            status="failed",
            error=exc,
            graph_path=graph_path,
        )

    final_payload = {
        "application_id": getattr(
            last_state.get("applicant"),
            "applicant_id",
            None,
        ),
        "kyc_status": getattr(
            last_state.get("kyc_result"),
            "status",
            None,
        ),
        "credit_decision": getattr(
            last_state.get("credit_assessment"),
            "decision",
            None,
        ),
        "offer": _display(last_state.get("offer")),
        "reflection_log": _display(
            last_state.get("reflection_log", [])
        ),
        "graph_path": graph_path,
    }

    print("-" * 80)
    print("GRAPH FLOW")
    print("  " + " -> ".join(graph_path or ["no nodes executed"]))

    print("-" * 80)
    print("FINAL RESULT")
    print(
        json.dumps(
            final_payload,
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    )

    _record_event(
        run_id=run_id,
        event=(
            "application_completed"
            if exit_code == 0
            else "application_terminated"
        ),
        status=("completed" if exit_code == 0 else "failed"),
        output_data=final_payload,
        graph_path=graph_path,
    )

    print("-" * 80)
    print(f"Execution log: {CLI_LOG_PATH}")
    print(f"Completed    : {_utc_now()}")
    print(f"Exit status  : {exit_code}")
    print("=" * 80)

    return exit_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the synthetic loan-origination LangGraph application."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=(
            "Synthetic application JSON input. "
            f"Default: {DEFAULT_INPUT.relative_to(PROJECT_ROOT)}"
        ),
    )
    parser.add_argument(
        "--user-id",
        default="cli-user",
        help="Long-term memory identifier.",
    )
    parser.add_argument(
        "--thread-id",
        default=None,
        help="Checkpoint/thread identifier. Generated when omitted.",
    )
    parser.add_argument(
        "--disable-memory",
        action="store_true",
        help="Disable persistent long-term memory for this invocation.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    thread_id = args.thread_id or f"thread-{uuid.uuid4().hex[:12]}"

    try:
        return run_application(
            args.input.resolve(),
            user_id=args.user_id,
            thread_id=thread_id,
            memory_enabled=not args.disable_memory,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"INPUT ERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nExecution interrupted by user.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
