"""Worker-level context preparation pipeline for Phase 5."""
from __future__ import annotations

from typing import Any

from src.context.compress import CompressionResult, summarize_if_long
from src.context.select import select_context
from src.observability.audit_log import log_event
from src.state.schema import LoanApplicationState


def prepare_worker_context(
    state: LoanApplicationState,
    purpose: str,
    *,
    threshold: int = 20,
    keep_recent: int = 6,
) -> tuple[dict[str, Any], CompressionResult]:
    """Run compress-then-select once per long thread, then reuse the summary."""
    if state.get("compressed_summary"):
        compression = CompressionResult(
            summary=state["compressed_summary"],
            recent_messages=list(state.get("messages") or [])[-keep_recent:],
            compressed=True,
        )
    else:
        try:
            compression = summarize_if_long(
                state.get("messages") or [],
                threshold=threshold,
                keep_recent=keep_recent,
            )
        except Exception as exc:
            # Compression is context optimization, not policy authority.
            # Fail open to the uncompressed working context and record only
            # the exception class to avoid leaking prompt/message content.
            log_event(
                "context_compression_failed",
                user_id=state.get("user_id"),
                thread_id=state.get("thread_id"),
                purpose=purpose,
                error_type=type(exc).__name__,
            )
            compression = CompressionResult(
                summary=None,
                recent_messages=list(state.get("messages") or []),
                compressed=False,
            )

    effective_summary = compression.summary if compression.compressed else state.get("compressed_summary")
    selected_state = dict(state)
    selected_state["compressed_summary"] = effective_summary
    selected = select_context(selected_state, purpose)

    log_event(
        "context_prepared",
        user_id=state.get("user_id"),
        thread_id=state.get("thread_id"),
        purpose=purpose,
        compressed=compression.compressed,
        message_count=len(state.get("messages") or []),
        selected_fields=sorted(selected.keys()),
    )
    return selected, compression
