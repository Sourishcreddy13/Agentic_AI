"""Agentic lending-policy retrieval loop backed by MCP tools.

The model may decide that no lookup is needed. When a lookup is needed,
Gemini is the primary tool-calling model and Groq is the provider fallback.
RAG failures are explicit outcomes and never masquerade as "not needed".
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from src.llm.gateway import get_configured_provider_names, get_provider_model
from src.mcp_client import _normalize_mcp_result, get_mcp_tools
from src.observability.audit_log import fingerprint, log_event


@dataclass(frozen=True)
class PolicyLookupResult:
    called: bool
    queries: tuple[str, ...]
    results: tuple[dict[str, Any], ...]
    steps: int
    status: Literal["not_needed", "retrieved", "failed"] = "not_needed"
    provider: str | None = None
    error_type: str | None = None


def _run_async(coro):
    return asyncio.run(coro)


async def _load_policy_tool():
    tools = await get_mcp_tools()
    return next((tool for tool in tools if tool.name == "lending_policy_search"), None)


def _invoke_with_provider(
    model_factory,
    provider_name: str,
    policy_tool,
    task: str,
    max_steps: int,
) -> PolicyLookupResult:
    model = model_factory().bind_tools([policy_tool])
    messages = [
        SystemMessage(
            content=(
                "You are a policy-retrieval assistant inside a loan workflow. "
                "A lending_policy_search tool is available. Decide whether a "
                "policy lookup is necessary for the task. If policy text is "
                "already sufficient, do not call the tool. If a lookup is "
                "needed, call the tool with a focused query. Never make or "
                "change a lending decision yourself."
            )
        ),
        HumanMessage(content=task),
    ]

    all_results: list[dict[str, Any]] = []
    queries: list[str] = []

    for step in range(1, max_steps + 1):
        response = model.invoke(messages)
        tool_calls = getattr(response, "tool_calls", None) or []

        if not tool_calls:
            status = "retrieved" if all_results else "not_needed"
            return PolicyLookupResult(
                called=bool(all_results),
                queries=tuple(queries),
                results=tuple(all_results),
                steps=step,
                status=status,
                provider=provider_name,
            )

        messages.append(response)
        for call in tool_calls:
            args = call.get("args") or {}
            query = str(args.get("query", "")).strip()
            if query:
                queries.append(query)

            log_event(
                "rag_tool_called",
                provider=provider_name,
                tool=policy_tool.name,
                query_fingerprint=fingerprint(query),
            )
            result = _run_async(asyncio.wait_for(policy_tool.ainvoke(args), timeout=30))
            normalized = _normalize_mcp_result(result)
            if isinstance(normalized, list):
                all_results.extend(
                    item for item in normalized if isinstance(item, dict)
                )
            elif isinstance(normalized, dict):
                all_results.append(normalized)

            messages.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=call.get("id", "policy-search"),
                )
            )

    return PolicyLookupResult(
        called=bool(all_results),
        queries=tuple(queries),
        results=tuple(all_results),
        steps=max_steps,
        status="retrieved" if all_results else "not_needed",
        provider=provider_name,
    )


def agentic_policy_lookup(
    task: str,
    *,
    max_steps: int = 2,
) -> PolicyLookupResult:
    """Let the primary model decide whether/how to retrieve policy context."""
    try:
        policy_tool = _run_async(_load_policy_tool())
    except Exception as exc:
        log_event("rag_tool_load_failed", error_type=type(exc).__name__)
        return PolicyLookupResult(
            False, (), (), 0, status="failed", error_type=type(exc).__name__
        )

    if policy_tool is None:
        log_event("rag_tool_unavailable", tool="lending_policy_search")
        return PolicyLookupResult(
            False, (), (), 0, status="failed", error_type="tool_unavailable"
        )

    log_event("rag_lookup_started", task_fingerprint=fingerprint(task))

    primary_name, fallback_name = get_configured_provider_names()
    primary_error: Exception | None = None
    try:
        result = _invoke_with_provider(
            lambda: get_provider_model(primary_name),
            f"{primary_name}-primary",
            policy_tool,
            task,
            max_steps,
        )
        log_event(
            "rag_lookup_completed",
            provider=result.provider,
            status=result.status,
            called=result.called,
            steps=result.steps,
            result_count=len(result.results),
        )
        return result
    except Exception as exc:
        primary_error = exc
        log_event(
            "rag_primary_failed",
            provider=f"{primary_name}-primary",
            error_type=type(exc).__name__,
        )

    try:
        result = _invoke_with_provider(
            lambda: get_provider_model(fallback_name),
            f"{fallback_name}-fallback",
            policy_tool,
            task,
            max_steps,
        )
        log_event(
            "rag_lookup_completed",
            provider=result.provider,
            status=result.status,
            called=result.called,
            steps=result.steps,
            result_count=len(result.results),
        )
        return result
    except Exception as exc:
        log_event(
            "rag_fallback_failed",
            provider=f"{fallback_name}-fallback",
            error_type=type(exc).__name__,
        )
        return PolicyLookupResult(
            called=False,
            queries=(),
            results=(),
            steps=0,
            status="failed",
            provider=f"{fallback_name}-fallback",
            error_type=(
                f"primary={type(primary_error).__name__};"
                f"fallback={type(exc).__name__}"
            ),
        )
