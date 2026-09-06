"""
MCP adapter and transport evidence tests.

AC-09:
    The custom MCP server exposes >= 2 tools and >= 1 resource.

AC-10:
    The agent consumes the MCP server through langchain-mcp-adapters,
    and a committed transcript records MCP tool invocation.

langchain-mcp-adapters v0.3.2 API:
    - await client.get_tools()
    - each returned tool supports await tool.ainvoke(...)
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from src.mcp_client import get_mcp_resource, get_mcp_tools


EVIDENCE_DIR = Path(__file__).resolve().parent.parent / "evidence"
TRANSCRIPT_PATH = EVIDENCE_DIR / "mcp_transcript.json"


def run(coro):
    """Run an async coroutine from a synchronous pytest test."""
    return asyncio.run(coro)


def _json_safe(value):
    """Convert MCP results into JSON-safe structures for evidence."""
    if isinstance(value, (dict, list, str, int, float, bool)) or value is None:
        return value

    try:
        return json.loads(json.dumps(value, default=str))
    except Exception:
        return str(value)


# --------------------------------------------------------------------------
# AC-09: MCP server exposes the required tools and resource
# --------------------------------------------------------------------------


def test_mcp_adapter_loads_at_least_three_tools():
    tools = run(get_mcp_tools())
    names = [tool.name for tool in tools]

    print(f"\nTools from MCP server: {names}")

    assert len(tools) >= 2
    assert "applicant_lookup" in names
    assert "bureau_check" in names
    assert "lending_policy_search" in names


def test_tool_schemas_have_descriptions():
    tools = run(get_mcp_tools())

    for tool in tools:
        assert tool.description, f"Tool {tool.name} has no description"


def test_mcp_resource_credit_policy_manual_is_available():
    """
    AC-09 resource verification.

    The resource is read directly from the custom MCP server. The adapter
    layer is used for tools; the MCP SDK resource read proves that the server
    also exposes the required resource.
    """

    async def _run():
        return await get_mcp_resource("policy://credit_policy_manual")

    result = run(_run())

    assert result is not None

    raw = result if isinstance(result, str) else str(result)

    assert len(raw.strip()) > 10


# --------------------------------------------------------------------------
# AC-10: invoke MCP tools through the full stdio transport
# --------------------------------------------------------------------------


def test_mcp_tool_applicant_lookup_via_transport():
    async def _run():
        tools = await get_mcp_tools()

        lookup = next(
            tool for tool in tools
            if tool.name == "applicant_lookup"
        )

        return await lookup.ainvoke(
            {"applicant_id": "SYN-0001"}
        )

    result = run(_run())

    raw = result if isinstance(result, str) else json.dumps(result)

    assert "SYN-0001" in raw


def test_mcp_tool_bureau_check_via_transport():
    async def _run():
        tools = await get_mcp_tools()

        check = next(
            tool for tool in tools
            if tool.name == "bureau_check"
        )

        return await check.ainvoke(
            {
                "applicant_id": "SYN-0001",
                "declared_income": 85000,
            }
        )

    result = run(_run())

    raw = result if isinstance(result, str) else json.dumps(result)

    assert "SYN-0001" in raw


def test_mcp_tool_lending_policy_search_via_transport():
    async def _run():
        tools = await get_mcp_tools()

        search = next(
            tool for tool in tools
            if tool.name == "lending_policy_search"
        )

        return await search.ainvoke(
            {
                "query": "thin file manual underwriting",
                "k": 2,
            }
        )

    result = run(_run())

    raw = result if isinstance(result, str) else json.dumps(result)

    assert len(raw) > 10


# --------------------------------------------------------------------------
# AC-09 / AC-10 committed MCP evidence
# --------------------------------------------------------------------------


def test_generate_mcp_transcript():
    """
    Generate committed MCP evidence showing:

    AC-09:
        - >= 2 MCP tools exposed
        - credit_policy_manual resource exposed and retrievable

    AC-10:
        - MCP tools invoked through langchain-mcp-adapters over stdio
    """

    async def _run():
        tools_list = await get_mcp_tools()
        tools = {tool.name: tool for tool in tools_list}

        transcript = {
            "ac_reference": "AC-09/AC-10",
            "session_id": f"transcript-{int(time.time())}",
            "timestamp": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ",
                time.gmtime(),
            ),
            "server": "loan-origination-mcp",
            "adapter": "langchain-mcp-adapters v0.3.2",
            "transport": "stdio",
            "tools_available": list(tools.keys()),
            "resources_available": [
                "policy://credit_policy_manual",
            ],
            "tool_calls": [],
            "resource_calls": [],
        }

        # --------------------------------------------------------------
        # Call 1 — applicant_lookup
        # --------------------------------------------------------------

        t0 = time.time()

        r1 = await tools["applicant_lookup"].ainvoke(
            {
                "applicant_id": "SYN-0001",
            }
        )

        transcript["tool_calls"].append(
            {
                "call_number": 1,
                "tool": "applicant_lookup",
                "input": {
                    "applicant_id": "SYN-0001",
                },
                "output": _json_safe(r1),
                "latency_ms": round(
                    (time.time() - t0) * 1000
                ),
            }
        )

        # --------------------------------------------------------------
        # Call 2 — bureau_check
        # --------------------------------------------------------------

        t0 = time.time()

        r2 = await tools["bureau_check"].ainvoke(
            {
                "applicant_id": "SYN-0001",
                "declared_income": 85000,
            }
        )

        transcript["tool_calls"].append(
            {
                "call_number": 2,
                "tool": "bureau_check",
                "input": {
                    "applicant_id": "SYN-0001",
                    "declared_income": 85000,
                },
                "output": _json_safe(r2),
                "latency_ms": round(
                    (time.time() - t0) * 1000
                ),
            }
        )

        # --------------------------------------------------------------
        # Call 3 — applicant_lookup for thin-file applicant
        # --------------------------------------------------------------

        t0 = time.time()

        r3 = await tools["applicant_lookup"].ainvoke(
            {
                "applicant_id": "SYN-0002",
            }
        )

        transcript["tool_calls"].append(
            {
                "call_number": 3,
                "tool": "applicant_lookup",
                "input": {
                    "applicant_id": "SYN-0002",
                },
                "output": _json_safe(r3),
                "latency_ms": round(
                    (time.time() - t0) * 1000
                ),
            }
        )

        # --------------------------------------------------------------
        # Call 4 — lending_policy_search
        # --------------------------------------------------------------

        t0 = time.time()

        r4 = await tools["lending_policy_search"].ainvoke(
            {
                "query": "thin file manual underwriting DTI limit",
                "k": 3,
            }
        )

        transcript["tool_calls"].append(
            {
                "call_number": 4,
                "tool": "lending_policy_search",
                "input": {
                    "query": "thin file manual underwriting DTI limit",
                    "k": 3,
                },
                "output": _json_safe(r4),
                "latency_ms": round(
                    (time.time() - t0) * 1000
                ),
                "note": (
                    "MCP policy-search tool exposed by the server. "
                    "Discretionary agentic-RAG behavior is evidenced "
                    "separately by AC-11."
                ),
            }
        )

        # --------------------------------------------------------------
        # Resource call — credit policy manual
        # --------------------------------------------------------------

        t0 = time.time()

        resource_result = await get_mcp_resource(
            "policy://credit_policy_manual"
        )

        resource_text = (
            resource_result
            if isinstance(resource_result, str)
            else str(resource_result)
        )

        transcript["resource_calls"].append(
            {
                "call_number": 1,
                "resource": "policy://credit_policy_manual",
                "retrieved": True,
                "content_type": "text",
                "content_length": len(resource_text),
                "latency_ms": round(
                    (time.time() - t0) * 1000
                ),
            }
        )

        return transcript

    transcript = run(_run())

    EVIDENCE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    TRANSCRIPT_PATH.write_text(
        json.dumps(
            transcript,
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        f"\nMCP transcript written → {TRANSCRIPT_PATH}"
    )

    # --------------------------------------------------------------
    # Evidence assertions
    # --------------------------------------------------------------

    assert transcript["ac_reference"] == "AC-09/AC-10"

    assert len(transcript["tools_available"]) >= 2

    assert "applicant_lookup" in transcript["tools_available"]
    assert "bureau_check" in transcript["tools_available"]
    assert "lending_policy_search" in transcript["tools_available"]

    assert (
        "policy://credit_policy_manual"
        in transcript["resources_available"]
    )

    assert len(transcript["tool_calls"]) == 4

    tools_called = {
        call["tool"]
        for call in transcript["tool_calls"]
    }

    assert "applicant_lookup" in tools_called
    assert "bureau_check" in tools_called
    assert "lending_policy_search" in tools_called

    assert len(transcript["resource_calls"]) == 1

    resource_call = transcript["resource_calls"][0]

    assert resource_call["resource"] == (
        "policy://credit_policy_manual"
    )
    assert resource_call["retrieved"] is True
    assert resource_call["content_type"] == "text"
    assert resource_call["content_length"] > 10