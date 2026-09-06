"""
MCP client adapter (AC-10).

v0.3.2 API: MultiServerMCPClient has no async context manager.
- `await client.get_tools()` — async, loads tool schemas (each invocation
  creates a fresh session internally, so no persistent connection to manage).
- Tools returned are standard LangChain BaseTool objects; each `.ainvoke()`
  spawns a new MCP session under the hood.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import StdioConnection

from src.observability.audit_log import log_event

REPO_ROOT = Path(__file__).resolve().parent.parent
SERVER_SCRIPT = REPO_ROOT / "mcp_server" / "server.py"


def _build_client() -> MultiServerMCPClient:
    connection: StdioConnection = {
        "transport": "stdio",
        "command": sys.executable,
        "args": [str(SERVER_SCRIPT)],
        "cwd": str(REPO_ROOT),
    }
    return MultiServerMCPClient({"loan_origination": connection})


async def get_mcp_tools():
    """
    Returns a list of LangChain BaseTool objects ready for bind_tools().
    Each tool invocation creates its own MCP session (v0.3.2 behaviour).
    """
    client = _build_client()
    return await client.get_tools()


def _normalize_mcp_result(result):
    """Normalize common langchain-mcp-adapters result representations."""
    import json

    if isinstance(result, dict):
        return result
    if isinstance(result, str):
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            return result
    if isinstance(result, list):
        for item in result:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    try:
                        return json.loads(text)
                    except json.JSONDecodeError:
                        pass
            text = getattr(item, "text", None)
            if isinstance(text, str):
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    pass
    structured = getattr(result, "structured_content", None)
    if structured is not None:
        return structured
    artifact = getattr(result, "artifact", None)
    if isinstance(artifact, dict) and artifact.get("structured_content") is not None:
        return artifact["structured_content"]
    return result


def invoke_mcp_tool_sync(tool_name: str, arguments: dict):
    """Synchronously invoke one MCP tool through langchain-mcp-adapters."""
    async def _invoke():
        tools = await get_mcp_tools()
        tool = next((t for t in tools if t.name == tool_name), None)
        if tool is None:
            log_event("mcp_tool_not_found", tool=tool_name)
            raise RuntimeError(f"MCP tool not found: {tool_name}")
        log_event("mcp_tool_started", tool=tool_name, arguments=arguments)
        try:
            result = await tool.ainvoke(arguments)
            normalized = _normalize_mcp_result(result)
            count = len(normalized) if isinstance(normalized, list) else 1
            log_event("mcp_tool_completed", tool=tool_name, result_count=count, success=True)
            return normalized
        except Exception as exc:
            log_event("mcp_tool_failed", tool=tool_name, error_type=type(exc).__name__)
            raise

    return asyncio.run(_invoke())

async def get_mcp_resource(resource_uri: str):
    """
    Read a resource directly from the custom MCP server.

    This is used for AC-09 evidence because get_mcp_tools() only exposes tools.
    """
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client, StdioServerParameters

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER_SCRIPT)],
        cwd=str(REPO_ROOT),
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await session.read_resource(resource_uri)