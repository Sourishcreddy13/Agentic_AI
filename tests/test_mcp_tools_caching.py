"""AC-09/AC-10: get_mcp_tools() caches the tool/schema listing.

Previously the project built a fresh MultiServerMCPClient and re-ran the
stdio ListTools handshake on *every* call (including once per
invoke_mcp_tool_sync call), i.e. two subprocess spawns per tool call. The
schema list is invariant within a process, so it is now cached at module
scope in src/mcp_client.py after the first successful load, and reset only
via reset_mcp_tools_cache() or refresh=True. This test spies on
_build_client to prove the client/session is genuinely built once across
repeated calls, rather than asserting on log lines or timing.
"""
from __future__ import annotations

import asyncio

import src.mcp_client as mcp_client
from src.mcp_client import get_mcp_tools, reset_mcp_tools_cache


def run(coro):
    return asyncio.run(coro)


def test_get_mcp_tools_builds_client_once_across_repeated_calls(monkeypatch):
    reset_mcp_tools_cache()
    build_calls = []
    real_build_client = mcp_client._build_client

    def spy_build_client():
        build_calls.append(1)
        return real_build_client()

    monkeypatch.setattr(mcp_client, "_build_client", spy_build_client)

    try:
        first = run(get_mcp_tools())
        second = run(get_mcp_tools())
        third = run(get_mcp_tools())
    finally:
        reset_mcp_tools_cache()

    assert len(build_calls) == 1, (
        f"expected exactly one client build across 3 calls, got {len(build_calls)}"
    )
    assert first is second is third
    assert len(first) >= 2  # AC-09: at least 2 tools exposed


def test_refresh_true_forces_a_rebuild(monkeypatch):
    reset_mcp_tools_cache()
    build_calls = []
    real_build_client = mcp_client._build_client

    def spy_build_client():
        build_calls.append(1)
        return real_build_client()

    monkeypatch.setattr(mcp_client, "_build_client", spy_build_client)

    try:
        run(get_mcp_tools())
        run(get_mcp_tools(refresh=True))
    finally:
        reset_mcp_tools_cache()

    assert len(build_calls) == 2


def test_reset_mcp_tools_cache_forces_a_rebuild_on_next_call(monkeypatch):
    reset_mcp_tools_cache()
    build_calls = []
    real_build_client = mcp_client._build_client

    def spy_build_client():
        build_calls.append(1)
        return real_build_client()

    monkeypatch.setattr(mcp_client, "_build_client", spy_build_client)

    try:
        run(get_mcp_tools())
        reset_mcp_tools_cache()
        run(get_mcp_tools())
    finally:
        reset_mcp_tools_cache()

    assert len(build_calls) == 2
