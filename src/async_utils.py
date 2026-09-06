"""Shared helper for calling async code from LangGraph's synchronous nodes.

`asyncio.run()` raises `RuntimeError: asyncio.run() cannot be called from a
running event loop` if it is ever invoked while an event loop is already
running. Every call site in this project (cli.py, Streamlit's app.py) is a
plain synchronous call stack today, so plain `asyncio.run()` has always
worked in practice — but the tech stack explicitly lists FastAPI as an
optional interface, and mounting this graph behind an async FastAPI route
would trigger exactly that error the first time a worker reaches an MCP or
agentic-RAG call.

`run_sync` keeps today's behavior identical (a bare `asyncio.run()`, same
performance, same semantics) when no loop is running, and transparently
falls back to running the coroutine in a short-lived worker thread with its
own event loop when one is already running, so this code stays usable from
an async host without a rewrite.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
from typing import Any, Coroutine, TypeVar

T = TypeVar("T")


def run_sync(coro: "Coroutine[Any, Any, T]") -> T:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No loop running in this thread — the common case for cli.py and
        # Streamlit today. Identical to calling asyncio.run() directly.
        return asyncio.run(coro)

    # A loop is already running on this thread (e.g. an async FastAPI
    # request handler). Run the coroutine to completion on its own loop in
    # a separate thread instead of blocking or raising.
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()
