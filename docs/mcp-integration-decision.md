# MCP Integration Decision (rubric §7.6)

## Chosen Approach: Custom stdio MCP Server via `langchain-mcp-adapters`

The loan-origination copilot exposes its backend domain capabilities
through a custom MCP server (`mcp_server/server.py`) and consumes its tools
through `langchain-mcp-adapters` (`src/mcp_client.py`). The adapter presents
MCP tools as standard LangChain tools that can be invoked by the LangGraph
worker nodes.

The MCP server exposes three domain tools and one policy resource:

```text
Tools:
  applicant_lookup
  bureau_check
  lending_policy_search

Resource:
  policy://credit_policy_manual
```

The committed transcript `evidence/mcp_transcript.json` records both actual
tool invocations and retrieval of the policy resource.

## Alternatives Considered

| Option | Pros | Cons | Verdict |
|---|---|---|---|
| **MCP server (chosen)** | Discoverable tools/resource; framework interoperability; clear tool/resource boundary | Subprocess overhead per call (see "Tradeoffs" below) | ✅ Chosen |
| Direct `@tool` functions | Zero overhead; simplest | Locked to LangChain; no MCP resource primitive | ❌ Rejected |
| Raw DB call from node | Fast | Bypasses interoperability and tool boundary | ❌ Rejected |
| REST API wrapper | HTTP-native | Requires a running service; unnecessary for the local/no-Docker scope | ❌ Rejected |

## Why MCP Specifically

1. **Discoverability** — Tools are self-describing and can be consumed by
   compatible clients without rewriting bindings.
2. **Resource primitive** — `credit_policy_manual` is exposed as a readable
   MCP resource rather than a callable tool, matching the semantic
   distinction between reference data and actions.
3. **Transport swappability** — The server currently uses stdio. A future
   multi-client deployment can change transport without rewriting domain
   tools.

## Tradeoffs Accepted, and What Was Done About Them

- **Tool-schema listing.** `langchain_mcp_adapters.client.MultiServerMCPClient.get_tools()`
  runs its own stdio `ListTools` handshake — a full subprocess round trip
  separate from actually calling a tool. Earlier revisions rebuilt the
  client and re-ran that handshake on *every single* `invoke_mcp_tool_sync`
  call (two subprocess spawns per tool call: one to list tools, one inside
  `.ainvoke()` to call it). The schema list is invariant within a process,
  so `src/mcp_client.py` now caches it at module scope after the first
  successful load (`reset_mcp_tools_cache()` / `refresh=True` available for
  tests that restart the server). This halves the subprocess overhead per
  tool call. Proof: `tests/test_mcp_tools_caching.py` spies on the
  client-build function and shows exactly one build across three calls.
- **Per-invocation session.** The remaining per-call session inside
  `tool.ainvoke()` is inherent to how `langchain-mcp-adapters` v0.3.2
  manages MCP sessions for this transport, and would need a
  persistent-session refactor to remove entirely — acceptable within the
  local-copilot scope where tool calls are not on a hot path.
- **Event-loop safety.** `invoke_mcp_tool_sync` (and the agentic-RAG tool
  loop) bridge async MCP calls into LangGraph's synchronous node functions.
  Both previously called `asyncio.run()` unconditionally, which raises if
  ever invoked from a host that already has an event loop running (e.g. an
  async web framework). `src/async_utils.py::run_sync()` now detects a
  running loop and, if present, runs the coroutine on a helper thread
  instead — harmless for today's sync-only CLI/Streamlit call stacks, but
  removes a latent break for a future async interface.
- **Semantic retrieval.** `lending_policy_search` uses a local Chroma index
  with Sentence-Transformers embeddings. The MCP tool contract remains
  stable regardless of how retrieval is implemented behind it.
