# MCP Integration Decision (§7.6)

## Chosen Approach: Custom stdio MCP Server via `langchain-mcp-adapters`

The loan-origination copilot exposes its backend domain capabilities through a
custom MCP server (`mcp_server/server.py`) and consumes its tools through
`langchain-mcp-adapters`.

The adapter presents MCP tools as standard LangChain tools that can be invoked
by the LangGraph worker nodes.

The MCP server exposes three domain tools and one policy resource:

```text
Tools:
  applicant_lookup
  bureau_check
  lending_policy_search

Resource:
  policy://credit_policy_manual
```

The committed MCP transcript in `evidence/mcp_transcript.json` records both
actual tool invocations and retrieval of the policy resource.

## Alternatives Considered

| Option | Pros | Cons | Verdict |
|---|---|---|---|
| **MCP server (chosen)** | Discoverable tools/resource; framework interoperability; clear tool/resource boundary | Subprocess overhead per call | ✅ Chosen |
| Direct `@tool` functions | Zero overhead; simplest | Locked to LangChain; no MCP resource primitive | ❌ Rejected |
| Raw DB call from node | Fast | Bypasses interoperability and tool boundary | ❌ Rejected |
| REST API wrapper | HTTP-native | Requires a running service; unnecessary for local/no-Docker scope | ❌ Rejected |

## Why MCP Specifically

1. **Discoverability** — Tools are self-describing and can be consumed by
   compatible clients without rewriting bindings.

2. **Resource primitive** — `credit_policy_manual` is exposed as a readable MCP
   resource rather than a callable tool, matching the semantic distinction
   between reference data and actions.

3. **Transport swappability** — The server currently uses stdio. A future
   multi-client deployment can change transport without rewriting domain tools.

## Tradeoffs Accepted

- Tool invocation through the selected adapter version creates subprocess/session
  overhead. This is acceptable within the local copilot scope.
- `lending_policy_search` uses a local Chroma index with Sentence-Transformers
  embeddings. The MCP contract remains stable while retrieval remains semantic.
