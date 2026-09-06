from langchain_core.messages import AIMessage

from src.rag.agentic_rag import PolicyLookupResult

from src.rag import agentic_rag


class _FakeTool:
    name = "lending_policy_search"

    async def ainvoke(self, args):
        return [{"clause_id": "DTI-001", "snippet": "Standard max DTI 0.40", "score": 1.0}]


class _FakeModel:
    def __init__(self):
        self.called = 0

    def bind_tools(self, tools):
        self.tools = tools
        return self

    def invoke(self, messages):
        self.called += 1
        if self.called == 1:
            return AIMessage(
                content="",
                tool_calls=[{
                    "name": "lending_policy_search",
                    "args": {"query": "standard DTI limit"},
                    "id": "call-1",
                }],
            )
        return AIMessage(content="No more lookup needed.")


def test_agentic_rag_allows_model_to_call_policy_tool(monkeypatch):
    model = _FakeModel()
    monkeypatch.setattr(agentic_rag, "get_configured_provider_names", lambda: ("gemini", "groq"))
    monkeypatch.setattr(agentic_rag, "get_provider_model", lambda name: model)
    monkeypatch.setattr(agentic_rag, "_load_policy_tool", lambda: None)

    # Patch the async loader correctly for the test.
    async def load_tool():
        return _FakeTool()
    monkeypatch.setattr(agentic_rag, "_load_policy_tool", load_tool)

    result = agentic_rag.agentic_policy_lookup("Need the standard DTI policy")
    assert result.called is True
    assert result.status == "retrieved"
    assert result.queries == ("standard DTI limit",)
    assert result.results[0]["clause_id"] == "DTI-001"
    assert result.steps == 2


def test_agentic_rag_can_decline_to_call_tool(monkeypatch):
    class NoCallModel(_FakeModel):
        def invoke(self, messages):
            return AIMessage(content="No lookup needed.")

    monkeypatch.setattr(agentic_rag, "get_configured_provider_names", lambda: ("gemini", "groq"))
    monkeypatch.setattr(agentic_rag, "get_provider_model", lambda name: NoCallModel())

    async def load_tool():
        return _FakeTool()
    monkeypatch.setattr(agentic_rag, "_load_policy_tool", load_tool)

    result = agentic_rag.agentic_policy_lookup("Explain an already known policy fact")
    assert result.called is False
    assert result.status == "not_needed"
    assert result.results == ()


def test_agentic_rag_falls_back_to_groq_when_gemini_fails(monkeypatch):
    class FailingModel:
        def bind_tools(self, tools):
            return self
        def invoke(self, messages):
            raise RuntimeError("synthetic Gemini failure")

    class GroqModel:
        def bind_tools(self, tools):
            self.tools = tools
            return self
        def invoke(self, messages):
            if not any(getattr(m, "tool_calls", None) for m in messages):
                return AIMessage(
                    content="",
                    tool_calls=[{
                        "name": "lending_policy_search",
                        "args": {"query": "standard DTI limit"},
                        "id": "groq-call-1",
                    }],
                )
            return AIMessage(content="done")

    monkeypatch.setattr(agentic_rag, "get_configured_provider_names", lambda: ("gemini", "groq"))
    monkeypatch.setattr(agentic_rag, "get_provider_model", lambda name: FailingModel() if name == "gemini" else GroqModel())

    async def load_tool():
        return _FakeTool()
    monkeypatch.setattr(agentic_rag, "_load_policy_tool", load_tool)

    result = agentic_rag.agentic_policy_lookup("Need the DTI policy")
    assert result.status == "retrieved"
    assert result.provider == "groq-fallback"
    assert result.called is True


def test_agentic_rag_reports_failure_distinct_from_not_needed(monkeypatch):
    async def load_tool():
        raise RuntimeError("synthetic MCP outage")
    monkeypatch.setattr(agentic_rag, "_load_policy_tool", load_tool)

    result = agentic_rag.agentic_policy_lookup("Need policy")
    assert isinstance(result, PolicyLookupResult)
    assert result.status == "failed"
    assert result.called is False
    assert result.error_type == "RuntimeError"
