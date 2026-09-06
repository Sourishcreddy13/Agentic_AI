"""Unit tests for the production Phase 3 LLM gateway."""
from __future__ import annotations

import pytest
from src.state.schema import ApplicantProfile, KYCResult
from langchain_core.messages import HumanMessage


class _FakeStructured:
    def __init__(self, return_value):
        self._value = return_value

    def invoke(self, messages):
        return self._value


class _FakeModel:
    def __init__(self, return_value):
        self._value = return_value
        self.calls = []

    def with_structured_output(self, schema, **kwargs):
        self.calls.append(kwargs)
        return _FakeStructured(self._value)


class _FailModel:
    def with_structured_output(self, schema, **kwargs):
        return self

    def invoke(self, messages):
        raise RuntimeError("synthetic provider failure")


def test_gateway_returns_primary_result_when_gemini_succeeds(monkeypatch):
    import src.llm.gateway as gw

    expected = ApplicantProfile(
        applicant_id="SYN-0001",
        full_name="Test",
        dob_synthetic="1990-01-01",
        declared_income=50000,
        declared_employment="Tester",
    )
    primary = _FakeModel(expected)
    fallback = _FakeModel(KYCResult(status="pass", checks_performed=["id"], confidence=0.9))
    monkeypatch.setattr(gw, "_gemini_model", lambda: primary)
    monkeypatch.setattr(gw, "_groq_model", lambda: fallback)

    result = gw.invoke_structured_with_fallback(
        ApplicantProfile,
        [HumanMessage(content="Return the synthetic applicant profile.")],
)

    assert isinstance(result, ApplicantProfile)
    assert result.applicant_id == "SYN-0001"
    assert primary.calls == [{"method": "json_schema"}]
    assert fallback.calls == []


def test_gateway_falls_back_to_configured_backup_when_primary_fails(monkeypatch):
    import src.llm.gateway as gw

    expected = KYCResult(status="pass", checks_performed=["id"], confidence=0.9)
    fallback = _FakeModel(expected)
    monkeypatch.setattr(gw, "_gemini_model", lambda: _FailModel())
    monkeypatch.setattr(gw, "_groq_model", lambda: fallback)

    result = gw.invoke_structured_with_fallback(
        KYCResult,
        [HumanMessage(content="Return the synthetic KYC result.")],
    )

    assert isinstance(result, KYCResult)
    assert result.status == "pass"
    assert fallback.calls == [{"method": "json_schema"}]


def test_gateway_raises_runtime_error_when_both_providers_fail(monkeypatch):
    import src.llm.gateway as gw

    monkeypatch.setattr(gw, "_gemini_model", lambda: _FailModel())
    monkeypatch.setattr(gw, "_groq_model", lambda: _FailModel())

    with pytest.raises(RuntimeError, match="Both configured LLM providers failed"):
        gw.invoke_structured_with_fallback(
            ApplicantProfile,
            [HumanMessage(content="Return the synthetic applicant profile.")],
        )


def test_gateway_error_message_names_both_providers(monkeypatch):
    import src.llm.gateway as gw

    monkeypatch.setattr(gw, "_gemini_model", lambda: _FailModel())
    monkeypatch.setattr(gw, "_groq_model", lambda: _FailModel())

    with pytest.raises(RuntimeError) as exc_info:
        gw.invoke_structured_with_fallback(
            ApplicantProfile,
            [HumanMessage(content="Return the synthetic applicant profile.")],
        )

    assert "primary=gemini" in str(exc_info.value)
    assert "fallback=groq" in str(exc_info.value)


def test_gemini_model_uses_config_without_temperature(monkeypatch):
    langchain_google_genai = pytest.importorskip("langchain_google_genai")
    import src.llm.gateway as gw

    captured = {}

    class FakeGemini:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        langchain_google_genai,
        "ChatGoogleGenerativeAI",
        FakeGemini,
    )
    monkeypatch.setattr(gw, "get_google_api_key", lambda: "test-key")
    monkeypatch.setattr(
        gw,
        "get_primary_llm_config",
        lambda: {
            "provider": "gemini",
            "model": "gemini-3.8-flash",
            "request_timeout_seconds": 60,
            "max_retries": 1,
            "thinking_level": "medium",
        },
    )

    gw._gemini_model()

    assert captured["model"] == "gemini-3.8-flash"
    assert captured["api_key"] == "test-key"
    assert captured["timeout"] == 60
    assert captured["max_retries"] == 1
    assert captured["thinking_level"] == "medium"
    assert "temperature" not in captured
