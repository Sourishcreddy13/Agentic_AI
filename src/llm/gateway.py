"""LLM provider gateway with configured primary/fallback providers."""

from __future__ import annotations

from typing import Any, Callable, Type

from pydantic import BaseModel

from src.config import (
    get_fallback_llm_config,
    get_google_api_key,
    get_groq_api_key,
    get_primary_llm_config,
)
from src.observability.audit_log import log_event


SUPPORTED_PROVIDERS = {"gemini", "groq"}


def _gemini_model():
    """Create the configured Gemini chat model."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    config = get_primary_llm_config()

    return ChatGoogleGenerativeAI(
        model=config["model"],
        api_key=get_google_api_key(),
        timeout=config.get("request_timeout_seconds", 60),
        max_retries=config.get("max_retries", 1),
        thinking_level=config.get("thinking_level", "medium"),
    )


def _groq_model():
    """Create the configured Groq fallback chat model."""
    from langchain_groq import ChatGroq

    config = get_fallback_llm_config()

    return ChatGroq(
        model=config["model"],
        temperature=0.0,
        timeout=config.get("request_timeout_seconds", 60),
        max_retries=config.get("max_retries", 1),
        api_key=get_groq_api_key(),
    )


def _provider_factory(provider_name: str) -> Callable[[], Any]:
    """
    Resolve a provider factory dynamically.

    Dynamic resolution is intentional so tests can monkeypatch
    _gemini_model / _groq_model without stale function references.
    """
    provider_name = provider_name.lower().strip()

    factories: dict[str, Callable[[], Any]] = {
        "gemini": _gemini_model,
        "groq": _groq_model,
    }

    try:
        return factories[provider_name]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported LLM provider: {provider_name!r}. "
            f"Supported providers: {sorted(factories)}"
        ) from exc


def _configured_provider_pair() -> tuple[
    tuple[str, dict[str, Any]],
    tuple[str, dict[str, Any]] | None,
]:
    """
    Return the configured primary and fallback provider definitions.

    Provider configuration is validated independently from factory
    resolution so runtime monkeypatching remains testable.
    """
    primary = get_primary_llm_config()
    fallback = get_fallback_llm_config()

    primary_name = str(primary.get("provider", "")).lower().strip()
    fallback_name = str(fallback.get("provider", "")).lower().strip()

    if primary_name not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unsupported primary LLM provider: {primary_name!r}. "
            f"Supported providers: {sorted(SUPPORTED_PROVIDERS)}"
        )

    if fallback_name not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unsupported fallback LLM provider: {fallback_name!r}. "
            f"Supported providers: {sorted(SUPPORTED_PROVIDERS)}"
        )

    return (primary_name, primary), (fallback_name, fallback)


def get_configured_provider_names() -> tuple[str, str]:
    """Return the configured primary and fallback provider names."""
    primary, fallback = _configured_provider_pair()

    if fallback is None:
        return primary[0], ""

    return primary[0], fallback[0]


def get_provider_model(provider_name: str):
    """Instantiate a configured provider model by provider name."""
    factory = _provider_factory(provider_name)
    return factory()


def invoke_structured_with_fallback(
    schema: Type[BaseModel],
    messages: list[Any],
):
    """
    Invoke the configured primary provider and then the fallback provider.

    Both providers are expected to return the requested Pydantic schema.

    The provider factory is resolved dynamically on each invocation so that
    tests can monkeypatch provider constructors correctly.
    """
    if not messages:
        raise ValueError(
            "invoke_structured_with_fallback requires at least one message."
        )

    primary, fallback = _configured_provider_pair()

    providers = [primary]

    if fallback is not None:
        providers.append(fallback)

    last_error: Exception | None = None

    for provider_name, _config in providers:
        factory = _provider_factory(provider_name)

        try:
            log_event(
                "llm_invocation_started",
                provider=provider_name,
                schema=getattr(schema, "__name__", str(schema)),
            )

            structured = factory().with_structured_output(
                schema,
                method="json_schema",
            )

            result = structured.invoke(messages)

            log_event(
                "llm_invocation_completed",
                provider=provider_name,
                schema=getattr(schema, "__name__", str(schema)),
                success=True,
            )

            return result

        except Exception as exc:
            last_error = exc

            log_event(
                "llm_invocation_failed",
                provider=provider_name,
                schema=getattr(schema, "__name__", str(schema)),
                error_type=type(exc).__name__,
            )

    provider_names = [provider[0] for provider in providers]

    if len(provider_names) == 1:
        raise RuntimeError(
            f"Configured LLM provider failed: "
            f"provider={provider_names[0]}; "
            f"last_error={str(last_error)[:180]}"
        ) from last_error

    raise RuntimeError(
        f"Both configured LLM providers failed: "
        f"primary={provider_names[0]}, "
        f"fallback={provider_names[1]}; "
        f"last_error={str(last_error)[:180]}"
    ) from last_error