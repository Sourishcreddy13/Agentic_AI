# LLM Provider Policy

The system uses **Google Gemini as the primary model provider**. A configured
backup API provider is permitted for availability and graceful degradation;
the current implementation uses **Groq as the backup provider**.

The provider boundary is centralized in `src/llm/gateway.py`. Worker nodes request
structured outputs through the shared gateway and do not instantiate provider
SDKs directly.

Both primary and fallback providers are required to produce the same requested
Pydantic schema at structured-output boundaries.

## Failure Handling

Provider failures are observable and bounded by the configured request timeout
and retry settings.

The gateway attempts the configured primary provider first. When the primary
provider fails, the configured backup provider is attempted.

If both providers fail, the gateway raises a terminal error for the graph's
reflection/self-healing controller to classify and handle.

## Provider Configuration

The active configuration is:

```text
Primary   → Gemini
Fallback  → Groq
```

Provider selection is dynamic at runtime so the boundary remains testable and
supports deterministic provider fakes in the regression suite.

No provider API key is stored in source control. Credentials are supplied
through environment variables described by `.env.example`.
