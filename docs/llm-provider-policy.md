# LLM Provider Policy

The system uses **Google Gemini as the model provider**, per
`specs/loan_origination_spec.md`. A configured backup API provider is used
for availability and graceful degradation; the current implementation uses
**Groq as the backup provider**. The backup provider is not part of the
original spec's baseline — it is an administrator-approved deviation.
See `docs/deviations.md` (DEV-001) for the rationale, scope, and approval
record; this document describes only how that already-approved backup is
wired into the gateway.

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

## Provider Platforms (Routing Baseline)
The architecture maps logical roles to external API platforms:
* Primary Platform: Google Gemini
* Fallback Platform: Groq (via Admin Deviation DEV-001)

## Provider Configuration (Active Manifest)
The specific model deployments configured at the gateway boundary:
* Primary Model String: gemini-3.8-flash
* Fallback Model String: openai/gpt-oss-120b

Credentials for these configurations are supplied via environment variables (see `.env.example`). No API keys are stored in source control.

## Provider Selection
Provider selection is completely dynamic at runtime. The gateway evaluation loop checks health and errors to swap providers automatically. This decoupled boundary supports injecting deterministic provider fakes into the regression testing suite.

No provider API key is stored in source control. Credentials are supplied
through environment variables described by `.env.example`.
