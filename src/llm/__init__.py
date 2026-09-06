"""LLM provider abstraction for Phase 3."""

from .gateway import invoke_structured_with_fallback

__all__ = [
    "invoke_structured_with_fallback",
]
