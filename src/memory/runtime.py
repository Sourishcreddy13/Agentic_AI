"""Runtime configuration helpers for Phase 4 memory behavior."""
from __future__ import annotations

from typing import Any


def memory_enabled(config: Any = None) -> bool:
    """Return memory_enabled from LangGraph's per-invocation config.

    Memory is on by default for normal application execution. Tests explicitly
    disable it by passing memory_enabled=False in the invocation config or by
    patching this helper in the hermetic test fixture.
    """
    if config is None:
        return True
    configurable = config.get("configurable", {}) if isinstance(config, dict) else {}
    return bool(configurable.get("memory_enabled", True))


def memory_store_path(config: Any = None) -> str | None:
    """Optional per-invocation Chroma path, intended primarily for isolated tests."""
    if config is None or not isinstance(config, dict):
        return None
    configurable = config.get("configurable", {})
    return configurable.get("memory_store_path")
