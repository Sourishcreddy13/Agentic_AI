from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"
CONFIG_FILE = PROJECT_ROOT / "config.yaml"

load_dotenv(ENV_FILE)


def _load_yaml_config() -> dict[str, Any]:
    """Load application configuration from config.yaml."""
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {CONFIG_FILE}"
        )

    with CONFIG_FILE.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    if not isinstance(config, dict):
        raise ValueError(
            "config.yaml must contain a top-level mapping."
        )

    return config


CONFIG = _load_yaml_config()


def get_llm_config() -> dict[str, Any]:
    return CONFIG.get("llm", {})


def get_primary_llm_config() -> dict[str, Any]:
    config = get_llm_config().get("primary")

    if not isinstance(config, dict):
        raise ValueError(
            "config.yaml: llm.primary must be a mapping."
        )

    return config


def get_fallback_llm_config() -> dict[str, Any]:
    config = get_llm_config().get("fallback")

    if not isinstance(config, dict):
        raise ValueError(
            "config.yaml: llm.fallback must be a mapping."
        )

    return config


def get_agent_config(agent_name: str) -> dict[str, Any]:
    config = CONFIG.get("agents", {}).get(agent_name, {})

    if not isinstance(config, dict):
        raise ValueError(
            f"config.yaml: agents.{agent_name} must be a mapping."
        )

    return config


def get_rag_config() -> dict[str, Any]:
    config = CONFIG.get("rag", {})

    if not isinstance(config, dict):
        raise ValueError(
            "config.yaml: rag must be a mapping."
        )

    return config


def get_mcp_config() -> dict[str, Any]:
    config = CONFIG.get("mcp", {})

    if not isinstance(config, dict):
        raise ValueError(
            "config.yaml: mcp must be a mapping."
        )

    return config


def get_google_api_key() -> str:
    value = os.getenv("GOOGLE_API_KEY")

    if not value:
        raise RuntimeError(
            "GOOGLE_API_KEY is not configured. "
            "Set it in .env or the environment."
        )

    return value


def get_groq_api_key() -> str:
    value = os.getenv("GROQ_API_KEY")

    if not value:
        raise RuntimeError(
            "GROQ_API_KEY is not configured. "
            "Set it in .env or the environment."
        )

    return value