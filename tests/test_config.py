"""Configuration layer tests for the provider gateway."""

from src.config import get_fallback_llm_config, get_primary_llm_config


def test_primary_config_is_gemini():
    config = get_primary_llm_config()
    assert config["provider"] == "gemini"
    assert config["model"] == "gemini-3.8-flash"
    assert "temperature" not in config


def test_fallback_config_is_explicitly_configured():
    config = get_fallback_llm_config()
    assert config["provider"] in {"gemini", "groq"}
    assert config["model"]
