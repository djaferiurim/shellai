"""Tests for the provider factory and credential validation."""

import pytest

from shellai.config import Config
from shellai.providers import (
    AnthropicProvider,
    GeminiProvider,
    GroqProvider,
    OllamaProvider,
    OpenAIProvider,
    get_provider,
)


def test_factory_returns_ollama_by_default():
    cfg = Config()
    assert isinstance(get_provider(cfg), OllamaProvider)


def test_factory_openai():
    cfg = Config(provider="openai", openai_api_key="sk-test")
    assert isinstance(get_provider(cfg), OpenAIProvider)


def test_factory_groq():
    cfg = Config(provider="groq", groq_api_key="gsk-test")
    assert isinstance(get_provider(cfg), GroqProvider)


def test_factory_gemini():
    cfg = Config(provider="gemini", gemini_api_key="g-test")
    assert isinstance(get_provider(cfg), GeminiProvider)


def test_factory_anthropic():
    cfg = Config(provider="anthropic", anthropic_api_key="a-test")
    assert isinstance(get_provider(cfg), AnthropicProvider)


def test_factory_unknown_provider_raises():
    cfg = Config(provider="bogus")
    with pytest.raises(ValueError):
        get_provider(cfg)


def test_openai_missing_key_raises():
    cfg = Config(provider="openai", openai_api_key="")
    with pytest.raises(ValueError):
        get_provider(cfg)


def test_anthropic_missing_key_raises():
    cfg = Config(provider="anthropic", anthropic_api_key="")
    with pytest.raises(ValueError):
        get_provider(cfg)
