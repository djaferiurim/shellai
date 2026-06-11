"""LLM provider layer.

Both providers expose a single ``stream_chat`` method that yields text
chunks as they arrive, so the CLI can render tokens live.
"""

from __future__ import annotations

import json
from typing import Iterator, List, Protocol

import httpx

from .config import Config

Message = dict  # {"role": "user" | "assistant" | "system", "content": str}


class Provider(Protocol):
    """Common interface every LLM backend implements."""

    name: str

    def stream_chat(self, messages: List[Message]) -> Iterator[str]:
        """Yield response text chunks for the given conversation."""
        ...

    def list_models(self) -> List[str]:
        """Return available model names (best effort)."""
        ...


class OllamaProvider:
    """Talks to a local Ollama server (https://ollama.com)."""

    name = "ollama"

    def __init__(self, cfg: Config) -> None:
        self.host = cfg.ollama_host.rstrip("/")
        self.model = cfg.model
        self.temperature = cfg.temperature

    def stream_chat(self, messages: List[Message]) -> Iterator[str]:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": self.temperature},
        }
        with httpx.stream(
            "POST", f"{self.host}/api/chat", json=payload, timeout=None
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                if data.get("done"):
                    break
                chunk = data.get("message", {}).get("content", "")
                if chunk:
                    yield chunk

    def list_models(self) -> List[str]:
        try:
            resp = httpx.get(f"{self.host}/api/tags", timeout=10)
            resp.raise_for_status()
            return [m["name"] for m in resp.json().get("models", [])]
        except httpx.HTTPError:
            return []


class OpenAICompatibleProvider:
    """Base for any service speaking the OpenAI Chat Completions API.

    OpenAI, Groq, and Gemini (via its OpenAI-compatible endpoint) all share
    this exact wire format, so they differ only in base URL, key, and the
    error message shown when the key is missing.
    """

    name = "openai"
    _key_hint = "OPENAI_API_KEY"

    def __init__(self, base_url: str, api_key: str, model: str, temperature: float):
        if not api_key:
            raise ValueError(
                f"{self.name.title()} API key not set. Export {self._key_hint} or run "
                f"`oshell config set {self.name}_api_key <key>`."
            )
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature

    def stream_chat(self, messages: List[Message]) -> Iterator[str]:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "temperature": self.temperature,
        }
        with httpx.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=None,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[len("data: ") :]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                choices = data.get("choices") or []
                if not choices:
                    continue
                chunk = choices[0].get("delta", {}).get("content")
                if chunk:
                    yield chunk

    def list_models(self) -> List[str]:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            resp = httpx.get(
                f"{self.base_url}/models", headers=headers, timeout=10
            )
            resp.raise_for_status()
            return sorted(m["id"] for m in resp.json().get("data", []))
        except httpx.HTTPError:
            return []


class OpenAIProvider(OpenAICompatibleProvider):
    """OpenAI Chat Completions API."""

    name = "openai"
    _key_hint = "OPENAI_API_KEY"

    def __init__(self, cfg: Config) -> None:
        super().__init__(
            cfg.openai_base_url, cfg.openai_api_key, cfg.model, cfg.temperature
        )


class GroqProvider(OpenAICompatibleProvider):
    """Groq — extremely fast inference, OpenAI-compatible API."""

    name = "groq"
    _key_hint = "GROQ_API_KEY"

    def __init__(self, cfg: Config) -> None:
        model = cfg.model if cfg.model else "llama-3.3-70b-versatile"
        super().__init__(
            "https://api.groq.com/openai/v1",
            cfg.groq_api_key,
            model,
            cfg.temperature,
        )


class GeminiProvider(OpenAICompatibleProvider):
    """Google Gemini via its OpenAI-compatible endpoint."""

    name = "gemini"
    _key_hint = "GEMINI_API_KEY"

    def __init__(self, cfg: Config) -> None:
        model = cfg.model if cfg.model else "gemini-1.5-flash"
        super().__init__(
            "https://generativelanguage.googleapis.com/v1beta/openai",
            cfg.gemini_api_key,
            model,
            cfg.temperature,
        )


class AnthropicProvider:
    """Anthropic Claude via the native Messages API (SSE streaming)."""

    name = "anthropic"

    def __init__(self, cfg: Config) -> None:
        if not cfg.anthropic_api_key:
            raise ValueError(
                "Anthropic API key not set. Export ANTHROPIC_API_KEY or run "
                "`oshell config set anthropic_api_key <key>`."
            )
        self.api_key = cfg.anthropic_api_key
        self.model = cfg.model if cfg.model else "claude-3-5-sonnet-latest"
        self.temperature = cfg.temperature

    @staticmethod
    def _split(messages: List[Message]):
        """Anthropic takes the system prompt separately from the turns."""
        system = ""
        turns: List[Message] = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                turns.append({"role": m["role"], "content": m["content"]})
        return system, turns

    def stream_chat(self, messages: List[Message]) -> Iterator[str]:
        system, turns = self._split(messages)
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": turns,
            "max_tokens": 4096,
            "temperature": self.temperature,
            "stream": True,
        }
        if system:
            payload["system"] = system
        with httpx.stream(
            "POST",
            "https://api.anthropic.com/v1/messages",
            json=payload,
            headers=headers,
            timeout=None,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                try:
                    data = json.loads(line[len("data: ") :])
                except json.JSONDecodeError:
                    continue
                if data.get("type") == "content_block_delta":
                    chunk = data.get("delta", {}).get("text")
                    if chunk:
                        yield chunk

    def list_models(self) -> List[str]:
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }
        try:
            resp = httpx.get(
                "https://api.anthropic.com/v1/models", headers=headers, timeout=10
            )
            resp.raise_for_status()
            return sorted(m["id"] for m in resp.json().get("data", []))
        except httpx.HTTPError:
            return []


def get_provider(cfg: Config) -> Provider:
    """Factory that returns the configured provider instance."""
    providers = {
        "openai": OpenAIProvider,
        "ollama": OllamaProvider,
        "anthropic": AnthropicProvider,
        "groq": GroqProvider,
        "gemini": GeminiProvider,
    }
    cls = providers.get(cfg.provider)
    if cls is None:
        valid = ", ".join(sorted(providers))
        raise ValueError(f"Unknown provider: {cfg.provider!r} (use one of: {valid})")
    return cls(cfg)
