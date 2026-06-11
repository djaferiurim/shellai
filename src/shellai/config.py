"""Configuration loading and persistence for ShellAI.

Settings are resolved in this order (highest priority first):
1. Command-line flags
2. Environment variables
3. Config file (~/.config/shellai/config.json on Linux/macOS)
4. Built-in defaults
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from platformdirs import user_config_dir

APP_NAME = "shellai"

CONFIG_DIR = Path(user_config_dir(APP_NAME))
CONFIG_PATH = CONFIG_DIR / "config.json"


@dataclass
class Config:
    """Resolved runtime configuration."""

    provider: str = "ollama"  # ollama | openai | anthropic | groq | gemini
    model: str = "llama3.2"
    system_prompt: str = "You are a helpful, concise assistant."
    persona: str = ""  # named preset; overrides system_prompt when set
    temperature: float = 0.7

    # Provider endpoints / credentials
    ollama_host: str = "http://localhost:11434"
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    groq_api_key: str = ""
    gemini_api_key: str = ""

    # Retrieval (chat-with-your-files)
    embed_model: str = "nomic-embed-text"  # ollama embedding model

    # Media generation
    image_model: str = "gpt-image-1"
    image_size: str = "1024x1024"
    video_model: str = "minimax/video-01"
    replicate_api_token: str = ""
    media_output_dir: str = ""  # empty -> ~/ShellAI/media

    @classmethod
    def load(cls) -> "Config":
        """Load config from file + environment, applying defaults."""
        data: dict = {}
        if CONFIG_PATH.exists():
            try:
                data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                data = {}

        cfg = cls(**{k: v for k, v in data.items() if k in cls.__annotations__})

        # Environment overrides
        cfg.openai_api_key = os.environ.get("OPENAI_API_KEY", cfg.openai_api_key)
        if os.environ.get("OPENAI_BASE_URL"):
            cfg.openai_base_url = os.environ["OPENAI_BASE_URL"]
        if os.environ.get("OLLAMA_HOST"):
            cfg.ollama_host = os.environ["OLLAMA_HOST"]
        if os.environ.get("SHELLAI_PROVIDER"):
            cfg.provider = os.environ["SHELLAI_PROVIDER"]
        if os.environ.get("SHELLAI_MODEL"):
            cfg.model = os.environ["SHELLAI_MODEL"]
        if os.environ.get("REPLICATE_API_TOKEN"):
            cfg.replicate_api_token = os.environ["REPLICATE_API_TOKEN"]
        cfg.anthropic_api_key = os.environ.get(
            "ANTHROPIC_API_KEY", cfg.anthropic_api_key
        )
        cfg.groq_api_key = os.environ.get("GROQ_API_KEY", cfg.groq_api_key)
        cfg.gemini_api_key = os.environ.get("GEMINI_API_KEY", cfg.gemini_api_key)

        return cfg

    def save(self) -> Path:
        """Persist current config to disk."""
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(
            json.dumps(asdict(self), indent=2), encoding="utf-8"
        )
        return CONFIG_PATH
