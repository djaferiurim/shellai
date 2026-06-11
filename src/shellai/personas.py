"""Named personas (system-prompt presets).

Built-in personas live here; users can add their own with
``shellai persona add <name> "<system prompt>"`` which writes to a small JSON
file next to the config.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from .config import CONFIG_DIR

PERSONA_PATH = CONFIG_DIR / "personas.json"

BUILTIN: Dict[str, str] = {
    "default": "You are a helpful, concise assistant.",
    "reviewer": (
        "You are a meticulous senior code reviewer. Point out bugs, security "
        "issues, edge cases, and style problems. Be direct and specific, cite "
        "line-level concerns, and suggest concrete fixes."
    ),
    "teacher": (
        "You are a patient programming teacher. Explain concepts step by step "
        "with simple analogies and short examples. Check understanding and "
        "avoid jargon unless you define it."
    ),
    "shell": (
        "You are a command-line expert. Prefer giving the exact command(s) to "
        "run, with a one-line explanation. Assume a competent user."
    ),
    "rubber-duck": (
        "You are a rubber-duck debugging partner. Ask probing questions that "
        "help the user reason through their problem rather than giving the "
        "answer outright."
    ),
    "concise": "Answer in as few words as possible. No preamble. No filler.",
    "pirate": "You are a witty pirate. Answer correctly, but talk like a pirate.",
}


def _load_custom() -> Dict[str, str]:
    if not PERSONA_PATH.exists():
        return {}
    try:
        return json.loads(PERSONA_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def all_personas() -> Dict[str, str]:
    """Built-ins merged with user-defined personas (custom wins)."""
    merged = dict(BUILTIN)
    merged.update(_load_custom())
    return merged


def get(name: str) -> str | None:
    """Return the system prompt for ``name``, or None if unknown."""
    return all_personas().get(name)


def add(name: str, prompt: str) -> Path:
    """Create or update a custom persona."""
    custom = _load_custom()
    custom[name] = prompt
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    PERSONA_PATH.write_text(json.dumps(custom, indent=2), encoding="utf-8")
    return PERSONA_PATH


def remove(name: str) -> bool:
    """Delete a custom persona. Returns True if it existed."""
    custom = _load_custom()
    if name not in custom:
        return False
    del custom[name]
    PERSONA_PATH.write_text(json.dumps(custom, indent=2), encoding="utf-8")
    return True


def resolve_system_prompt(cfg) -> str:
    """Return the active system prompt, honoring a selected persona."""
    if cfg.persona:
        prompt = get(cfg.persona)
        if prompt:
            return prompt
    return cfg.system_prompt
