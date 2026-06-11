"""Conversation history persistence (JSONL session files)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List

from platformdirs import user_data_dir

from .config import APP_NAME
from .providers import Message

HISTORY_DIR = Path(user_data_dir(APP_NAME)) / "sessions"


def _ensure_dir() -> None:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)


def new_session_path() -> Path:
    """Return a fresh, timestamped session file path."""
    _ensure_dir()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return HISTORY_DIR / f"chat-{stamp}.jsonl"


def append(path: Path, message: Message) -> None:
    """Append a single message to a session file."""
    _ensure_dir()
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(message) + "\n")


def load(path: Path) -> List[Message]:
    """Load all messages from a session file."""
    if not path.exists():
        return []
    messages: List[Message] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                messages.append(json.loads(line))
    return messages


def list_sessions() -> List[Path]:
    """Return saved sessions, newest first."""
    if not HISTORY_DIR.exists():
        return []
    return sorted(
        HISTORY_DIR.glob("chat-*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
