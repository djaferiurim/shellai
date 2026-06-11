"""Chat-with-your-files: a tiny local RAG index.

Embeddings come from Ollama (default ``nomic-embed-text``) or OpenAI, so we
keep the "no heavy dependencies" promise — vectors are stored as JSON and
similarity search is plain Python. Good enough for personal knowledge bases.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import httpx
from platformdirs import user_data_dir

from .config import APP_NAME, Config

INDEX_DIR = Path(user_data_dir(APP_NAME)) / "index"
INDEX_PATH = INDEX_DIR / "store.json"

TEXT_SUFFIXES = {
    ".txt", ".md", ".markdown", ".rst", ".py", ".js", ".ts", ".tsx", ".jsx",
    ".json", ".yaml", ".yml", ".toml", ".cfg", ".ini", ".html", ".css",
    ".java", ".go", ".rs", ".rb", ".php", ".c", ".cpp", ".h", ".hpp", ".sh",
    ".sql", ".csv",
}
MAX_FILE_BYTES = 500_000
CHUNK_CHARS = 1200
CHUNK_OVERLAP = 150


@dataclass
class Chunk:
    source: str
    text: str
    vector: List[float]


def _chunk_text(text: str) -> Iterable[str]:
    step = CHUNK_CHARS - CHUNK_OVERLAP
    for start in range(0, len(text), step):
        piece = text[start : start + CHUNK_CHARS].strip()
        if piece:
            yield piece


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class Embedder:
    """Embeds text via Ollama or OpenAI, matching the configured provider."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        # Use OpenAI embeddings only when explicitly on OpenAI with a key.
        self.use_openai = cfg.provider == "openai" and bool(cfg.openai_api_key)

    def embed(self, text: str) -> List[float]:
        if self.use_openai:
            return self._embed_openai(text)
        return self._embed_ollama(text)

    def _embed_ollama(self, text: str) -> List[float]:
        host = self.cfg.ollama_host.rstrip("/")
        # Newer Ollama exposes /api/embed; older ones use /api/embeddings.
        try:
            resp = httpx.post(
                f"{host}/api/embed",
                json={"model": self.cfg.embed_model, "input": text},
                timeout=120,
            )
            if resp.status_code == 404:
                raise httpx.HTTPStatusError("no /api/embed", request=resp.request, response=resp)
            resp.raise_for_status()
            data = resp.json()
            embeddings = data.get("embeddings")
            if embeddings:
                return embeddings[0]
            if data.get("embedding"):
                return data["embedding"]
        except httpx.HTTPStatusError:
            pass

        resp = httpx.post(
            f"{host}/api/embeddings",
            json={"model": self.cfg.embed_model, "prompt": text},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["embedding"]

    def _embed_openai(self, text: str) -> List[float]:
        resp = httpx.post(
            f"{self.cfg.openai_base_url.rstrip('/')}/embeddings",
            json={"model": "text-embedding-3-small", "input": text},
            headers={"Authorization": f"Bearer {self.cfg.openai_api_key}"},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]


def _iter_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if any(part in {".git", "node_modules", "__pycache__", ".venv"} for part in path.parts):
            continue
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                continue
        except OSError:
            continue
        yield path


def build_index(root: Path, cfg: Config, progress=None) -> int:
    """Index every text file under ``root``. Returns the chunk count."""
    embedder = Embedder(cfg)
    chunks: List[dict] = []
    for path in _iter_files(root):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        rel = str(path.relative_to(root))
        for piece in _chunk_text(text):
            vector = embedder.embed(piece)
            chunks.append({"source": rel, "text": piece, "vector": vector})
            if progress:
                progress(rel)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.write_text(
        json.dumps(
            {
                "root": str(root.resolve()),
                "created": time.time(),
                "provider": "openai" if embedder.use_openai else "ollama",
                "chunks": chunks,
            }
        ),
        encoding="utf-8",
    )
    return len(chunks)


def index_exists() -> bool:
    return INDEX_PATH.exists()


def search(query: str, cfg: Config, top_k: int = 5) -> List[Tuple[str, str, float]]:
    """Return the top_k (source, text, score) matches for ``query``."""
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            "No index found. Run `shellai index <folder>` first."
        )
    store = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    chunks = store.get("chunks", [])
    if not chunks:
        return []

    qvec = Embedder(cfg).embed(query)
    scored = [
        (c["source"], c["text"], _cosine(qvec, c["vector"])) for c in chunks
    ]
    scored.sort(key=lambda x: x[2], reverse=True)
    return scored[:top_k]
