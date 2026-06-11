"""Media generation backends: images (OpenAI) and video (Replicate).

Both return the path(s) to the saved file(s). Outputs are written to a
timestamped folder so repeated runs never clobber each other.
"""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

import httpx
from platformdirs import user_pictures_dir

from .config import APP_NAME, Config

# How long to wait for slow video jobs (seconds).
VIDEO_POLL_TIMEOUT = 600
VIDEO_POLL_INTERVAL = 3


@dataclass
class MediaResult:
    """Where the generated media landed, plus the prompt actually used."""

    paths: List[Path]
    prompt: str


def _output_dir(cfg: Config, kind: str) -> Path:
    base = (
        Path(cfg.media_output_dir)
        if cfg.media_output_dir
        else Path(user_pictures_dir()) / APP_NAME
    )
    stamp = time.strftime("%Y%m%d-%H%M%S")
    target = base / kind / stamp
    target.mkdir(parents=True, exist_ok=True)
    return target


def _slug(text: str, limit: int = 40) -> str:
    keep = [c if c.isalnum() else "-" for c in text.lower()]
    slug = "".join(keep).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug[:limit] or "media"


class ImageGenerator:
    """Generates images with the OpenAI Images API (gpt-image-1 / DALL·E 3)."""

    def __init__(self, cfg: Config) -> None:
        if not cfg.openai_api_key:
            raise ValueError(
                "OpenAI API key not set. Export OPENAI_API_KEY or run "
                "`oshell config set openai_api_key <key>`."
            )
        self.base_url = cfg.openai_base_url.rstrip("/")
        self.api_key = cfg.openai_api_key
        self.model = cfg.image_model
        self.size = cfg.image_size
        self.cfg = cfg

    def generate(self, prompt: str, n: int = 1) -> MediaResult:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model,
            "prompt": prompt,
            "n": n,
            "size": self.size,
        }
        resp = httpx.post(
            f"{self.base_url}/images/generations",
            json=payload,
            headers=headers,
            timeout=180,
        )
        resp.raise_for_status()
        data = resp.json().get("data", [])

        out_dir = _output_dir(self.cfg, "images")
        slug = _slug(prompt)
        paths: List[Path] = []
        for i, item in enumerate(data):
            suffix = "" if len(data) == 1 else f"-{i + 1}"
            dest = out_dir / f"{slug}{suffix}.png"
            if item.get("b64_json"):
                dest.write_bytes(base64.b64decode(item["b64_json"]))
            elif item.get("url"):
                img = httpx.get(item["url"], timeout=120)
                img.raise_for_status()
                dest.write_bytes(img.content)
            else:
                continue
            paths.append(dest)

        if not paths:
            raise RuntimeError("Image API returned no usable data.")
        return MediaResult(paths=paths, prompt=prompt)


class VideoGenerator:
    """Generates video via Replicate's prediction API.

    Works with any text-to-video model on Replicate (default
    ``minimax/video-01``); just change ``video_model`` in config.
    """

    def __init__(self, cfg: Config) -> None:
        if not cfg.replicate_api_token:
            raise ValueError(
                "Replicate token not set. Export REPLICATE_API_TOKEN or run "
                "`oshell config set replicate_api_token <token>`."
            )
        self.token = cfg.replicate_api_token
        self.model = cfg.video_model
        self.cfg = cfg

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

    def generate(self, prompt: str) -> MediaResult:
        # Use the model-scoped predictions endpoint so we don't need a version hash.
        url = f"https://api.replicate.com/v1/models/{self.model}/predictions"
        resp = httpx.post(
            url,
            json={"input": {"prompt": prompt}},
            headers={**self._headers(), "Prefer": "wait"},
            timeout=120,
        )
        resp.raise_for_status()
        prediction = resp.json()

        prediction = self._await_completion(prediction)
        output = prediction.get("output")
        urls = self._collect_urls(output)
        if not urls:
            raise RuntimeError(
                f"Video job finished with status '{prediction.get('status')}' "
                "but produced no output URL."
            )

        out_dir = _output_dir(self.cfg, "videos")
        slug = _slug(prompt)
        paths: List[Path] = []
        for i, video_url in enumerate(urls):
            suffix = "" if len(urls) == 1 else f"-{i + 1}"
            dest = out_dir / f"{slug}{suffix}.mp4"
            data = httpx.get(video_url, timeout=300)
            data.raise_for_status()
            dest.write_bytes(data.content)
            paths.append(dest)
        return MediaResult(paths=paths, prompt=prompt)

    def _await_completion(self, prediction: dict) -> dict:
        terminal = {"succeeded", "failed", "canceled"}
        deadline = time.time() + VIDEO_POLL_TIMEOUT
        while prediction.get("status") not in terminal:
            if time.time() > deadline:
                raise TimeoutError("Video generation timed out.")
            poll_url = prediction.get("urls", {}).get("get")
            if not poll_url:
                break
            time.sleep(VIDEO_POLL_INTERVAL)
            r = httpx.get(poll_url, headers=self._headers(), timeout=60)
            r.raise_for_status()
            prediction = r.json()

        if prediction.get("status") == "failed":
            raise RuntimeError(f"Video generation failed: {prediction.get('error')}")
        return prediction

    @staticmethod
    def _collect_urls(output) -> List[str]:
        if not output:
            return []
        if isinstance(output, str):
            return [output]
        if isinstance(output, list):
            return [u for u in output if isinstance(u, str)]
        return []
