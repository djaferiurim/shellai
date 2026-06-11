"""Storyboard mode: brief → planned scenes → image per scene → stitched video.

The LLM acts as a director: it breaks a one-line brief into a sequence of
scenes (each with an image prompt and a caption). We generate an image for
every scene, then stitch them into an MP4 slideshow with ffmpeg.

ffmpeg is only required for the final stitch — the per-scene images are saved
regardless, so the command still produces useful output if ffmpeg is missing.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

from .config import APP_NAME, Config
from .media import ImageGenerator, _output_dir, _slug
from .providers import Message, Provider

_DIRECTOR_SYSTEM = """\
You are a creative director planning a short visual story. Given a brief, break \
it into a sequence of distinct scenes. Respond with ONLY a JSON array; each \
element is an object: {{"prompt": "<detailed text-to-image prompt with \
composition, lighting, style and mood>", "caption": "<short on-screen \
caption>"}}. Produce exactly {n} scenes. No markdown, no code fences, no extra \
text."""


@dataclass
class Scene:
    prompt: str
    caption: str
    image: Optional[Path] = None


@dataclass
class StoryboardResult:
    scenes: List[Scene]
    video: Optional[Path]
    image_dir: Path


def _parse_scenes(raw: str, fallback_brief: str, n: int) -> List[Scene]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    # Pull out the first JSON array if the model added prose.
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1:
        text = text[start : end + 1]
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = []

    scenes: List[Scene] = []
    for item in data:
        if isinstance(item, dict) and item.get("prompt"):
            scenes.append(
                Scene(prompt=str(item["prompt"]), caption=str(item.get("caption", "")))
            )
    # Fallback: if parsing failed, make simple scenes from the brief.
    if not scenes:
        scenes = [
            Scene(prompt=f"{fallback_brief}, scene {i + 1}", caption="")
            for i in range(n)
        ]
    return scenes


class StoryboardAgent:
    """Plans scenes with the LLM and renders them into a video."""

    def __init__(self, provider: Provider, image_gen: ImageGenerator, cfg: Config):
        self.provider = provider
        self.image_gen = image_gen
        self.cfg = cfg

    def plan(self, brief: str, n: int) -> List[Scene]:
        system = _DIRECTOR_SYSTEM.format(n=n)
        messages: List[Message] = [
            {"role": "system", "content": system},
            {"role": "user", "content": brief},
        ]
        raw = "".join(self.provider.stream_chat(messages))
        return _parse_scenes(raw, brief, n)

    def render(
        self,
        brief: str,
        n: int = 4,
        seconds_per_scene: float = 2.5,
        on_scene: Optional[Callable[[int, Scene], None]] = None,
    ) -> StoryboardResult:
        scenes = self.plan(brief, n)

        out_dir = _output_dir(self.cfg, "storyboards")
        for i, scene in enumerate(scenes):
            result = self.image_gen.generate(scene.prompt, n=1)
            # Move/rename the generated image into the storyboard folder, ordered.
            src = result.paths[0]
            dest = out_dir / f"scene-{i + 1:02d}.png"
            dest.write_bytes(src.read_bytes())
            scene.image = dest
            if on_scene:
                on_scene(i, scene)

        video = self._stitch(scenes, out_dir, seconds_per_scene)
        return StoryboardResult(scenes=scenes, video=video, image_dir=out_dir)

    @staticmethod
    def ffmpeg_available() -> bool:
        return shutil.which("ffmpeg") is not None

    def _stitch(
        self, scenes: List[Scene], out_dir: Path, seconds_per_scene: float
    ) -> Optional[Path]:
        images = [s.image for s in scenes if s.image]
        if not images or not self.ffmpeg_available():
            return None

        # Build a concat demuxer list with a per-image duration.
        list_path = out_dir / "scenes.txt"
        lines = []
        for img in images:
            lines.append(f"file '{img.as_posix()}'")
            lines.append(f"duration {seconds_per_scene}")
        # The concat demuxer needs the last file repeated (no trailing duration).
        lines.append(f"file '{images[-1].as_posix()}'")
        list_path.write_text("\n".join(lines), encoding="utf-8")

        video_path = out_dir / "storyboard.mp4"
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_path),
            "-vf",
            "scale=1024:-2:force_original_aspect_ratio=decrease,"
            "pad=1024:1024:(ow-iw)/2:(oh-ih)/2,format=yuv420p",
            "-r",
            "30",
            str(video_path),
        ]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300
            )
        except (subprocess.TimeoutExpired, OSError):
            return None
        return video_path if proc.returncode == 0 and video_path.exists() else None
