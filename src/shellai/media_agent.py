"""The media agent: turns a short brief into rich media.

It uses the chat LLM to *enhance* a user's brief into a detailed, model-ready
prompt (composition, lighting, style, camera, mood…), then hands that prompt
to the image or video generator. This is the "sophisticated" part — you write
"a fox in a forest" and get a cinematic, well-specified prompt.
"""

from __future__ import annotations

from .media import ImageGenerator, MediaResult, VideoGenerator
from .providers import Message, Provider

_IMAGE_ENHANCE = """\
You are a prompt engineer for text-to-image models. Rewrite the user's brief \
into ONE vivid, detailed image prompt. Include subject, composition, lighting, \
color palette, art style, and mood. Keep it under 80 words. Output ONLY the \
prompt text — no quotes, no preamble, no explanation."""

_VIDEO_ENHANCE = """\
You are a prompt engineer for text-to-video models. Rewrite the user's brief \
into ONE detailed video prompt describing the scene, subject motion, camera \
movement, lighting, and mood. Keep it under 80 words. Output ONLY the prompt \
text — no quotes, no preamble, no explanation."""


class MediaAgent:
    """Enhances a brief with the LLM, then generates media."""

    def __init__(self, provider: Provider) -> None:
        self.provider = provider

    def enhance(self, brief: str, kind: str) -> str:
        """Expand ``brief`` into a detailed prompt for the given media kind."""
        system = _IMAGE_ENHANCE if kind == "image" else _VIDEO_ENHANCE
        messages: list[Message] = [
            {"role": "system", "content": system},
            {"role": "user", "content": brief},
        ]
        enhanced = "".join(self.provider.stream_chat(messages)).strip()
        # Strip stray surrounding quotes some models add.
        enhanced = enhanced.strip('"').strip()
        return enhanced or brief

    def make_image(
        self, brief: str, generator: ImageGenerator, n: int = 1, enhance: bool = True
    ) -> MediaResult:
        prompt = self.enhance(brief, "image") if enhance else brief
        return generator.generate(prompt, n=n)

    def make_video(
        self, brief: str, generator: VideoGenerator, enhance: bool = True
    ) -> MediaResult:
        prompt = self.enhance(brief, "video") if enhance else brief
        return generator.generate(prompt)
