"""Render a polished terminal-session demo GIF for AiShell.

This is a self-contained renderer (Pillow only) used to produce ``demo.gif``
for the README. It animates typed commands and representative AiShell output
in a Catppuccin-Mocha terminal — no headless browser or recorder required.

Usage:
    python tools/make_demo.py [output_path]
"""
from __future__ import annotations

import os
import sys
from copy import deepcopy

from PIL import Image, ImageDraw, ImageFont

# --- theme (Catppuccin Mocha) ---------------------------------------------
BG = (30, 30, 46)
FG = (205, 214, 244)
COMMENT = (108, 112, 134)
BLUE = (137, 180, 250)
GREEN = (166, 227, 161)
CYAN = (148, 226, 213)
YELLOW = (249, 226, 175)
PINK = (245, 194, 231)
MAUVE = (203, 166, 247)
RED = (243, 139, 168)
SUBTLE = (147, 153, 178)

WIDTH, HEIGHT = 1180, 680
PAD = 24
FONT_SIZE = 20
LINE_H = 28
MAX_LINES = (HEIGHT - 2 * PAD) // LINE_H

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\CascadiaCode.ttf",
    r"C:\Windows\Fonts\CascadiaMono.ttf",
    r"C:\Windows\Fonts\consola.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
]


def load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


FONT = load_font(FONT_SIZE)

# A line is a list of (text, color) segments.
Segment = tuple
Line = list


def prompt_segs() -> Line:
    return [("PS ", SUBTLE), ("~/aishell", BLUE), ("> ", GREEN)]


class Term:
    def __init__(self) -> None:
        self.lines: list[Line] = []
        self.frames: list[list[Line]] = []

    def _emit(self, active: Line | None = None) -> None:
        screen = list(self.lines)
        if active is not None:
            screen.append(active)
        # keep only the last MAX_LINES visible
        if len(screen) > MAX_LINES:
            screen = screen[-MAX_LINES:]
        self.frames.append(deepcopy(screen))

    def type_command(self, cmd: str, step: int = 2, hold: int = 8) -> None:
        base = prompt_segs()
        for i in range(0, len(cmd) + 1, step):
            self._emit(base + [(cmd[:i], FG)])
        full = base + [(cmd, FG)]
        for _ in range(hold):
            self._emit(full)
        self.lines.append(base + [(cmd, FG)])

    def print_lines(self, lines: list[Line], per_line: int = 2, end_hold: int = 10) -> None:
        for line in lines:
            self.lines.append(line)
            for _ in range(per_line):
                self._emit()
        for _ in range(end_hold):
            self._emit()

    def blank(self, n: int = 1) -> None:
        for _ in range(n):
            self.lines.append([("", FG)])


def render_frame(screen: list[Line]) -> Image.Image:
    img = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(img)
    # title bar dots
    for i, col in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        draw.ellipse([PAD + i * 22, 14, PAD + i * 22 + 12, 26], fill=col)
    y = PAD + 18
    for line in screen:
        x = PAD
        for text, color in line:
            draw.text((x, y), text, font=FONT, fill=color)
            x += draw.textlength(text, font=FONT)
        y += LINE_H
    return img


def build() -> Term:
    t = Term()

    # intro comment
    t.lines.append([("# AiShell - chat with any LLM from your terminal", COMMENT)])
    t._emit()

    # 1) one-shot question
    t.type_command('ai "explain monads like I am five"')
    t.print_lines([
        [("A monad is like a ", FG), ("lunchbox", YELLOW), (".", FG)],
        [("You put a sandwich in, and there are rules for how to", FG)],
        [("open it, add chips, and close it again - so nothing", FG)],
        [("spills. The box keeps everything tidy.", FG)],
    ])
    t.blank()

    # 2) persona
    t.type_command('ai --persona pirate "what is git?"')
    t.print_lines([
        [("Arrr! ", PINK), ("Git be the chest where ye stow yer code,", FG)],
        [("matey. Every ", FG), ("commit", CYAN), (" be a snapshot buried on", FG)],
        [("the map, so ye can always sail back to calmer seas.", FG)],
    ])
    t.blank()

    # 3) the coding agent
    t.type_command('ai agent --yolo "create hello.py and run it"')
    t.print_lines([
        [("● ", MAUVE), ("thought  ", MAUVE), ("I'll write hello.py, then execute it.", SUBTLE)],
        [("→ ", BLUE), ("write_file ", BLUE), ("hello.py", FG)],
        [("  ✓ wrote 1 file", GREEN)],
        [("→ ", BLUE), ("run_command ", BLUE), ("python hello.py", FG)],
        [("  Hello from AiShell!", FG)],
        [("✓ ", GREEN), ("done  ", GREEN), ("Created and ran hello.py.", FG)],
    ], per_line=3)
    t.blank()

    # 4) local RAG
    t.type_command("ai index README.md")
    t.print_lines([
        [("Indexed ", FG), ("README.md", CYAN), (" - 12 chunks embedded.", FG)],
    ])
    t.type_command('ai ask "which providers does AiShell support?"')
    t.print_lines([
        [("AiShell supports ", FG), ("Ollama", YELLOW), (", ", FG), ("OpenAI", YELLOW),
         (", ", FG), ("Groq", YELLOW), (",", FG)],
        [("Gemini", YELLOW), (", and ", FG), ("Anthropic", YELLOW), (".", FG)],
        [("sources: ", SUBTLE), ("README.md", BLUE)],
    ], end_hold=22)

    return t


def main() -> None:
    out = sys.argv[1] if len(sys.argv) > 1 else "demo.gif"
    term = build()
    print(f"rendering {len(term.frames)} frames...")
    images = [render_frame(s) for s in term.frames]
    # quantize for a smaller palette-based GIF
    images = [im.quantize(colors=64, method=Image.MEDIANCUT) for im in images]
    images[0].save(
        out,
        save_all=True,
        append_images=images[1:],
        duration=70,
        loop=0,
        optimize=True,
        disposal=2,
    )
    size_kb = round(os.path.getsize(out) / 1024)
    print(f"wrote {out} ({size_kb} KB)")


if __name__ == "__main__":
    main()
