# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-06-11

### Added
- **Chat** — streaming terminal chat with live Markdown rendering, history, slash
  commands, and stdin pipe support.
- **Coding agent** — sandboxed ReAct agent (`ai agent`) with read/write/list/run
  tools, approval prompts, and `--yolo` auto-approve.
- **Interactive agent mode** (`ai agent -i`) — keeps conversation state across
  tasks so follow-ups remember earlier work.
- **Media generation** — `ai image` and `ai video` with an LLM prompt-enhancing
  media agent.
- **Storyboard mode** (`ai storyboard`) — plans scenes, renders an image per
  scene, and stitches them into an MP4 via ffmpeg.
- **Providers** — Ollama, OpenAI, Groq, Gemini, and Anthropic.
- **Personas** — built-in presets plus custom user personas.
- **Natural-language shell** (`ai do`) — turns plain English into shell commands
  with confirmation.
- **Local RAG** — `ai index` / `ai ask` over your files with pure-Python cosine
  search.
- Test suite (pytest), GitHub Actions CI matrix, PyPI publish workflow, demo
  tape, and contributing guide.

[Unreleased]: https://github.com/djaferiurim/aishell/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/djaferiurim/aishell/releases/tag/v0.1.0
