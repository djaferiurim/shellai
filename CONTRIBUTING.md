# Contributing to AiShell

Thanks for your interest in making AiShell better! 🐚

## Getting started

```bash
git clone https://github.com/djaferiurim/aishell
cd aishell
pip install -e ".[dev]"
pytest          # make sure the suite is green
```

## Project layout

```
src/aishell/
  cli.py          # Typer commands & entry point
  config.py       # config loading / persistence
  providers.py    # LLM backends (OpenAI, Ollama, Anthropic, Groq, Gemini)
  agent.py        # autonomous coding agent (ReAct loop)
  tools.py        # sandboxed file/command tools for the agent
  media.py        # image (OpenAI) + video (Replicate) generation
  media_agent.py  # LLM prompt-enhancement for media
  storyboard.py   # multi-scene story → images → stitched video
  retrieval.py    # local RAG (index + search)
  personas.py     # system-prompt presets
tests/            # pytest suite
```

## How to contribute

1. **Open an issue first** for anything non-trivial so we can align on the approach.
2. Create a branch: `git checkout -b feature/my-thing`.
3. Make your change with a focused commit history.
4. Add or update tests in `tests/`.
5. Run `pytest` and make sure everything passes.
6. Open a pull request describing the *why*, not just the *what*.

## Guidelines

- Keep dependencies minimal — AiShell's appeal is being lightweight.
- Match the existing style (type hints, short docstrings, `rich` for output).
- New network calls should fail gracefully with a clear, actionable message.
- Anything that writes files or runs commands must respect the approval flow.

## Good first issues

- 🎨 Add a new built-in **persona** preset.
- 🔌 Add another **provider adapter** (e.g. Mistral, Cohere, OpenRouter).
- 🧰 Add a new **agent tool** (e.g. `apply_patch`, `search_files`).
- 💬 Add a new **slash command** to the interactive chat.
- 🧪 Increase **test coverage** for `cli.py` helpers.
- 📝 Improve docs, examples, or the demo tape.

## Code of conduct

Be kind and constructive. We're all here to build something fun and useful.
