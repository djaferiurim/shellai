"""oshell command-line interface.

Usage examples
--------------
    oshell                       # start an interactive chat
    oshell "explain async/await" # one-shot question
    oshell --provider openai --model gpt-4o-mini "summarize this"
    cat error.log | oshell "what is wrong here?"
    oshell models                # list available models
    oshell config set provider openai
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.syntax import Syntax

from . import __version__, history, personas, retrieval
from .agent import CodingAgent
from .config import Config
from .media import ImageGenerator, VideoGenerator
from .media_agent import MediaAgent
from .providers import Message, get_provider
from .storyboard import StoryboardAgent
from .tools import Toolbox

app = typer.Typer(
    add_completion=False,
    help="A fast, beautiful terminal chat for any LLM — OpenAI or local Ollama.",
    no_args_is_help=False,
)
config_app = typer.Typer(help="View and edit configuration.")
app.add_typer(config_app, name="config")
persona_app = typer.Typer(help="Manage personas (system-prompt presets).")
app.add_typer(persona_app, name="persona")

console = Console()

SLASH_HELP = """\
[bold]Slash commands[/bold]
  /exit, /quit   end the chat
  /clear         forget the conversation so far
  /system <txt>  set a new system prompt
  /help          show this help
"""


def _read_stdin() -> str:
    """Return piped stdin text, if any."""
    if not sys.stdin.isatty():
        return sys.stdin.read().strip()
    return ""


def _render_stream(provider, messages: List[Message]) -> str:
    """Stream a response, rendering live Markdown. Returns full text."""
    full = ""
    try:
        with Live(console=console, refresh_per_second=15, vertical_overflow="visible") as live:
            for chunk in provider.stream_chat(messages):
                full += chunk
                live.update(Markdown(full))
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
    return full


def _one_shot(prompt: str, cfg: Config) -> None:
    provider = get_provider(cfg)
    messages: List[Message] = [
        {"role": "system", "content": personas.resolve_system_prompt(cfg)},
        {"role": "user", "content": prompt},
    ]
    _render_stream(provider, messages)


def _interactive(cfg: Config) -> None:
    provider = get_provider(cfg)
    session = history.new_session_path()
    system_prompt = personas.resolve_system_prompt(cfg)
    messages: List[Message] = [{"role": "system", "content": system_prompt}]

    persona_note = f" · persona [green]{cfg.persona}[/green]" if cfg.persona else ""
    console.print(
        Panel.fit(
            f"[bold cyan]oshell[/bold cyan] v{__version__}  "
            f"· provider [green]{cfg.provider}[/green] "
            f"· model [green]{cfg.model}[/green]" + persona_note + "\n"
            "Type [bold]/help[/bold] for commands, [bold]/exit[/bold] to quit.",
            border_style="cyan",
        )
    )

    while True:
        try:
            user = Prompt.ask("[bold blue]you[/bold blue]")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Bye![/dim]")
            break

        if not user.strip():
            continue

        if user.startswith("/"):
            cmd, _, arg = user[1:].partition(" ")
            cmd = cmd.lower()
            if cmd in ("exit", "quit"):
                console.print("[dim]Bye![/dim]")
                break
            if cmd == "clear":
                messages = [{"role": "system", "content": system_prompt}]
                console.print("[dim]Conversation cleared.[/dim]")
                continue
            if cmd == "system":
                system_prompt = arg.strip() or system_prompt
                messages[0] = {"role": "system", "content": system_prompt}
                console.print("[dim]System prompt updated.[/dim]")
                continue
            if cmd == "help":
                console.print(SLASH_HELP)
                continue
            console.print(f"[red]Unknown command:[/red] /{cmd}")
            continue

        messages.append({"role": "user", "content": user})
        history.append(session, {"role": "user", "content": user})

        console.print("[bold green]ai[/bold green]")
        reply = _render_stream(provider, messages)
        if reply:
            messages.append({"role": "assistant", "content": reply})
            history.append(session, {"role": "assistant", "content": reply})


@app.callback(invoke_without_command=True)
def main(
    version: bool = typer.Option(
        False, "--version", "-V", help="Show version and exit."
    ),
) -> None:
    """A fast, beautiful terminal chat for any LLM."""
    if version:
        console.print(f"oshell v{__version__}")
        raise typer.Exit()


@app.command()
def chat(
    prompt: Optional[List[str]] = typer.Argument(
        None, help="Ask a one-shot question. Omit to start interactive chat."
    ),
    provider: Optional[str] = typer.Option(
        None, "--provider", "-p", help="LLM provider: ollama, openai, anthropic, groq, gemini."
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="Model name to use."
    ),
    persona: Optional[str] = typer.Option(
        None, "--persona", help="Named persona/preset (see `ai persona list`)."
    ),
) -> None:
    """Start a chat, or answer a one-shot prompt (the default command)."""
    cfg = Config.load()
    if provider:
        cfg.provider = provider
    if model:
        cfg.model = model
    if persona:
        cfg.persona = persona

    piped = _read_stdin()
    prompt_text = " ".join(prompt) if prompt else ""
    combined = "\n\n".join(p for p in (prompt_text, piped) if p).strip()

    try:
        if combined:
            _one_shot(combined, cfg)
        else:
            _interactive(cfg)
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1)
    except Exception as exc:  # noqa: BLE001 - surface a friendly message
        console.print(f"[red]Request failed:[/red] {exc}")
        raise typer.Exit(code=1)


@app.command()
def models() -> None:
    """List models available from the current provider."""
    cfg = Config.load()
    try:
        provider = get_provider(cfg)
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1)

    found = provider.list_models()
    if not found:
        console.print(
            f"[yellow]No models found for provider '{cfg.provider}'.[/yellow]"
        )
        raise typer.Exit(code=1)
    console.print(f"[bold]Models ({cfg.provider}):[/bold]")
    for name in found:
        marker = " [green](current)[/green]" if name == cfg.model else ""
        console.print(f"  • {name}{marker}")


def _approve(summary: str, detail: str) -> bool:
    """Ask the user to approve a mutating agent action."""
    console.print(f"\n[bold yellow]⚠ {summary}[/bold yellow]")
    if detail:
        lexer = "python" if summary.lower().endswith(".py") else "text"
        if summary.startswith("Run shell command"):
            lexer = "bash"
        console.print(
            Panel(
                Syntax(detail, lexer, theme="ansi_dark", word_wrap=True),
                border_style="yellow",
            )
        )
    return Confirm.ask("[bold]Proceed?[/bold]", default=True)


@app.command()
def agent(
    task: Optional[List[str]] = typer.Argument(
        None, help="What you want the agent to do."
    ),
    provider: Optional[str] = typer.Option(
        None, "--provider", "-p", help="LLM provider: 'ollama' or 'openai'."
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="Model name to use."
    ),
    workdir: Path = typer.Option(
        Path.cwd(), "--dir", "-d", help="Workspace root the agent may touch."
    ),
    yolo: bool = typer.Option(
        False, "--yolo", help="Skip confirmations (auto-approve writes & commands)."
    ),
    interactive: bool = typer.Option(
        False, "--interactive", "-i", help="Keep taking follow-up tasks after each run."
    ),
    max_steps: int = typer.Option(
        25, "--max-steps", help="Maximum reasoning/action steps."
    ),
) -> None:
    """Run the autonomous coding agent on a task."""
    cfg = Config.load()
    if provider:
        cfg.provider = provider
    if model:
        cfg.model = model

    task_text = " ".join(task).strip() if task else ""
    # In interactive mode we can start without a task and prompt in the loop.
    if not task_text and not interactive:
        task_text = Prompt.ask("[bold blue]What should the agent build?[/bold blue]")
        if not task_text.strip():
            console.print("[yellow]No task given.[/yellow]")
            raise typer.Exit(code=1)

    try:
        prov = get_provider(cfg)
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1)

    root = workdir.resolve()
    toolbox = Toolbox(root=root, approve=_approve, auto_approve=yolo)
    coding_agent = CodingAgent(prov, toolbox, max_steps=max_steps)

    console.print(
        Panel.fit(
            f"[bold cyan]oshell Agent[/bold cyan] · {cfg.provider}/{cfg.model}\n"
            f"workspace: [green]{root}[/green]"
            + ("  [red](--yolo: no confirmations)[/red]" if yolo else "")
            + ("  [cyan](interactive)[/cyan]" if interactive else ""),
            border_style="cyan",
        )
    )

    def _run_one(task_str: str) -> None:
        console.print(f"[bold]Task:[/bold] {task_str}\n")
        for event in coding_agent.run(task_str):
            if event.kind == "thought":
                console.print(f"[dim italic]💭 {event.data['text']}[/dim italic]")
            elif event.kind == "action":
                args = event.data["args"]
                detail = args.get("path") or args.get("command") or ""
                console.print(
                    f"[bold magenta]→ {event.data['action']}[/bold magenta] "
                    f"[dim]{detail}[/dim]"
                )
            elif event.kind == "observation":
                color = "green" if event.data["ok"] else "red"
                out = event.data["output"]
                snippet = out if len(out) < 500 else out[:500] + "\n… (truncated)"
                console.print(f"[{color}]{snippet}[/{color}]")
            elif event.kind == "finish":
                console.print(
                    Panel.fit(
                        f"[bold green]✓ {event.data['summary']}[/bold green]",
                        border_style="green",
                    )
                )
            elif event.kind == "error":
                console.print(f"[red]{event.data['message']}[/red]")

    try:
        if not interactive:
            _run_one(task_text)
            return

        # Interactive mode: run the first task (if given), then loop.
        if task_text:
            _run_one(task_text)
        while True:
            try:
                follow = Prompt.ask("\n[bold blue]agent ▸ next task[/bold blue] "
                                    "[dim](/exit to quit)[/dim]")
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]Bye![/dim]")
                break
            if follow.strip().lower() in ("/exit", "/quit", "exit", "quit"):
                console.print("[dim]Bye![/dim]")
                break
            if not follow.strip():
                continue
            console.print()
            _run_one(follow)
    except KeyboardInterrupt:
        console.print("\n[yellow]Agent interrupted.[/yellow]")
    except Exception as exc:  # noqa: BLE001 - friendly surface
        console.print(f"[red]Agent failed:[/red] {exc}")
        raise typer.Exit(code=1)


def _chat_provider_for_media(cfg: Config):
    """Return a chat provider for prompt enhancement (best effort)."""
    try:
        return get_provider(cfg)
    except ValueError:
        return None


@app.command()
def image(
    brief: Optional[List[str]] = typer.Argument(
        None, help="What to draw, e.g. 'a fox in a misty forest'."
    ),
    count: int = typer.Option(1, "--count", "-n", help="How many images."),
    size: Optional[str] = typer.Option(
        None, "--size", "-s", help="Image size, e.g. 1024x1024."
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="Image model (default gpt-image-1)."
    ),
    raw: bool = typer.Option(
        False, "--raw", help="Use your brief verbatim (skip LLM enhancement)."
    ),
) -> None:
    """Generate image(s) from a brief, with LLM prompt-enhancement."""
    cfg = Config.load()
    if size:
        cfg.image_size = size
    if model:
        cfg.image_model = model

    brief_text = " ".join(brief).strip() if brief else ""
    if not brief_text:
        brief_text = Prompt.ask("[bold blue]Describe the image[/bold blue]")
    if not brief_text.strip():
        console.print("[yellow]No description given.[/yellow]")
        raise typer.Exit(code=1)

    try:
        generator = ImageGenerator(cfg)
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1)

    media_agent = MediaAgent(_chat_provider_for_media(cfg)) if not raw else None
    prompt = brief_text
    if media_agent is not None:
        with console.status("[cyan]Enhancing prompt…[/cyan]"):
            prompt = media_agent.enhance(brief_text, "image")
        console.print(f"[dim]Prompt:[/dim] {prompt}")

    try:
        with console.status("[cyan]Generating image…[/cyan]"):
            result = generator.generate(prompt, n=count)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Image generation failed:[/red] {exc}")
        raise typer.Exit(code=1)

    console.print(
        Panel.fit(
            "[bold green]✓ Saved:[/bold green]\n"
            + "\n".join(f"  • {p}" for p in result.paths),
            border_style="green",
        )
    )


@app.command()
def video(
    brief: Optional[List[str]] = typer.Argument(
        None, help="What to film, e.g. 'a drone shot over a neon city'."
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="Replicate video model (e.g. minimax/video-01)."
    ),
    raw: bool = typer.Option(
        False, "--raw", help="Use your brief verbatim (skip LLM enhancement)."
    ),
) -> None:
    """Generate a short video from a brief, with LLM prompt-enhancement."""
    cfg = Config.load()
    if model:
        cfg.video_model = model

    brief_text = " ".join(brief).strip() if brief else ""
    if not brief_text:
        brief_text = Prompt.ask("[bold blue]Describe the video[/bold blue]")
    if not brief_text.strip():
        console.print("[yellow]No description given.[/yellow]")
        raise typer.Exit(code=1)

    try:
        generator = VideoGenerator(cfg)
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1)

    media_agent = MediaAgent(_chat_provider_for_media(cfg)) if not raw else None
    prompt = brief_text
    if media_agent is not None:
        with console.status("[cyan]Enhancing prompt…[/cyan]"):
            prompt = media_agent.enhance(brief_text, "video")
        console.print(f"[dim]Prompt:[/dim] {prompt}")

    console.print(
        f"[dim]Generating with [bold]{cfg.video_model}[/bold] "
        "— this can take a few minutes…[/dim]"
    )
    try:
        with console.status("[cyan]Rendering video…[/cyan]"):
            result = generator.generate(prompt)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Video generation failed:[/red] {exc}")
        raise typer.Exit(code=1)

    console.print(
        Panel.fit(
            "[bold green]✓ Saved:[/bold green]\n"
            + "\n".join(f"  • {p}" for p in result.paths),
            border_style="green",
        )
    )


@app.command()
def storyboard(
    brief: Optional[List[str]] = typer.Argument(
        None, help="The story to tell, e.g. 'a seed growing into a giant tree'."
    ),
    scenes: int = typer.Option(4, "--scenes", "-n", help="Number of scenes."),
    seconds: float = typer.Option(
        2.5, "--seconds", help="Seconds each scene is shown in the video."
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="Image model (default gpt-image-1)."
    ),
    provider: Optional[str] = typer.Option(None, "--provider", "-p"),
) -> None:
    """Plan a multi-scene story, generate an image per scene, stitch into a video."""
    cfg = Config.load()
    if provider:
        cfg.provider = provider
    if model:
        cfg.image_model = model

    brief_text = " ".join(brief).strip() if brief else ""
    if not brief_text:
        brief_text = Prompt.ask("[bold blue]Describe the story[/bold blue]")
    if not brief_text.strip():
        console.print("[yellow]No story given.[/yellow]")
        raise typer.Exit(code=1)

    chat_provider = _chat_provider_for_media(cfg)
    if chat_provider is None:
        console.print("[red]Error:[/red] a chat provider is required to plan scenes.")
        raise typer.Exit(code=1)
    try:
        image_gen = ImageGenerator(cfg)
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1)

    agent = StoryboardAgent(chat_provider, image_gen, cfg)

    if not agent.ffmpeg_available():
        console.print(
            "[yellow]Note:[/yellow] ffmpeg not found — scene images will be saved "
            "but not stitched into a video. Install ffmpeg to enable stitching."
        )

    console.print(
        Panel.fit(
            f"[bold cyan]oshell Storyboard[/bold cyan] · {scenes} scenes\n"
            f"[dim]{brief_text}[/dim]",
            border_style="cyan",
        )
    )

    def _on_scene(i: int, scene) -> None:
        console.print(
            f"[bold magenta]Scene {i + 1}[/bold magenta] "
            f"[dim]{scene.caption or scene.prompt[:60]}[/dim]\n"
            f"  [green]{scene.image}[/green]"
        )

    try:
        console.print("[cyan]Directing and rendering scenes…[/cyan]")
        result = agent.render(
            brief_text, n=scenes, seconds_per_scene=seconds, on_scene=_on_scene
        )
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Storyboard failed:[/red] {exc}")
        raise typer.Exit(code=1)

    lines = [f"  • {s.image}" for s in result.scenes if s.image]
    if result.video:
        lines.insert(0, f"  🎬 [bold]{result.video}[/bold]")
    console.print(
        Panel.fit(
            "[bold green]✓ Storyboard ready:[/bold green]\n" + "\n".join(lines),
            border_style="green",
        )
    )


@app.command()
def do(
    request: Optional[List[str]] = typer.Argument(
        None, help="What you want to do, e.g. 'compress all PNGs in this folder'."
    ),
    provider: Optional[str] = typer.Option(None, "--provider", "-p"),
    model: Optional[str] = typer.Option(None, "--model", "-m"),
) -> None:
    """Translate a plain-English request into a shell command, then run it."""
    import json as _json
    import platform
    import subprocess

    cfg = Config.load()
    if provider:
        cfg.provider = provider
    if model:
        cfg.model = model

    request_text = " ".join(request).strip() if request else ""
    if not request_text:
        request_text = Prompt.ask("[bold blue]What do you want to do?[/bold blue]")
    if not request_text.strip():
        console.print("[yellow]Nothing to do.[/yellow]")
        raise typer.Exit(code=1)

    shell = "PowerShell" if platform.system() == "Windows" else "bash"
    system = (
        f"You translate natural-language requests into a single {shell} command "
        f"for {platform.system()}. Respond ONLY with a JSON object: "
        '{"command": "<the command>", "explanation": "<one short line>"}. '
        "No markdown, no code fences."
    )
    try:
        prov = get_provider(cfg)
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1)

    with console.status("[cyan]Thinking…[/cyan]"):
        raw = "".join(
            prov.stream_chat(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": request_text},
                ]
            )
        ).strip()

    if raw.startswith("```"):
        import re as _re

        raw = _re.sub(r"^```[a-zA-Z]*\n?", "", raw)
        raw = _re.sub(r"\n?```$", "", raw).strip()
    try:
        data = _json.loads(raw)
        command = data["command"]
        explanation = data.get("explanation", "")
    except (ValueError, KeyError):
        console.print(f"[red]Could not parse a command from the model.[/red]\n{raw}")
        raise typer.Exit(code=1)

    console.print(Panel(Syntax(command, "bash", theme="ansi_dark", word_wrap=True),
                        title="Suggested command", border_style="yellow"))
    if explanation:
        console.print(f"[dim]{explanation}[/dim]")

    if not Confirm.ask("[bold]Run it?[/bold]", default=False):
        console.print("[dim]Skipped.[/dim]")
        raise typer.Exit()

    proc = subprocess.run(command, shell=True)
    raise typer.Exit(code=proc.returncode)


@app.command()
def index(
    folder: Path = typer.Argument(..., help="Folder of files to index."),
) -> None:
    """Index a folder so you can chat with your files (`ai ask`)."""
    cfg = Config.load()
    root = folder.resolve()
    if not root.is_dir():
        console.print(f"[red]Not a directory:[/red] {folder}")
        raise typer.Exit(code=1)

    console.print(f"[cyan]Indexing[/cyan] {root} …")
    count = {"n": 0}

    def _tick(_src: str) -> None:
        count["n"] += 1

    try:
        with console.status("[cyan]Embedding chunks…[/cyan]"):
            total = retrieval.build_index(root, cfg, progress=_tick)
    except Exception as exc:  # noqa: BLE001
        console.print(
            f"[red]Indexing failed:[/red] {exc}\n"
            "[dim]Tip: with Ollama, run `ollama pull nomic-embed-text` first.[/dim]"
        )
        raise typer.Exit(code=1)

    console.print(
        Panel.fit(
            f"[bold green]✓ Indexed {total} chunks[/bold green] from {root}",
            border_style="green",
        )
    )


@app.command()
def ask(
    question: Optional[List[str]] = typer.Argument(
        None, help="A question about your indexed files."
    ),
    top_k: int = typer.Option(5, "--top-k", "-k", help="How many chunks to retrieve."),
    provider: Optional[str] = typer.Option(None, "--provider", "-p"),
    model: Optional[str] = typer.Option(None, "--model", "-m"),
) -> None:
    """Answer a question using your indexed files as context (RAG)."""
    cfg = Config.load()
    if provider:
        cfg.provider = provider
    if model:
        cfg.model = model

    question_text = " ".join(question).strip() if question else ""
    if not question_text:
        question_text = Prompt.ask("[bold blue]Ask about your files[/bold blue]")
    if not question_text.strip():
        console.print("[yellow]No question given.[/yellow]")
        raise typer.Exit(code=1)

    try:
        with console.status("[cyan]Searching your files…[/cyan]"):
            hits = retrieval.search(question_text, cfg, top_k=top_k)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Search failed:[/red] {exc}")
        raise typer.Exit(code=1)

    if not hits:
        console.print("[yellow]No relevant content found in the index.[/yellow]")
        raise typer.Exit(code=1)

    context = "\n\n".join(
        f"[Source: {src}]\n{text}" for src, text, _score in hits
    )
    sources = sorted({src for src, _t, _s in hits})

    try:
        prov = get_provider(cfg)
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1)

    messages: List[Message] = [
        {
            "role": "system",
            "content": "Answer the question using ONLY the provided context. "
            "If the answer isn't in the context, say so. Cite sources by name.",
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question_text}",
        },
    ]
    _render_stream(prov, messages)
    console.print("\n[dim]Sources: " + ", ".join(sources) + "[/dim]")


@persona_app.command("list")
def persona_list() -> None:
    """List available personas."""
    cfg = Config.load()
    for name, prompt in sorted(personas.all_personas().items()):
        marker = " [green](active)[/green]" if name == cfg.persona else ""
        preview = prompt if len(prompt) < 70 else prompt[:70] + "…"
        console.print(f"[cyan]{name}[/cyan]{marker}\n  [dim]{preview}[/dim]")


@persona_app.command("use")
def persona_use(name: str) -> None:
    """Set the active persona (persisted to config)."""
    if personas.get(name) is None:
        console.print(f"[red]Unknown persona:[/red] {name}")
        raise typer.Exit(code=1)
    cfg = Config.load()
    cfg.persona = name
    cfg.save()
    console.print(f"[green]Active persona →[/green] {name}")


@persona_app.command("add")
def persona_add(name: str, prompt: str) -> None:
    """Create or update a custom persona."""
    path = personas.add(name, prompt)
    console.print(f"[green]Saved persona[/green] '{name}'  ([dim]{path}[/dim])")


@persona_app.command("remove")
def persona_remove(name: str) -> None:
    """Delete a custom persona."""
    if personas.remove(name):
        console.print(f"[green]Removed[/green] '{name}'")
    else:
        console.print(f"[yellow]No custom persona named[/yellow] '{name}'")


@persona_app.command("clear")
def persona_clear() -> None:
    """Stop using any persona (revert to the default system prompt)."""
    cfg = Config.load()
    cfg.persona = ""
    cfg.save()
    console.print("[green]Persona cleared.[/green]")


@config_app.command("show")
def config_show() -> None:
    """Print the current configuration."""
    cfg = Config.load()
    from dataclasses import asdict

    for key, value in asdict(cfg).items():
        if key.endswith(("api_key", "api_token")) and value:
            value = value[:6] + "…" + value[-4:]
        console.print(f"[cyan]{key}[/cyan] = {value}")


@config_app.command("set")
def config_set(key: str, value: str) -> None:
    """Set a config value, e.g. `oshell config set provider openai`."""
    cfg = Config.load()
    if not hasattr(cfg, key):
        console.print(f"[red]Unknown key:[/red] {key}")
        raise typer.Exit(code=1)

    current = getattr(cfg, key)
    if isinstance(current, float):
        setattr(cfg, key, float(value))
    else:
        setattr(cfg, key, value)

    path = cfg.save()
    console.print(f"[green]Saved[/green] {key} → {value}  ([dim]{path}[/dim])")


@config_app.command("path")
def config_path() -> None:
    """Print the path to the config file."""
    from .config import CONFIG_PATH

    console.print(str(CONFIG_PATH))


# Subcommands that should NOT be treated as a chat prompt.
_SUBCOMMANDS = {"chat", "models", "config", "agent", "image", "video",
                "storyboard", "persona", "do", "index", "ask"}
_ROOT_FLAGS = {"--version", "-V", "--help"}


def main_entry() -> None:
    """Console-script entry point.

    Makes ``chat`` the default command so ``ai "question"`` works while
    real subcommands (``models``, ``config``) and root flags still route
    correctly.
    """
    argv = sys.argv[1:]
    if not argv:
        argv = ["chat"]
    elif argv[0] not in _SUBCOMMANDS and argv[0] not in _ROOT_FLAGS:
        argv = ["chat", *argv]
    sys.argv = [sys.argv[0], *argv]
    app()


if __name__ == "__main__":
    main_entry()
