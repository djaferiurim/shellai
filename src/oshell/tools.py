"""Tools the coding agent can invoke.

Every tool is sandboxed to a *workspace root*: paths are resolved and must
stay inside that root, so the agent cannot read or clobber files elsewhere
on the machine. Mutating tools (``write_file``, ``run_command``) go through
an approval callback so the user stays in control.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict

# A callback the agent uses to ask the user before a risky action.
# Receives a human-readable description and returns True to proceed.
ApprovalFn = Callable[[str, str], bool]

MAX_READ_BYTES = 100_000
COMMAND_TIMEOUT = 120  # seconds


@dataclass
class ToolResult:
    """Outcome of running a tool, fed back to the model as an observation."""

    ok: bool
    output: str

    def render(self) -> str:
        status = "OK" if self.ok else "ERROR"
        return f"[{status}] {self.output}"


class Toolbox:
    """Holds the workspace root and exposes the agent's tools."""

    def __init__(self, root: Path, approve: ApprovalFn, auto_approve: bool = False):
        self.root = root.resolve()
        self.approve = approve
        self.auto_approve = auto_approve

    # ----- path safety ---------------------------------------------------
    def _resolve(self, rel: str) -> Path:
        """Resolve ``rel`` inside the workspace, rejecting escapes."""
        target = (self.root / rel).resolve()
        if target != self.root and self.root not in target.parents:
            raise ValueError(
                f"Path '{rel}' is outside the workspace root and is not allowed."
            )
        return target

    def _confirm(self, summary: str, detail: str) -> bool:
        if self.auto_approve:
            return True
        return self.approve(summary, detail)

    # ----- tools ---------------------------------------------------------
    def read_file(self, path: str) -> ToolResult:
        try:
            target = self._resolve(path)
        except ValueError as exc:
            return ToolResult(False, str(exc))
        if not target.is_file():
            return ToolResult(False, f"File not found: {path}")
        data = target.read_bytes()[:MAX_READ_BYTES]
        text = data.decode("utf-8", errors="replace")
        return ToolResult(True, text)

    def write_file(self, path: str, content: str) -> ToolResult:
        try:
            target = self._resolve(path)
        except ValueError as exc:
            return ToolResult(False, str(exc))

        action = "Overwrite" if target.exists() else "Create"
        preview = content if len(content) < 800 else content[:800] + "\n… (truncated)"
        if not self._confirm(
            f"{action} file: {path}",
            preview,
        ):
            return ToolResult(False, f"User declined to write '{path}'.")

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return ToolResult(True, f"Wrote {len(content)} chars to {path}.")

    def list_dir(self, path: str = ".") -> ToolResult:
        try:
            target = self._resolve(path)
        except ValueError as exc:
            return ToolResult(False, str(exc))
        if not target.is_dir():
            return ToolResult(False, f"Not a directory: {path}")
        entries = []
        for child in sorted(target.iterdir()):
            mark = "/" if child.is_dir() else ""
            entries.append(child.name + mark)
        listing = "\n".join(entries) if entries else "(empty)"
        return ToolResult(True, listing)

    def run_command(self, command: str) -> ToolResult:
        if not self._confirm("Run shell command", command):
            return ToolResult(False, f"User declined to run: {command}")
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=str(self.root),
                capture_output=True,
                text=True,
                timeout=COMMAND_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(False, f"Command timed out after {COMMAND_TIMEOUT}s.")
        out = (proc.stdout or "") + (proc.stderr or "")
        out = out.strip() or "(no output)"
        if len(out) > MAX_READ_BYTES:
            out = out[:MAX_READ_BYTES] + "\n… (truncated)"
        return ToolResult(proc.returncode == 0, f"exit={proc.returncode}\n{out}")

    # ----- dispatch ------------------------------------------------------
    def dispatch(self, action: str, args: Dict) -> ToolResult:
        """Run a named tool with the given args dict."""
        handlers = {
            "read_file": lambda a: self.read_file(a.get("path", "")),
            "write_file": lambda a: self.write_file(
                a.get("path", ""), a.get("content", "")
            ),
            "list_dir": lambda a: self.list_dir(a.get("path", ".")),
            "run_command": lambda a: self.run_command(a.get("command", "")),
        }
        handler = handlers.get(action)
        if handler is None:
            return ToolResult(False, f"Unknown action: {action!r}")
        try:
            return handler(args or {})
        except Exception as exc:  # noqa: BLE001 - report tool failures to the model
            return ToolResult(False, f"Tool raised: {exc}")


TOOL_REFERENCE = """\
Available tools (use exactly these action names):
- read_file   args: {"path": "relative/path"}            -> returns file contents
- write_file  args: {"path": "relative/path", "content": "..."} -> creates/overwrites a file
- list_dir    args: {"path": "relative/dir"}             -> lists directory entries
- run_command args: {"command": "shell command"}         -> runs a command in the workspace
- finish      args: {"summary": "what you accomplished"} -> ends the task
"""
