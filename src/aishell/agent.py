"""The AiShell coding agent.

A small but capable ReAct-style loop: the model is asked to reply with a
single JSON object describing one action at a time. We execute the action,
feed the result back as an observation, and repeat until the model calls
``finish`` (or we hit the step limit).

This JSON protocol works with *any* chat model — it does not rely on
provider-specific function/tool calling, so the same agent runs on local
Ollama models and on OpenAI alike.
"""

from __future__ import annotations

import json
import re
from typing import Iterator, List

from .providers import Message, Provider
from .tools import TOOL_REFERENCE, Toolbox

MAX_STEPS = 25

SYSTEM_PROMPT = f"""\
You are AiShell Agent, an autonomous coding assistant working inside a user's \
project directory. You accomplish the user's task by taking ONE action at a \
time using the available tools.

{TOOL_REFERENCE}

RESPONSE FORMAT — respond with a SINGLE JSON object and nothing else:
{{"thought": "brief reasoning about the next step",
  "action": "<one tool name above>",
  "args": {{ ...arguments for that tool... }}}}

Rules:
- Output ONLY the JSON object. No markdown, no code fences, no extra prose.
- Take one action per response. Wait for the observation before continuing.
- Explore with list_dir / read_file before editing unfamiliar files.
- When writing a file, include its COMPLETE intended contents in "content".
- When the task is done, use the "finish" action with a short summary.
- Keep commands cross-platform where possible.
"""


def _extract_json(text: str) -> dict:
    """Best-effort parse of a JSON object from a model response."""
    text = text.strip()
    # Strip code fences if the model added them anyway.
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fall back to the first balanced {...} block.
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model response.")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise ValueError("Unbalanced JSON in model response.")


class AgentEvent:
    """A step in the agent's run, yielded for live display."""

    def __init__(self, kind: str, **data):
        self.kind = kind  # "thought" | "action" | "observation" | "finish" | "error"
        self.data = data


class CodingAgent:
    """Drives the think-act-observe loop."""

    def __init__(self, provider: Provider, toolbox: Toolbox, max_steps: int = MAX_STEPS):
        self.provider = provider
        self.toolbox = toolbox
        self.max_steps = max_steps
        # Conversation state persists across tasks so the agent can take
        # follow-up instructions in interactive mode while remembering what
        # it already did.
        self.messages: List[Message] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    def _complete(self, messages: List[Message]) -> str:
        """Collect a full (non-streamed) completion from the provider."""
        return "".join(self.provider.stream_chat(messages))

    def run(self, task: str) -> Iterator[AgentEvent]:
        """Run the agent on ``task``, yielding events as it progresses.

        Conversation history is retained between calls, so calling ``run``
        again with a follow-up continues the same session.
        """
        self.messages.append({"role": "user", "content": f"Task: {task}"})

        for _ in range(self.max_steps):
            raw = self._complete(self.messages)
            try:
                step = _extract_json(raw)
            except ValueError as exc:
                yield AgentEvent("error", message=f"{exc}\nModel said: {raw[:300]}")
                # Nudge the model back to the protocol.
                self.messages.append({"role": "assistant", "content": raw})
                self.messages.append(
                    {
                        "role": "user",
                        "content": "Your last message was not valid JSON. "
                        "Respond with exactly one JSON object as instructed.",
                    }
                )
                continue

            thought = step.get("thought", "")
            action = step.get("action", "")
            args = step.get("args", {}) or {}

            if thought:
                yield AgentEvent("thought", text=thought)

            self.messages.append({"role": "assistant", "content": json.dumps(step)})

            if action == "finish":
                yield AgentEvent("finish", summary=args.get("summary", "Done."))
                return

            yield AgentEvent("action", action=action, args=args)
            result = self.toolbox.dispatch(action, args)
            yield AgentEvent("observation", ok=result.ok, output=result.output)

            self.messages.append(
                {"role": "user", "content": f"Observation: {result.render()}"}
            )

        yield AgentEvent(
            "error", message=f"Reached the step limit ({self.max_steps}). Stopping."
        )
