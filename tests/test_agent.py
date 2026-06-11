"""Tests for the agent's JSON extraction helper."""

import json

import pytest

from oshell.agent import CodingAgent, _extract_json
from oshell.tools import Toolbox


def test_plain_json():
    out = _extract_json('{"action": "finish", "args": {}}')
    assert out["action"] == "finish"


def test_json_in_code_fence():
    raw = '```json\n{"action": "read_file", "args": {"path": "a.py"}}\n```'
    out = _extract_json(raw)
    assert out["action"] == "read_file"
    assert out["args"]["path"] == "a.py"


def test_json_with_surrounding_prose():
    raw = 'Sure! Here is my step:\n{"thought": "x", "action": "list_dir", "args": {}} done'
    out = _extract_json(raw)
    assert out["action"] == "list_dir"
    assert out["thought"] == "x"


def test_nested_braces():
    raw = '{"action": "write_file", "args": {"path": "a", "content": "{ }"}}'
    out = _extract_json(raw)
    assert out["args"]["content"] == "{ }"


def test_no_json_raises():
    with pytest.raises(ValueError):
        _extract_json("there is no json here")


class _ScriptedProvider:
    """Returns queued responses one per stream_chat call."""

    name = "scripted"

    def __init__(self, responses):
        self._responses = list(responses)

    def stream_chat(self, messages):
        yield self._responses.pop(0)

    def list_models(self):
        return []


def test_agent_finishes_and_writes(tmp_path):
    provider = _ScriptedProvider(
        [
            json.dumps(
                {
                    "thought": "create the file",
                    "action": "write_file",
                    "args": {"path": "out.txt", "content": "hello"},
                }
            ),
            json.dumps({"action": "finish", "args": {"summary": "done"}}),
        ]
    )
    box = Toolbox(tmp_path, lambda s, d: True, auto_approve=True)
    agent = CodingAgent(provider, box, max_steps=5)

    kinds = [ev.kind for ev in agent.run("make a file")]
    assert "finish" in kinds
    assert (tmp_path / "out.txt").read_text() == "hello"


def test_agent_retains_state_across_runs(tmp_path):
    provider = _ScriptedProvider(
        [json.dumps({"action": "finish", "args": {"summary": "one"}})]
    )
    box = Toolbox(tmp_path, lambda s, d: True, auto_approve=True)
    agent = CodingAgent(provider, box, max_steps=5)

    list(agent.run("first task"))
    before = len(agent.messages)

    provider._responses.append(
        json.dumps({"action": "finish", "args": {"summary": "two"}})
    )
    list(agent.run("second task"))
    # The second run appended to the same conversation rather than resetting.
    assert len(agent.messages) > before
    assert any("second task" in m["content"] for m in agent.messages)
