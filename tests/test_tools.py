"""Tests for the agent toolbox: path sandboxing and file operations."""

import pytest

from oshell.tools import Toolbox


def _approve_all(summary, detail):
    return True


def _deny_all(summary, detail):
    return False


def test_write_and_read_roundtrip(tmp_workspace):
    box = Toolbox(tmp_workspace, _approve_all)
    res = box.write_file("hello.txt", "hi there")
    assert res.ok
    read = box.read_file("hello.txt")
    assert read.ok
    assert read.output == "hi there"


def test_write_requires_approval(tmp_workspace):
    box = Toolbox(tmp_workspace, _deny_all)
    res = box.write_file("nope.txt", "data")
    assert not res.ok
    assert not (tmp_workspace / "nope.txt").exists()


def test_auto_approve_skips_callback(tmp_workspace):
    box = Toolbox(tmp_workspace, _deny_all, auto_approve=True)
    res = box.write_file("yes.txt", "data")
    assert res.ok
    assert (tmp_workspace / "yes.txt").read_text() == "data"


def test_path_escape_is_blocked(tmp_workspace):
    box = Toolbox(tmp_workspace, _approve_all)
    res = box.write_file("../escape.txt", "x")
    assert not res.ok
    assert "outside the workspace" in res.output
    assert not (tmp_workspace.parent / "escape.txt").exists()


def test_read_missing_file(tmp_workspace):
    box = Toolbox(tmp_workspace, _approve_all)
    res = box.read_file("ghost.txt")
    assert not res.ok


def test_list_dir(tmp_workspace):
    (tmp_workspace / "a.txt").write_text("1")
    (tmp_workspace / "sub").mkdir()
    box = Toolbox(tmp_workspace, _approve_all)
    res = box.list_dir(".")
    assert res.ok
    assert "a.txt" in res.output
    assert "sub/" in res.output


def test_dispatch_unknown_action(tmp_workspace):
    box = Toolbox(tmp_workspace, _approve_all)
    res = box.dispatch("frobnicate", {})
    assert not res.ok
    assert "Unknown action" in res.output


def test_write_creates_parent_dirs(tmp_workspace):
    box = Toolbox(tmp_workspace, _approve_all)
    res = box.write_file("nested/deep/file.txt", "ok")
    assert res.ok
    assert (tmp_workspace / "nested" / "deep" / "file.txt").exists()
