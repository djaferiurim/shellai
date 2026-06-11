"""Tests for personas (built-in + custom presets)."""

import oshell.personas as personas


def test_builtin_personas_present():
    all_p = personas.all_personas()
    assert "reviewer" in all_p
    assert "teacher" in all_p
    assert personas.get("reviewer")


def test_get_unknown_returns_none():
    assert personas.get("does-not-exist") is None


def test_add_and_get_custom(tmp_path, monkeypatch):
    path = tmp_path / "personas.json"
    monkeypatch.setattr(personas, "PERSONA_PATH", path)
    monkeypatch.setattr(personas, "CONFIG_DIR", tmp_path)

    personas.add("legal", "You are a contracts expert.")
    assert personas.get("legal") == "You are a contracts expert."
    assert "legal" in personas.all_personas()


def test_custom_overrides_builtin(tmp_path, monkeypatch):
    path = tmp_path / "personas.json"
    monkeypatch.setattr(personas, "PERSONA_PATH", path)
    monkeypatch.setattr(personas, "CONFIG_DIR", tmp_path)

    personas.add("reviewer", "custom override")
    assert personas.get("reviewer") == "custom override"


def test_remove_custom(tmp_path, monkeypatch):
    path = tmp_path / "personas.json"
    monkeypatch.setattr(personas, "PERSONA_PATH", path)
    monkeypatch.setattr(personas, "CONFIG_DIR", tmp_path)

    personas.add("temp", "x")
    assert personas.remove("temp") is True
    assert personas.remove("temp") is False


class _Cfg:
    persona = ""
    system_prompt = "base prompt"


def test_resolve_system_prompt_default():
    assert personas.resolve_system_prompt(_Cfg()) == "base prompt"


def test_resolve_system_prompt_with_persona():
    cfg = _Cfg()
    cfg.persona = "concise"
    assert personas.resolve_system_prompt(cfg) == personas.get("concise")


def test_resolve_system_prompt_unknown_persona_falls_back():
    cfg = _Cfg()
    cfg.persona = "nope"
    assert personas.resolve_system_prompt(cfg) == "base prompt"
