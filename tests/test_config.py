"""Tests for configuration loading and environment overrides."""

import json

import shellai.config as config_module
from shellai.config import Config


def test_defaults():
    cfg = Config()
    assert cfg.provider == "ollama"
    assert cfg.temperature == 0.7
    assert cfg.openai_api_key == ""


def test_load_from_file(tmp_path, monkeypatch):
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps({"provider": "openai", "model": "gpt-4o"}))
    monkeypatch.setattr(config_module, "CONFIG_PATH", cfg_path)
    # Clear any env that could interfere.
    for var in ("SHELLAI_PROVIDER", "SHELLAI_MODEL", "OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)

    cfg = Config.load()
    assert cfg.provider == "openai"
    assert cfg.model == "gpt-4o"


def test_env_overrides_file(tmp_path, monkeypatch):
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps({"provider": "openai"}))
    monkeypatch.setattr(config_module, "CONFIG_PATH", cfg_path)
    monkeypatch.setenv("SHELLAI_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")

    cfg = Config.load()
    assert cfg.provider == "groq"
    assert cfg.groq_api_key == "gsk_test"


def test_unknown_keys_ignored(tmp_path, monkeypatch):
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps({"provider": "ollama", "bogus": 123}))
    monkeypatch.setattr(config_module, "CONFIG_PATH", cfg_path)
    cfg = Config.load()
    assert cfg.provider == "ollama"
    assert not hasattr(cfg, "bogus")


def test_save_roundtrip(tmp_path, monkeypatch):
    cfg_path = tmp_path / "config.json"
    monkeypatch.setattr(config_module, "CONFIG_PATH", cfg_path)
    monkeypatch.setattr(config_module, "CONFIG_DIR", tmp_path)

    cfg = Config()
    cfg.provider = "anthropic"
    cfg.save()

    data = json.loads(cfg_path.read_text())
    assert data["provider"] == "anthropic"
