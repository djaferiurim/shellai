"""Tests for storyboard scene parsing and agent conversation state."""

from shellai.storyboard import Scene, _parse_scenes


def test_parse_clean_json():
    raw = '[{"prompt": "a sunrise", "caption": "Dawn"}, {"prompt": "a sunset", "caption": "Dusk"}]'
    scenes = _parse_scenes(raw, "fallback", 2)
    assert len(scenes) == 2
    assert scenes[0].prompt == "a sunrise"
    assert scenes[0].caption == "Dawn"


def test_parse_json_in_code_fence():
    raw = '```json\n[{"prompt": "x"}]\n```'
    scenes = _parse_scenes(raw, "fallback", 1)
    assert len(scenes) == 1
    assert scenes[0].prompt == "x"


def test_parse_json_with_prose():
    raw = 'Here are the scenes:\n[{"prompt": "a", "caption": "c"}]\nEnjoy!'
    scenes = _parse_scenes(raw, "fallback", 1)
    assert scenes[0].prompt == "a"


def test_parse_falls_back_when_invalid():
    scenes = _parse_scenes("not json at all", "a tree growing", 3)
    assert len(scenes) == 3
    assert all("a tree growing" in s.prompt for s in scenes)


def test_scene_dataclass_defaults():
    s = Scene(prompt="p", caption="c")
    assert s.image is None
