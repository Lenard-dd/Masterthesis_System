"""Workstation configuration overrides must be explicit and fail closed."""

import pytest

from llm_skill_robot.utils import get_config_dir, get_skills_dir


def test_override_selects_config_without_replacing_skill_library(monkeypatch, tmp_path):
    original_skills = get_skills_dir()
    monkeypatch.setenv("LLM_SKILL_CONFIG_DIR", str(tmp_path))
    assert get_config_dir() == tmp_path
    assert get_skills_dir() == original_skills


def test_invalid_override_does_not_fall_back_to_lab_config(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_SKILL_CONFIG_DIR", str(tmp_path / "missing"))
    with pytest.raises(FileNotFoundError, match="LLM_SKILL_CONFIG_DIR"):
        get_config_dir()
