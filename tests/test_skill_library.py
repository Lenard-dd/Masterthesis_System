import pytest
from pathlib import Path
from llm_skill_robot.core.skill_library import SkillLibrary


def test_skill_library_load():
    """Test loading skills from directory."""
    library = SkillLibrary()
    skills = library.list_skills()
    assert len(skills) > 0
    assert "move_to_named_target" in skills


def test_get_skill():
    """Test getting a specific skill."""
    library = SkillLibrary()
    skill = library.get_skill("move_to_named_target")
    assert skill is not None
    assert skill.name == "Move to Named Target"
    assert len(skill.parameters) == 1
    assert skill.parameters[0].name == "target_name"


def test_get_unknown_skill():
    """Test getting a non-existent skill."""
    library = SkillLibrary()
    skill = library.get_skill("unknown_skill")
    assert skill is None