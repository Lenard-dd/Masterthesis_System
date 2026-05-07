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
    parameters = {parameter.name: parameter for parameter in skill.parameters}
    assert "target_name" in parameters
    assert parameters["target_name"].required is True
    assert parameters["velocity_scale"].default == 0.03
    assert parameters["velocity_scale"].max_value == 0.1
    assert parameters["acceleration_scale"].default == 0.03
    assert parameters["acceleration_scale"].max_value == 0.1


def test_get_unknown_skill():
    """Test getting a non-existent skill."""
    library = SkillLibrary()
    skill = library.get_skill("unknown_skill")
    assert skill is None
