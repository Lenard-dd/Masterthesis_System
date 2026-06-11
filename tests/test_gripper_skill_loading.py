from llm_skill_robot.core.skill_library import SkillLibrary


def test_gripper_skills_load():
    library = SkillLibrary()

    for skill_id in (
        "open_gripper",
        "close_gripper",
        "set_gripper_width",
        "get_gripper_state",
    ):
        assert library.get_skill(skill_id) is not None


def test_set_gripper_width_requires_width_parameter():
    library = SkillLibrary()
    skill = library.get_skill("set_gripper_width")
    parameters = {parameter.name: parameter for parameter in skill.parameters}

    assert parameters["width_m"].required is True
    assert parameters["width_m"].type == "float"


def test_gripper_width_max_metadata():
    library = SkillLibrary()
    skill = library.get_skill("set_gripper_width")
    parameters = {parameter.name: parameter for parameter in skill.parameters}

    assert parameters["width_m"].max_value == 0.14
