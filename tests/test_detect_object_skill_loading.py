from llm_skill_robot.core.skill_library import SkillLibrary


def test_detect_object_skill_loads():
    library = SkillLibrary()

    assert library.get_skill("detect_object") is not None


def test_detect_object_query_is_required():
    library = SkillLibrary()
    skill = library.get_skill("detect_object")
    parameters = {parameter.name: parameter for parameter in skill.parameters}

    assert parameters["query"].required is True
    assert parameters["query"].type == "string"


def test_detect_object_min_confidence_default_metadata():
    library = SkillLibrary()
    skill = library.get_skill("detect_object")
    parameters = {parameter.name: parameter for parameter in skill.parameters}

    assert parameters["min_confidence"].default == 0.5
    assert parameters["min_confidence"].min_value == 0.0
    assert parameters["min_confidence"].max_value == 1.0


def test_detect_object_backend_method():
    library = SkillLibrary()
    skill = library.get_skill("detect_object")

    assert skill.execution.backend_method == "detect_object"
