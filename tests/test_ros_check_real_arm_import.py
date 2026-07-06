import importlib


def test_module_imports_without_ros_initialization():
    module = importlib.import_module("llm_skill_robot.ros_check_real_arm")

    assert callable(module.main)
