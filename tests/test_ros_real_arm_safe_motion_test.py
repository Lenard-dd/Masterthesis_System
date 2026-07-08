import importlib

from llm_skill_robot.ros_real_arm_safe_motion_test import (
    LastMotionPlan,
    validate_execute_confirmation,
    validate_execution_precheck,
    validate_menu_target,
)
from llm_skill_robot.safety.real_arm_safety import load_real_arm_safety


def test_module_imports_without_ros_initialization():
    module = importlib.import_module("llm_skill_robot.ros_real_arm_safe_motion_test")

    assert callable(module.main)


def test_menu_helper_blocks_unknown_target():
    result = validate_menu_target("unknown_target", load_real_arm_safety())

    assert result["allowed"] is False


def test_execution_requires_execute():
    assert validate_execute_confirmation("EXECUTE")["allowed"] is True
    assert validate_execute_confirmation("execute")["allowed"] is False
    assert validate_execute_confirmation("YES")["allowed"] is False


def test_blocked_skill_cannot_execute():
    plan = LastMotionPlan(
        plan_id="plan_1",
        target_name="observe",
        skill_id="approach_grasp",
        summary={"velocity_scale": 0.01, "acceleration_scale": 0.01},
    )

    result = validate_execution_precheck(plan, load_real_arm_safety())

    assert result["allowed"] is False
    assert "blocked" in result["reason"]


def test_safe_home_target_accepted():
    result = validate_menu_target("safe_home", load_real_arm_safety())

    assert result["allowed"] is True
