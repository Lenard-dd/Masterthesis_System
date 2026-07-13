import importlib

import pytest

from llm_skill_robot.ros_real_gripper_test import (
    assert_real_gripper_test_allowed,
    build_command_preview,
    close_command_preview,
    confirmation_is_yes,
)
from llm_skill_robot.safety.hardware_mode import load_hardware_mode


def test_module_imports_without_ros_initialization():
    module = importlib.import_module("llm_skill_robot.ros_real_gripper_test")
    assert callable(module.main)


@pytest.mark.parametrize("response", ["yes", "Yes", "YES ", " YES", "Y", "", "NO"])
def test_confirmation_helper_rejects_everything_except_exact_yes(response):
    assert confirmation_is_yes(response) is False


def test_confirmation_helper_accepts_exact_yes():
    assert confirmation_is_yes("YES") is True


def test_gripper_only_test_is_allowed_with_full_supervised_arm_enabled():
    assert assert_real_gripper_test_allowed(load_hardware_mode()) is None


def test_command_preview_contains_width_speed_and_force():
    preview = build_command_preview(0.07, 0.02, 10.0)

    assert preview == {
        "width_m": 0.07,
        "speed_mps": 0.02,
        "force_N": 10.0,
    }


def test_close_command_uses_safe_close_width():
    preview = close_command_preview(
        {
            "safe_close_width_m": 0.04,
            "default_speed_mps": 0.02,
            "default_force_N": 10.0,
        },
        load_hardware_mode(),
    )

    assert preview["width_m"] == 0.04
    assert preview["width_m"] != 0.0
