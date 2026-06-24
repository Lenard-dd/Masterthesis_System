import importlib

import pytest

from llm_skill_robot.robot.robotiq_2f140_real_backend import (
    DRIVER_NOT_CONFIGURED_MESSAGE,
    Robotiq2F140RealBackend,
)


def _config(interface_type="auto"):
    return {
        "gripper": {
            "default_open_width_m": 0.14,
            "safe_close_width_m": 0.04,
            "default_speed_mps": 0.02,
            "default_force_N": 10.0,
            "real_gripper_enabled": True,
            "driver": {
                "interface_type": interface_type,
                "command_topic": "",
                "state_topic": "",
                "action_name": "",
                "service_name": "",
                "width_unit": "meters",
            },
        }
    }


def test_close_gripper_default_uses_safe_close_width():
    result = Robotiq2F140RealBackend(config=_config("dry_run")).close_gripper(
        confirmed=True
    )

    assert result["status"] == "DRY_RUN"
    assert result["output"]["command_preview"]["width_m"] == 0.04
    assert result["output"]["command_sent"] is False


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"width_m": -0.01}, "below minimum"),
        ({"width_m": 0.15}, "exceeds maximum"),
        ({"width_m": 0.04, "speed_mps": 0.04}, "speed_mps"),
        ({"width_m": 0.04, "force_N": 21.0}, "force_N"),
    ],
)
def test_unsafe_commands_are_rejected_before_driver(kwargs, message):
    backend = Robotiq2F140RealBackend(config=_config("dry_run"))

    result = backend.set_gripper_width(confirmed=True, **kwargs)

    assert result["success"] is False
    assert result["status"] == "REJECTED"
    assert message in result["message"]
    assert backend.last_command is None


def test_human_confirmation_is_required_before_driver():
    result = Robotiq2F140RealBackend(config=_config("dry_run")).open_gripper()

    assert result["success"] is False
    assert result["status"] == "REJECTED"
    assert "Human confirmation is required" in result["message"]


def test_auto_interface_returns_not_available():
    result = Robotiq2F140RealBackend(config=_config("auto")).open_gripper(
        confirmed=True
    )

    assert result["success"] is False
    assert result["status"] == "NOT_AVAILABLE"
    assert result["message"] == DRIVER_NOT_CONFIGURED_MESSAGE
    assert result["output"]["command_sent"] is False


def test_partial_config_still_uses_hardware_mode_enablement():
    backend = Robotiq2F140RealBackend(
        config={"driver": {"interface_type": "auto"}}
    )

    result = backend.open_gripper(confirmed=True)

    assert result["status"] == "NOT_AVAILABLE"
    assert result["message"] == DRIVER_NOT_CONFIGURED_MESSAGE


def test_dry_run_does_not_send_real_command():
    class NodeThatMustNotBeUsed:
        def create_publisher(self, *args, **kwargs):
            raise AssertionError("dry_run attempted to create a ROS publisher")

    backend = Robotiq2F140RealBackend(
        node=NodeThatMustNotBeUsed(), config=_config("dry_run")
    )
    result = backend.set_gripper_width(0.07, confirmed=True)

    assert result["success"] is False
    assert result["status"] == "DRY_RUN"
    assert result["output"]["command_sent"] is False
    assert backend.last_command is None


def test_get_state_without_feedback_topic_reports_unavailable():
    result = Robotiq2F140RealBackend(config=_config()).get_gripper_state()

    assert result["success"] is False
    assert result["output"]["mode"] == "real_hardware"
    assert result["output"]["feedback_available"] is False
    assert result["output"]["real_gripper_enabled"] is True


def test_modules_import_without_ros_graph():
    assert importlib.import_module(
        "llm_skill_robot.robot.robotiq_2f140_real_backend"
    ).Robotiq2F140RealBackend
    assert callable(
        importlib.import_module("llm_skill_robot.ros_check_real_gripper").main
    )
