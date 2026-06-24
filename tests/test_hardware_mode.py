import pytest

from llm_skill_robot.safety.hardware_mode import (
    assert_arm_execution_allowed,
    assert_gripper_execution_allowed,
    clamp_or_reject_gripper_command,
    load_hardware_mode,
    print_hardware_mode,
)


def test_default_hardware_mode_loads():
    mode = load_hardware_mode()
    assert mode.mode == "real_camera_real_gripper_arm_safe"
    assert mode.camera.enabled is True
    assert mode.camera.type == "d435i"


def test_arm_execution_is_blocked():
    with pytest.raises(PermissionError, match="Real arm execution is disabled"):
        assert_arm_execution_allowed(load_hardware_mode())


def test_gripper_execution_is_allowed():
    assert assert_gripper_execution_allowed(load_hardware_mode()) is None


@pytest.mark.parametrize(
    ("command", "message"),
    [
        ({"width_m": -0.001, "speed_mps": 0.01, "force_N": 10.0}, "below minimum"),
        ({"width_m": 0.141, "speed_mps": 0.01, "force_N": 10.0}, "exceeds maximum"),
        ({"width_m": 0.04, "speed_mps": 0.031, "force_N": 10.0}, "speed_mps"),
        ({"width_m": 0.04, "speed_mps": 0.01, "force_N": 20.1}, "force_N"),
    ],
)
def test_unsafe_gripper_command_is_rejected(command, message):
    with pytest.raises(ValueError, match=message):
        clamp_or_reject_gripper_command(mode=load_hardware_mode(), **command)


def test_safe_gripper_command_is_accepted_unchanged():
    command = {"width_m": 0.04, "speed_mps": 0.03, "force_N": 20.0}
    assert clamp_or_reject_gripper_command(
        mode=load_hardware_mode(), **command
    ) == command


def test_hardware_mode_summary_is_printed(capsys):
    print_hardware_mode(load_hardware_mode())
    output = capsys.readouterr().out
    assert "Camera: REAL D435i" in output
    assert "SAM3: conda subprocess" in output
    assert "GraspGenX: conda subprocess" in output
    assert "Arm execution: DISABLED" in output
    assert "Gripper execution: ENABLED with limits" in output
