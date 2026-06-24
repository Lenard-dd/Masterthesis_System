import importlib

from llm_skill_robot.ros_graspgenx_gripper_width_test import (
    WIDTH_MISSING_MESSAGE,
    command_candidate_width_if_confirmed,
    extract_candidate_width,
)
from llm_skill_robot.safety.hardware_mode import load_hardware_mode


class FakeGripperBackend:
    def __init__(self):
        self.commands = []

    def set_gripper_width(self, width_m, speed_mps, force_N, confirmed=False):
        self.commands.append(
            {
                "width_m": width_m,
                "speed_mps": speed_mps,
                "force_N": force_N,
                "confirmed": confirmed,
            }
        )
        return {"success": True, "status": "SENT", "message": "fake sent"}


def test_module_imports_without_ros_initialization():
    module = importlib.import_module(
        "llm_skill_robot.ros_graspgenx_gripper_width_test"
    )
    assert callable(module.main)


def test_missing_candidate_width_stops_before_backend():
    backend = FakeGripperBackend()

    result = command_candidate_width_if_confirmed(
        {"candidate_id": "g1", "score": 0.9},
        backend,
        load_hardware_mode(),
        "YES",
        speed_mps=0.02,
        force_N=10.0,
    )

    assert result["success"] is False
    assert result["message"] == WIDTH_MISSING_MESSAGE
    assert backend.commands == []


def test_extract_missing_width_raises_exact_message():
    try:
        extract_candidate_width({"candidate_id": "g1"})
    except ValueError as exc:
        assert str(exc) == WIDTH_MISSING_MESSAGE
    else:
        raise AssertionError("missing candidate width was accepted")


def test_unsafe_candidate_width_is_rejected_before_backend():
    backend = FakeGripperBackend()

    result = command_candidate_width_if_confirmed(
        {"candidate_id": "g1", "gripper_width_m": 0.20},
        backend,
        load_hardware_mode(),
        "YES",
        speed_mps=0.02,
        force_N=10.0,
    )

    assert result["status"] == "REJECTED"
    assert "exceeds maximum" in result["message"]
    assert backend.commands == []


def test_fake_candidate_width_is_sent_after_confirmation(monkeypatch):
    module = importlib.import_module(
        "llm_skill_robot.ros_graspgenx_gripper_width_test"
    )
    monkeypatch.setattr(module, "confirmation_is_yes", lambda value: value == "MOCKED_YES")
    backend = FakeGripperBackend()

    result = module.command_candidate_width_if_confirmed(
        {"candidate_id": "g1", "gripper_width_m": 0.065},
        backend,
        load_hardware_mode(),
        "MOCKED_YES",
        speed_mps=0.02,
        force_N=10.0,
    )

    assert result["success"] is True
    assert backend.commands == [
        {
            "width_m": 0.065,
            "speed_mps": 0.02,
            "force_N": 10.0,
            "confirmed": True,
        }
    ]


def test_arm_execution_function_is_never_called(monkeypatch):
    from llm_skill_robot.robot.ur5_moveit_plan_backend import UR5MoveItPlanBackend

    arm_calls = []
    monkeypatch.setattr(
        UR5MoveItPlanBackend,
        "execute_cached_plan_simulated",
        lambda *args, **kwargs: arm_calls.append((args, kwargs)),
    )
    backend = FakeGripperBackend()

    result = command_candidate_width_if_confirmed(
        {"candidate_id": "g1", "gripper_width_m": 0.05},
        backend,
        load_hardware_mode(),
        "YES",
        speed_mps=0.02,
        force_N=10.0,
    )

    assert result["success"] is True
    assert arm_calls == []
