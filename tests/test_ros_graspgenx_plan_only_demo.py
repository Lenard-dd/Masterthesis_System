import importlib

import pytest

from llm_skill_robot.ros_graspgenx_plan_only_demo import (
    attempt_real_arm_execution,
    create_preview_poses,
    plan_pose_sequence,
)
from llm_skill_robot.safety.hardware_mode import load_hardware_mode


FAKE_CANDIDATE = {
    "candidate_id": "grasp_1",
    "score": 0.93,
    "gripper_width_m": 0.06,
    "pose": {
        "frame": "base_link",
        "position": {"x": 0.4, "y": 0.1, "z": 0.3},
        "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
    },
}


class AllowValidator:
    def validate_motion_plan_summary(self, summary):
        return {"decision": "ALLOW", "reasons": [], "warnings": []}


class FakePlanOnlyBackend:
    def __init__(self):
        self.plan_calls = []
        self.execution_calls = []

    def plan_to_pose(self, **kwargs):
        self.plan_calls.append(kwargs)
        plan_id = f"plan_{len(self.plan_calls)}"
        return {
            "success": True,
            "message": "fake plan only",
            "plan_id": plan_id,
            "summary": {
                "success": True,
                "num_trajectory_points": 2,
                "trajectory_preview": {},
            },
        }

    def get_plan_end_state(self, plan_id):
        return {"fake_start_after": plan_id}

    def execute_cached_plan_simulated(self, *args, **kwargs):
        self.execution_calls.append((args, kwargs))
        raise AssertionError("HW4 attempted trajectory execution")


def test_module_imports_without_ros_initialization():
    module = importlib.import_module("llm_skill_robot.ros_graspgenx_plan_only_demo")
    assert callable(module.main)


def test_fake_planner_reaches_all_plan_only_stages():
    backend = FakePlanOnlyBackend()
    poses = create_preview_poses(FAKE_CANDIDATE)

    result = plan_pose_sequence(backend, poses, AllowValidator())

    assert result["success"] is True
    assert [stage["stage"] for stage in result["stages"]] == [
        "pregrasp",
        "grasp",
        "retreat",
    ]
    assert len(backend.plan_calls) == 3
    assert backend.plan_calls[0]["start_state"] is None
    assert backend.plan_calls[1]["start_state"] == {"fake_start_after": "plan_1"}


def test_plan_only_sequence_never_calls_execution_function():
    backend = FakePlanOnlyBackend()

    result = plan_pose_sequence(
        backend, create_preview_poses(FAKE_CANDIDATE), AllowValidator()
    )

    assert result["success"] is True
    assert backend.execution_calls == []


def test_real_arm_execution_blocked_even_after_confirmation():
    with pytest.raises(PermissionError, match="disabled by hardware_mode.yaml"):
        attempt_real_arm_execution(load_hardware_mode(), confirmed=True)


def test_preview_pose_offsets_do_not_modify_grasp_pose():
    poses = create_preview_poses(FAKE_CANDIDATE)

    assert poses["grasp"]["position"] == {"x": 0.4, "y": 0.1, "z": 0.3}
    assert poses["pregrasp"]["position"]["z"] == pytest.approx(0.2)
    assert poses["retreat"]["position"]["z"] == pytest.approx(0.4)
