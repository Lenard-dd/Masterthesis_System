from llm_skill_robot.grasping.grasp_execution_backend import GraspExecutionBackend
from llm_skill_robot.safety.real_arm_safety import load_real_arm_safety


def _pose(z, frame="base_link"):
    return {
        "frame": frame,
        "position": {"x": 0.4, "y": 0.0, "z": z},
        "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
    }


def _grasp_plan(frame="base_link"):
    return {
        "pregrasp_pose": _pose(0.36, frame),
        "grasp_pose": _pose(0.31, frame),
        "retreat_pose": _pose(0.40, frame),
    }


class FakeMoveItBackend:
    def __init__(self):
        self.plan_calls = []
        self.execute_calls = []
        self.current_pose = _pose(0.31)

    def plan_to_pose(self, **kwargs):
        self.plan_calls.append(kwargs)
        return {
            "success": True,
            "plan_id": f"plan_{len(self.plan_calls)}",
            "summary": {
                "success": True,
                "num_trajectory_points": 4,
                "max_joint_delta_rad": 0.1,
                "total_joint_delta_rad": 0.2,
                "trajectory_preview": {"max_cartesian_step_m": 0.02},
            },
        }

    def get_current_pose(self):
        return self.current_pose

    def execute_cached_plan_real(self, plan_id, final_confirmed=False):
        self.execute_calls.append((plan_id, final_confirmed))
        return {"success": True, "message": "executed", "output": {}}


def _backend(moveit):
    return GraspExecutionBackend(
        moveit,
        load_real_arm_safety(),
        review_approval_provider=lambda _summary: True,
        confirmation_provider=lambda _prompt: "EXECUTE",
    )


def test_approach_grasp_uses_grasp_pose():
    moveit = FakeMoveItBackend()

    result = _backend(moveit).approach_grasp("cube_1", {"cube_1": _grasp_plan()})

    assert result["success"] is True
    assert moveit.plan_calls[0]["position"] == _grasp_plan()["grasp_pose"]["position"]


def test_retreat_grasp_uses_retreat_pose():
    moveit = FakeMoveItBackend()

    result = _backend(moveit).retreat_grasp("cube_1", {"cube_1": _grasp_plan()})

    assert result["success"] is True
    assert moveit.plan_calls[0]["position"] == _grasp_plan()["retreat_pose"]["position"]


def test_lift_object_caps_distance_to_safety_limit():
    moveit = FakeMoveItBackend()
    config = load_real_arm_safety()

    result = _backend(moveit).lift_object(
        "cube_1", {"cube_1": _grasp_plan()}, lift_distance_m=1.0
    )

    assert result["success"] is True
    assert result["summary"]["distance_m"] == config.limits.max_lift_distance_m
    assert moveit.plan_calls[0]["position"]["z"] == 0.31 + config.limits.max_lift_distance_m


def test_no_execution_without_exact_execute_confirmation():
    moveit = FakeMoveItBackend()

    result = _backend(moveit).approach_grasp(
        "cube_1", {"cube_1": _grasp_plan()}, confirmation="NO"
    )

    assert result["success"] is False
    assert moveit.execute_calls == []


def test_pose_not_in_base_link_is_blocked_before_planning():
    moveit = FakeMoveItBackend()

    result = _backend(moveit).approach_grasp(
        "cube_1", {"cube_1": _grasp_plan(frame="camera_color_optical_frame")}
    )

    assert result["success"] is False
    assert "base_link" in result["message"]
    assert moveit.plan_calls == []


def test_missing_grasp_plan_is_blocked_before_planning():
    moveit = FakeMoveItBackend()

    result = _backend(moveit).move_to_pregrasp("cube_1", {})

    assert result["success"] is False
    assert "GraspPlan" in result["message"]
    assert moveit.plan_calls == []
