from llm_skill_robot.core.schemas import MotionPlanCacheEntry
from llm_skill_robot.core.trajectory_validator import TrajectoryValidator
from llm_skill_robot.robot.ur5_moveit_plan_backend import PYMOVEIT2_UNAVAILABLE_MESSAGE
from llm_skill_robot.robot.ur5_moveit_plan_backend import UR5MoveItPlanBackend


class Pymoveit2UnavailableBackend(UR5MoveItPlanBackend):
    def _pymoveit2_interface_available(self) -> bool:
        self._pymoveit2_error = PYMOVEIT2_UNAVAILABLE_MESSAGE
        return False


class Pymoveit2NoPlanApiBackend(UR5MoveItPlanBackend):
    def _pymoveit2_interface_available(self) -> bool:
        self._moveit2 = object()
        return True


class FakeJointTrajectory:
    joint_names = ["joint_1"]
    points = [type("FakePoint", (), {"positions": [0.1], "velocities": [0.0]})()]


class FakeRobotTrajectory:
    joint_trajectory = FakeJointTrajectory()


class FakePosePlanResponse:
    trajectory = FakeRobotTrajectory()
    trajectory_start = object()
    error_code = type("FakeErrorCode", (), {"val": 1})()


class EmptyJointTrajectory:
    joint_names = ["joint_1"]
    points = []


class EmptyRobotTrajectory:
    joint_trajectory = EmptyJointTrajectory()


class EmptyPosePlanResponse:
    trajectory = EmptyRobotTrajectory()
    trajectory_start = object()
    error_code = type("FakeErrorCode", (), {"val": 1})()


class EmptyPosePlanServiceBackend(UR5MoveItPlanBackend):
    def _request_moveit_pose_plan_service(self, **kwargs):
        return EmptyPosePlanResponse()


class PosePlanServiceBackend(UR5MoveItPlanBackend):
    def _request_moveit_pose_plan_service(self, **kwargs):
        self.pose_request = kwargs
        return FakePosePlanResponse()


def test_plan_to_named_target_returns_success_in_dry_run_mode():
    backend = UR5MoveItPlanBackend(dry_run=True)

    result = backend.plan_to_named_target("home")

    assert result["success"] is True
    assert result["message"] == "Dry-run plan generated."
    assert result["plan_id"].startswith("plan_")
    assert result["summary"]["target_type"] == "named_target"
    assert result["summary"]["target_name"] == "home"
    assert result["summary"]["planning_group"] == "ur5_manipulator"


def test_plan_id_is_cached():
    backend = UR5MoveItPlanBackend(dry_run=True)

    result = backend.plan_to_named_target("home")

    assert result["plan_id"] in backend.plan_cache


def test_get_plan_returns_cached_plan():
    backend = UR5MoveItPlanBackend(dry_run=True)

    result = backend.plan_to_named_target("home", step_id=3)
    cached_plan = backend.get_plan(result["plan_id"])

    assert isinstance(cached_plan, MotionPlanCacheEntry)
    assert cached_plan.plan_id == result["plan_id"]
    assert cached_plan.step_id == 3
    assert cached_plan.summary.target_name == "home"
    assert backend.get_cached_plan(result["plan_id"]) is cached_plan
    assert backend.has_executable_trajectory(result["plan_id"]) is False
    assert backend.has_visual_preview(result["plan_id"]) is False


def test_clear_plan_cache_clears_cache():
    backend = UR5MoveItPlanBackend(dry_run=True)
    backend.plan_to_named_target("home")

    backend.clear_plan_cache()

    assert backend.list_cached_plans() == []
    assert backend.plan_cache == {}


def test_plan_to_pose_returns_success_in_dry_run_mode():
    backend = UR5MoveItPlanBackend(dry_run=True)

    result = backend.plan_to_pose(
        frame="base_link",
        position={"x": 0.4, "y": 0.0, "z": 0.3},
        orientation={"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
    )

    assert result["success"] is True
    assert result["message"] == "Dry-run plan generated."
    assert result["summary"]["target_type"] == "pose"
    assert result["summary"]["target_pose"]["frame"] == "base_link"
    assert result["summary"]["target_pose"]["position"]["x"] == 0.4
    assert result["plan_id"] in backend.plan_cache


def test_pose_plan_only_service_result_is_cached_without_execution():
    backend = PosePlanServiceBackend(dry_run=False, node=object(), execution_mode="plan_only")

    result = backend.plan_to_pose(
        frame="base_link",
        position={"x": 0.4, "y": 0.0, "z": 0.3},
        orientation={"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
    )

    assert result["success"] is True
    assert result["summary"]["target_type"] == "pose"
    assert result["summary"]["num_trajectory_points"] == 1
    assert result["summary"]["trajectory_preview"]["moveit_diagnostics"] == {
        "error_code_value": 1,
        "error_code_name": "SUCCESS",
        "trajectory_present": True,
        "joint_trajectory_points": 1,
    }
    cached = backend.get_plan(result["plan_id"])
    assert cached.robot_trajectory is FakePosePlanResponse.trajectory
    assert backend.pose_request["frame"] == "base_link"
    assert backend.pose_request["use_orientation_constraint"] is True


def test_pose_plan_can_disable_orientation_constraint_for_diagnostics():
    backend = PosePlanServiceBackend(dry_run=False, node=object(), execution_mode="plan_only")

    result = backend.plan_to_pose(
        frame="base_link",
        position={"x": 0.4, "y": 0.0, "z": 0.3},
        orientation={"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        use_orientation_constraint=False,
    )

    assert result["success"] is True
    assert backend.pose_request["use_orientation_constraint"] is False


def test_pose_plan_forwards_moveit_request_tuning_options():
    backend = PosePlanServiceBackend(dry_run=False, node=object(), execution_mode="plan_only")

    result = backend.plan_to_pose(
        frame="base_link",
        position={"x": 0.4, "y": 0.0, "z": 0.3},
        orientation={"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        planner_id="RRTConnectkConfigDefault",
        num_planning_attempts=8,
        allowed_planning_time=7.5,
        service_timeout_sec=12.0,
    )

    assert result["success"] is True
    assert backend.pose_request["planner_id"] == "RRTConnectkConfigDefault"
    assert backend.pose_request["num_planning_attempts"] == 8
    assert backend.pose_request["allowed_planning_time"] == 7.5
    assert backend.pose_request["service_timeout_sec"] == 12.0


def test_pose_plan_service_empty_trajectory_is_clear_failure():
    backend = EmptyPosePlanServiceBackend(
        dry_run=False, node=object(), execution_mode="plan_only"
    )

    result = backend.plan_to_pose(
        frame="base_link",
        position={"x": 0.4, "y": 0.0, "z": 0.3},
        orientation={"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
    )

    assert result["success"] is False
    assert result["message"] == "MoveIt returned an empty joint trajectory."
    assert result["summary"]["num_trajectory_points"] == 0
    assert result["summary"]["trajectory_preview"]["moveit_diagnostics"] == {
        "error_code_value": 1,
        "error_code_name": "SUCCESS",
        "trajectory_present": True,
        "joint_trajectory_points": 0,
    }


def test_pymoveit2_unavailable_returns_clear_failure():
    backend = Pymoveit2UnavailableBackend(dry_run=False, node=object())

    result = backend.plan_to_named_target("home")

    assert result["success"] is False
    assert result["message"] == PYMOVEIT2_UNAVAILABLE_MESSAGE


def test_pymoveit2_no_plan_api_caches_target_command_for_rviz_sim():
    backend = Pymoveit2NoPlanApiBackend(
        dry_run=False,
        node=object(),
        mode="rviz_sim",
        execution_config={
            "mode": "rviz_sim",
            "allow_real_robot_execution": False,
            "require_human_confirmation": True,
            "require_final_confirmation_before_execution": True,
            "allowed_execution_modes": ["dry_run", "plan_only", "rviz_sim"],
        },
    )

    result = backend.plan_to_named_target("home")
    validation = TrajectoryValidator({}).validate_motion_plan_summary(result["summary"])

    assert result["success"] is True
    assert result["summary"]["num_trajectory_points"] == 0
    assert result["summary"]["trajectory_preview"]["pymoveit2_target_command"] is True
    assert result["plan_id"] in backend.plan_cache
    assert result["plan_id"] in backend._raw_plan_cache
    assert backend.has_executable_trajectory(result["plan_id"]) is True
    assert backend.has_visual_preview(result["plan_id"]) is False
    assert validation["decision"] == "ALLOW"
    assert "No explicit trajectory preview available" in validation["warnings"][0]
