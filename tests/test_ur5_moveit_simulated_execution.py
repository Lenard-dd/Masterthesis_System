from llm_skill_robot.robot.ur5_moveit_plan_backend import UR5MoveItPlanBackend
from llm_skill_robot.robot.ur5_moveit_plan_backend import extract_joint_trajectory
from llm_skill_robot.core.schemas import MotionPlanSummary
from llm_skill_robot.core.trajectory_validator import TrajectoryValidator


class FakeJointTrajectory:
    def __init__(self, joint_names=None, points=None):
        self.joint_names = joint_names or ["joint_1", "joint_2"]
        self.points = [object(), object()] if points is None else points


class FakeRobotTrajectory:
    def __init__(self, joint_trajectory):
        self.joint_trajectory = joint_trajectory


class FakeEmptyMoveItPlanResult:
    def __init__(self):
        self.trajectory = FakeRobotTrajectory(FakeJointTrajectory(points=[]))


class FakeDisplayTrajectory:
    def __init__(self, trajectory):
        self.trajectory = trajectory


class MockRvizSimBackend(UR5MoveItPlanBackend):
    def _moveit_execution_interface_available(self, plan_id: str) -> bool:
        return plan_id in self.plan_cache

    def _execute_plan_in_rviz_sim(self, plan_id: str):
        return {"mocked": True, "plan_id": plan_id}


class FakeMoveIt2:
    def __init__(self):
        self.executed = None
        self.moved_to_configuration = None
        self.waited = False
        self.wait_result = True

    def move_to_configuration(self, joint_positions):
        self.moved_to_configuration = joint_positions

    def execute(self, joint_trajectory):
        self.executed = joint_trajectory

    def wait_until_executed(self):
        self.waited = True
        return self.wait_result


class EmptyPlanRvizSimBackend(UR5MoveItPlanBackend):
    def _pymoveit2_interface_available(self) -> bool:
        self._moveit2 = FakeMoveIt2()
        return True

    def _call_pymoveit2_plan_api(self, target_positions):
        return FakeEmptyMoveItPlanResult()


class RealExecutionActionFallbackBackend(UR5MoveItPlanBackend):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_executed_plan_id = None

    def _execute_plan_via_execute_trajectory_action(self, plan_id: str):
        self.action_executed_plan_id = plan_id
        return {
            "real_execution": True,
            "command_type": "moveit_execute_trajectory_action",
            "moveit_execution_succeeded": True,
        }


def _rviz_sim_config():
    return {
        "mode": "rviz_sim",
        "allow_real_robot_execution": False,
        "require_human_confirmation": True,
        "require_final_confirmation_before_execution": True,
        "allowed_execution_modes": ["dry_run", "plan_only", "rviz_sim"],
    }


def test_execution_blocked_if_mode_is_not_rviz_sim():
    backend = UR5MoveItPlanBackend(dry_run=True, execution_mode="plan_only")
    plan = backend.plan_to_named_target("home")

    result = backend.execute_cached_plan_simulated(
        plan["plan_id"],
        final_confirmed=True,
        trajectory_validation_allowed=True,
        trajectory_review_approved=True,
    )

    assert result["success"] is False
    assert "rviz_sim" in result["message"]


def test_real_robot_mode_remains_blocked():
    backend = UR5MoveItPlanBackend(dry_run=True, execution_mode="real_robot")
    plan = backend.plan_to_named_target("home")

    result = backend.execute_cached_plan_simulated(
        plan["plan_id"],
        final_confirmed=True,
        trajectory_validation_allowed=True,
        trajectory_review_approved=True,
    )

    assert result["success"] is False
    assert "rviz_sim" in result["message"]


def test_execution_blocked_without_final_confirmation(capsys):
    backend = UR5MoveItPlanBackend(
        dry_run=True,
        execution_mode="rviz_sim",
        execution_config=_rviz_sim_config(),
    )
    plan = backend.plan_to_named_target("home")

    result = backend.execute_cached_plan_simulated(
        plan["plan_id"],
        final_confirmed=False,
        trajectory_validation_allowed=True,
        trajectory_review_approved=True,
    )

    assert result["success"] is False
    assert result["message"] == "Final confirmation required before RViz/fake hardware execution."
    assert "Final confirmation required before RViz/fake hardware execution." in capsys.readouterr().out


def test_execution_blocked_for_unknown_plan_id():
    backend = UR5MoveItPlanBackend(
        dry_run=True,
        execution_mode="rviz_sim",
        execution_config=_rviz_sim_config(),
    )

    result = backend.execute_cached_plan_simulated(
        "plan_missing",
        final_confirmed=True,
        trajectory_validation_allowed=True,
        trajectory_review_approved=True,
    )

    assert result["success"] is False
    assert "No cached plan" in result["message"]


def test_empty_moveit_plan_falls_back_to_target_command_in_rviz_sim():
    backend = EmptyPlanRvizSimBackend(
        dry_run=False,
        execution_mode="rviz_sim",
        execution_config=_rviz_sim_config(),
    )

    result = backend.plan_to_named_target("home")

    assert result["success"] is True
    assert result["summary"]["num_trajectory_points"] == 0
    assert result["summary"]["trajectory_preview"]["pymoveit2_target_command"] is True
    validation = TrajectoryValidator({}).validate_motion_plan_summary(result["summary"])
    assert validation["decision"] == "ALLOW"


def test_execution_blocked_for_failed_cached_plan():
    backend = UR5MoveItPlanBackend(
        dry_run=True,
        execution_mode="rviz_sim",
        execution_config=_rviz_sim_config(),
    )
    summary = MotionPlanSummary(
        plan_id="plan_failed",
        target_type="named_target",
        target_name="home",
        planning_group="ur_manipulator",
        success=False,
        message="Planning failed.",
    )
    backend._cache_plan(summary=summary, skill_id="move_to_named_target")

    result = backend.execute_cached_plan_simulated(
        "plan_failed",
        final_confirmed=True,
        trajectory_validation_allowed=True,
        trajectory_review_approved=True,
    )

    assert result["success"] is False
    assert "planning failed" in result["message"]


def test_execution_succeeds_in_mocked_rviz_sim_mode():
    backend = MockRvizSimBackend(
        dry_run=True,
        execution_mode="rviz_sim",
        execution_config=_rviz_sim_config(),
    )
    plan = backend.plan_to_named_target("home")

    result = backend.execute_cached_plan_simulated(
        plan["plan_id"],
        final_confirmed=True,
        trajectory_validation_allowed=True,
        trajectory_review_approved=True,
    )

    assert result["success"] is True
    assert result["plan_id"] == plan["plan_id"]
    assert result["mode"] == "rviz_sim"
    assert result["output"] == {"mocked": True, "plan_id": plan["plan_id"]}


def test_simulated_execution_passes_joint_trajectory_to_pymoveit2():
    backend = UR5MoveItPlanBackend(
        dry_run=True,
        execution_mode="rviz_sim",
        execution_config=_rviz_sim_config(),
    )
    moveit2 = FakeMoveIt2()
    joint_trajectory = FakeJointTrajectory()
    summary = MotionPlanSummary(
        plan_id="plan_joint_traj",
        target_type="named_target",
        target_name="home",
        planning_group="ur_manipulator",
        success=True,
        message="Planning succeeded.",
        num_trajectory_points=2,
    )
    backend._moveit2 = moveit2
    backend._cache_plan(
        summary=summary,
        skill_id="move_to_named_target",
        joint_trajectory=joint_trajectory,
    )

    result = backend.execute_cached_plan_simulated(
        "plan_joint_traj",
        final_confirmed=True,
        trajectory_validation_allowed=True,
        trajectory_review_approved=True,
    )

    assert result["success"] is True
    assert result["output"]["command_type"] == "pymoveit2_joint_trajectory"
    assert result["output"]["trajectory_source"] == "cached_plan.joint_trajectory"
    assert result["output"]["moveit_execution_succeeded"] is True
    assert moveit2.executed is joint_trajectory
    assert moveit2.waited is True


def test_simulated_execution_reports_moveit_target_command_failure():
    backend = UR5MoveItPlanBackend(
        dry_run=True,
        execution_mode="rviz_sim",
        execution_config=_rviz_sim_config(),
    )
    moveit2 = FakeMoveIt2()
    moveit2.wait_result = False
    plan = backend._cache_pymoveit2_target_command(
        plan_id="plan_target_command",
        target="home",
        target_positions=[0.0, -1.57, 1.57, -1.57, -1.57, 0.0],
        velocity_scale=0.1,
        acceleration_scale=0.1,
        skill_id="move_to_named_target",
        step_id=1,
    )
    backend._moveit2 = moveit2

    result = backend.execute_cached_plan_simulated(
        plan["plan_id"],
        final_confirmed=True,
        trajectory_validation_allowed=True,
        trajectory_review_approved=True,
    )

    assert moveit2.moved_to_configuration == [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]
    assert result["success"] is False
    assert "not executed successfully" in result["message"]


def test_display_trajectory_is_not_used_for_simulated_execution():
    backend = UR5MoveItPlanBackend(
        dry_run=True,
        execution_mode="rviz_sim",
        execution_config=_rviz_sim_config(),
    )
    summary = MotionPlanSummary(
        plan_id="plan_display_only",
        target_type="named_target",
        target_name="home",
        planning_group="ur_manipulator",
        success=True,
        message="Planning succeeded.",
    )
    backend._moveit2 = FakeMoveIt2()
    backend._cache_plan(
        summary=summary,
        skill_id="move_to_named_target",
        display_trajectory=object(),
    )

    result = backend.execute_cached_plan_simulated(
        "plan_display_only",
        final_confirmed=True,
        trajectory_validation_allowed=True,
        trajectory_review_approved=True,
    )

    assert backend.has_visual_preview("plan_display_only") is True
    assert backend.has_executable_trajectory("plan_display_only") is False
    assert result["success"] is False
    assert "interface is not available" in result["message"]


def test_execution_blocked_without_trajectory_validation_allow():
    backend = UR5MoveItPlanBackend(
        dry_run=True,
        execution_mode="rviz_sim",
        execution_config=_rviz_sim_config(),
    )
    plan = backend.plan_to_named_target("home")

    result = backend.execute_cached_plan_simulated(
        plan["plan_id"],
        final_confirmed=True,
        trajectory_validation_allowed=False,
        trajectory_review_approved=True,
    )

    assert result["success"] is False
    assert "Trajectory validation must allow" in result["message"]


def test_execution_blocked_without_trajectory_review_approval():
    backend = UR5MoveItPlanBackend(
        dry_run=True,
        execution_mode="rviz_sim",
        execution_config=_rviz_sim_config(),
    )
    plan = backend.plan_to_named_target("home")

    result = backend.execute_cached_plan_simulated(
        plan["plan_id"],
        final_confirmed=True,
        trajectory_validation_allowed=True,
        trajectory_review_approved=False,
    )

    assert result["success"] is False
    assert "Trajectory review must be approved" in result["message"]


def test_extract_joint_trajectory_from_cached_joint_trajectory():
    backend = UR5MoveItPlanBackend(dry_run=True)
    summary = MotionPlanSummary(
        plan_id="plan_direct_joint_traj",
        target_type="named_target",
        target_name="home",
        planning_group="ur_manipulator",
        success=True,
        message="Planning succeeded.",
    )
    joint_trajectory = FakeJointTrajectory()
    entry = backend._cache_plan(
        summary=summary,
        skill_id="move_to_named_target",
        joint_trajectory=joint_trajectory,
    )

    result = extract_joint_trajectory(entry)

    assert result.success is True
    assert result.joint_trajectory is joint_trajectory
    assert result.source == "cached_plan.joint_trajectory"


def test_extract_joint_trajectory_from_robot_trajectory():
    backend = UR5MoveItPlanBackend(dry_run=True)
    summary = MotionPlanSummary(
        plan_id="plan_robot_traj",
        target_type="named_target",
        target_name="home",
        planning_group="ur_manipulator",
        success=True,
        message="Planning succeeded.",
    )
    joint_trajectory = FakeJointTrajectory()
    entry = backend._cache_plan(
        summary=summary,
        skill_id="move_to_named_target",
        robot_trajectory=FakeRobotTrajectory(joint_trajectory),
    )

    result = extract_joint_trajectory(entry)

    assert result.success is True
    assert result.joint_trajectory is joint_trajectory
    assert result.source == "cached_plan.robot_trajectory.joint_trajectory"


def test_extract_joint_trajectory_from_display_trajectory_first_trajectory():
    backend = UR5MoveItPlanBackend(dry_run=True)
    summary = MotionPlanSummary(
        plan_id="plan_display_traj",
        target_type="named_target",
        target_name="home",
        planning_group="ur_manipulator",
        success=True,
        message="Planning succeeded.",
    )
    joint_trajectory = FakeJointTrajectory()
    robot_trajectory = FakeRobotTrajectory(joint_trajectory)
    entry = backend._cache_plan(
        summary=summary,
        skill_id="move_to_named_target",
        display_trajectory=FakeDisplayTrajectory([robot_trajectory]),
    )

    result = extract_joint_trajectory(entry)

    assert result.success is True
    assert result.joint_trajectory is joint_trajectory
    assert result.source == "cached_plan.display_trajectory.trajectory[0].joint_trajectory"


def test_extract_joint_trajectory_converts_supported_dict_to_message():
    backend = UR5MoveItPlanBackend(dry_run=True)
    summary = MotionPlanSummary(
        plan_id="plan_dict_traj",
        target_type="named_target",
        target_name="home",
        planning_group="ur_manipulator",
        success=True,
        message="Planning succeeded.",
    )
    entry = backend._cache_plan(
        summary=summary,
        skill_id="move_to_named_target",
        joint_trajectory={
            "joint_names": ["joint_1"],
            "points": [{"positions": [0.0]}],
        },
    )

    result = extract_joint_trajectory(entry)

    assert result.success is True
    assert result.joint_trajectory is not None
    assert not isinstance(result.joint_trajectory, dict)
    assert result.joint_trajectory.joint_names == ["joint_1"]


def test_extract_joint_trajectory_rejects_invalid_dict():
    backend = UR5MoveItPlanBackend(dry_run=True)
    summary = MotionPlanSummary(
        plan_id="plan_invalid_dict_traj",
        target_type="named_target",
        target_name="home",
        planning_group="ur_manipulator",
        success=True,
        message="Planning succeeded.",
    )
    entry = backend._cache_plan(
        summary=summary,
        skill_id="move_to_named_target",
        joint_trajectory={"points": [{"positions": [0.0]}]},
    )

    result = extract_joint_trajectory(entry)

    assert result.success is False
    assert result.joint_trajectory is None
    assert "No valid JointTrajectory found" in result.reason


def test_real_execution_uses_execute_trajectory_action_fallback_without_pymoveit2():
    backend = RealExecutionActionFallbackBackend(
        dry_run=True,
        execution_mode="real_robot",
    )
    summary = MotionPlanSummary(
        plan_id="plan_real_action_fallback",
        target_type="pose",
        planning_group="ur_manipulator",
        success=True,
        message="Planning succeeded.",
    )
    entry = backend._cache_plan(
        summary=summary,
        skill_id="move_to_pregrasp",
        joint_trajectory=FakeJointTrajectory(),
    )

    result = backend.execute_cached_plan_real(entry.plan_id, final_confirmed=True)

    assert result["success"] is True
    assert result["output"]["command_type"] == "moveit_execute_trajectory_action"
    assert backend.action_executed_plan_id == entry.plan_id
