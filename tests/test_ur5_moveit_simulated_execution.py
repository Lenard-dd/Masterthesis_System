from llm_skill_robot.robot.ur5_moveit_plan_backend import UR5MoveItPlanBackend
from llm_skill_robot.core.schemas import MotionPlanSummary


class MockRvizSimBackend(UR5MoveItPlanBackend):
    def _moveit_execution_interface_available(self, plan_id: str) -> bool:
        return plan_id in self.plan_cache

    def _execute_plan_in_rviz_sim(self, plan_id: str):
        return {"mocked": True, "plan_id": plan_id}


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
    )

    assert result["success"] is False
    assert "rviz_sim" in result["message"]


def test_real_robot_mode_remains_blocked():
    backend = UR5MoveItPlanBackend(dry_run=True, execution_mode="real_robot")
    plan = backend.plan_to_named_target("home")

    result = backend.execute_cached_plan_simulated(
        plan["plan_id"],
        final_confirmed=True,
    )

    assert result["success"] is False
    assert "rviz_sim" in result["message"]


def test_execution_blocked_without_final_confirmation():
    backend = UR5MoveItPlanBackend(
        dry_run=True,
        execution_mode="rviz_sim",
        execution_config=_rviz_sim_config(),
    )
    plan = backend.plan_to_named_target("home")

    result = backend.execute_cached_plan_simulated(
        plan["plan_id"],
        final_confirmed=False,
    )

    assert result["success"] is False
    assert "Final human confirmation" in result["message"]


def test_execution_blocked_for_unknown_plan_id():
    backend = UR5MoveItPlanBackend(
        dry_run=True,
        execution_mode="rviz_sim",
        execution_config=_rviz_sim_config(),
    )

    result = backend.execute_cached_plan_simulated(
        "plan_missing",
        final_confirmed=True,
    )

    assert result["success"] is False
    assert "No cached plan" in result["message"]


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
    )

    assert result["success"] is True
    assert result["plan_id"] == plan["plan_id"]
    assert result["mode"] == "rviz_sim"
    assert result["output"] == {"mocked": True, "plan_id": plan["plan_id"]}
