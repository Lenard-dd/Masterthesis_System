from llm_skill_robot.core.schemas import MotionPlanCacheEntry
from llm_skill_robot.robot.ur5_moveit_plan_backend import UR5MoveItPlanBackend


def test_plan_to_named_target_returns_success_in_dry_run_mode():
    backend = UR5MoveItPlanBackend(dry_run=True)

    result = backend.plan_to_named_target("home")

    assert result["success"] is True
    assert result["message"] == "Dry-run plan generated."
    assert result["plan_id"].startswith("plan_")
    assert result["summary"]["target_type"] == "named_target"
    assert result["summary"]["target_name"] == "home"
    assert result["summary"]["planning_group"] == "ur_manipulator"


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
