from llm_skill_robot.core.schemas import MotionPlanSummary
from llm_skill_robot.core.trajectory_validator import TrajectoryValidator


def _safety_config():
    return {
        "trajectory_limits": {
            "max_joint_delta_rad": 0.5,
            "max_velocity_abs": 1.0,
            "min_duration_sec": 0.1,
        }
    }


def _summary(**overrides):
    data = {
        "plan_id": "plan-001",
        "target_type": "named_target",
        "target_name": "home",
        "planning_group": "ur_manipulator",
        "success": True,
        "message": "MoveIt plan generated.",
        "num_trajectory_points": 5,
        "duration_sec": 1.0,
        "joint_names": ["joint_1", "joint_2"],
        "max_joint_delta_rad": 0.2,
        "max_velocity_abs": 0.4,
        "trajectory_preview": {},
    }
    data.update(overrides)
    return MotionPlanSummary(**data)


def test_valid_summary_is_allowed():
    validator = TrajectoryValidator(_safety_config())

    result = validator.validate_motion_plan_summary(_summary())

    assert result["decision"] == "ALLOW"
    assert result["reasons"] == []


def test_failed_plan_is_blocked():
    validator = TrajectoryValidator(_safety_config())

    result = validator.validate_motion_plan_summary(_summary(success=False))

    assert result["decision"] == "BLOCK"
    assert "planning failed" in result["reasons"][0]


def test_zero_trajectory_points_blocked_in_non_dry_run():
    validator = TrajectoryValidator(_safety_config())

    result = validator.validate_motion_plan_summary(_summary(num_trajectory_points=0))

    assert result["decision"] == "BLOCK"
    assert "no trajectory points" in result["reasons"][0]


def test_zero_trajectory_points_allowed_in_dry_run():
    validator = TrajectoryValidator(_safety_config())

    result = validator.validate_motion_plan_summary(
        _summary(num_trajectory_points=0, trajectory_preview={"dry_run": True})
    )

    assert result["decision"] == "ALLOW"


def test_too_large_joint_delta_blocked():
    validator = TrajectoryValidator(_safety_config())

    result = validator.validate_motion_plan_summary(_summary(max_joint_delta_rad=0.8))

    assert result["decision"] == "BLOCK"
    assert "Max joint delta" in result["reasons"][0]


def test_too_high_velocity_blocked():
    validator = TrajectoryValidator(_safety_config())

    result = validator.validate_motion_plan_summary(_summary(max_velocity_abs=1.5))

    assert result["decision"] == "BLOCK"
    assert "Max velocity" in result["reasons"][0]
