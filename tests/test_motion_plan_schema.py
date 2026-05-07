import pytest
from pydantic import ValidationError

from llm_skill_robot.core.schemas import MotionPlanCacheEntry, MotionPlanSummary


def test_valid_motion_plan_summary_creation():
    """Test creating a valid plan-only MoveIt summary."""
    summary = MotionPlanSummary(
        plan_id="plan-001",
        target_type="named_target",
        target_name="home",
        planning_group="ur_manipulator",
        success=True,
        message="Plan generated successfully.",
        num_trajectory_points=12,
        duration_sec=1.5,
        joint_names=[
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
        ],
        max_joint_delta_rad=0.42,
        max_velocity_abs=0.25,
        trajectory_preview={"first_point": [0.0, -1.57], "last_point": [0.1, -1.2]},
    )

    assert summary.plan_id == "plan-001"
    assert summary.target_type == "named_target"
    assert summary.target_name == "home"
    assert summary.target_pose is None
    assert summary.planning_group == "ur_manipulator"
    assert summary.success is True
    assert summary.num_trajectory_points == 12
    assert summary.joint_names[0] == "shoulder_pan_joint"
    assert summary.trajectory_preview["last_point"] == [0.1, -1.2]


def test_motion_plan_summary_missing_required_fields_rejected():
    """Test missing required fields are rejected."""
    with pytest.raises(ValidationError):
        MotionPlanSummary(
            plan_id="plan-002",
            target_type="named_target",
            success=True,
            message="Missing planning group.",
        )


def test_motion_plan_cache_entry_creation():
    """Test creating a cache entry for a plan-only result."""
    summary = MotionPlanSummary(
        plan_id="plan-003",
        target_type="pose",
        target_pose={"position": {"x": 0.4, "y": 0.0, "z": 0.3}},
        planning_group="ur_manipulator",
        success=False,
        message="Planning failed.",
    )
    entry = MotionPlanCacheEntry(
        plan_id="plan-003",
        skill_id="move_to_pose",
        step_id=1,
        summary=summary,
        created_at="2026-05-07T12:00:00Z",
        raw_plan_available=True,
    )

    assert entry.plan_id == "plan-003"
    assert entry.skill_id == "move_to_pose"
    assert entry.step_id == 1
    assert entry.summary.message == "Planning failed."
    assert entry.raw_plan_available is True
