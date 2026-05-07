from llm_skill_robot.core.schemas import MotionPlanCacheEntry, MotionPlanSummary
from llm_skill_robot.core.trajectory_visualizer import TrajectoryVisualizer


def test_trajectory_visualizer_imports():
    visualizer = TrajectoryVisualizer()

    assert visualizer.topic == "/display_planned_path"


def test_publish_preview_returns_failure_without_raw_trajectory():
    summary = MotionPlanSummary(
        plan_id="plan-no-raw",
        target_type="named_target",
        target_name="home",
        planning_group="ur_manipulator",
        success=True,
        message="Target command cached.",
    )
    entry = MotionPlanCacheEntry(
        plan_id="plan-no-raw",
        skill_id="move_to_named_target",
        summary=summary,
        created_at="2026-05-07T12:00:00Z",
        raw_plan_available=False,
    )
    visualizer = TrajectoryVisualizer()

    result = visualizer.publish_preview(entry)

    assert result["success"] is False
    assert result["message"] == "No raw trajectory available for RViz visual preview."
