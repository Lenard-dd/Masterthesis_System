from llm_skill_robot.core.schemas import MotionPlanSummary
from llm_skill_robot.core.trajectory_reviewer import TrajectoryReviewer


def _summary(**overrides):
    data = {
        "plan_id": "plan-review-001",
        "target_type": "named_target",
        "target_name": "home",
        "planning_group": "ur_manipulator",
        "success": True,
        "message": "Plan generated.",
        "num_trajectory_points": 5,
        "duration_sec": 1.2,
        "max_joint_delta_rad": 0.2,
        "max_velocity_abs": 0.03,
        "trajectory_preview": {"first_point_positions": [0.0]},
    }
    data.update(overrides)
    return MotionPlanSummary(**data)


def test_review_string_contains_plan_id():
    reviewer = TrajectoryReviewer()

    review = reviewer.review(_summary())

    assert "plan-review-001" in review


def test_review_string_contains_target():
    reviewer = TrajectoryReviewer()

    review = reviewer.review(_summary())

    assert "home" in review


def test_missing_optional_fields_produce_warnings():
    reviewer = TrajectoryReviewer()

    review = reviewer.review(
        _summary(
            duration_sec=None,
            max_joint_delta_rad=None,
            max_velocity_abs=None,
            trajectory_preview={},
        )
    )

    assert "Warnings:" in review
    assert "duration_sec is missing or unavailable." in review
    assert "trajectory_preview is missing or unavailable." in review


def test_approval_function_accepts_yes(monkeypatch, capsys):
    reviewer = TrajectoryReviewer()
    monkeypatch.setattr("builtins.input", lambda prompt: "yes")

    approved = reviewer.ask_for_review_approval(_summary())

    assert approved is True
    assert "Trajectory Preview Review" in capsys.readouterr().out


def test_approval_function_rejects_default_no(monkeypatch):
    reviewer = TrajectoryReviewer()
    monkeypatch.setattr("builtins.input", lambda prompt: "")

    approved = reviewer.ask_for_review_approval(_summary())

    assert approved is False
