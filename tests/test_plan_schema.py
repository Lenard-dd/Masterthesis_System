import pytest
from llm_skill_robot.core.plan import RobotPlan, PlanStep


def test_robot_plan_creation():
    """Test creating a valid RobotPlan."""
    plan = RobotPlan(
        task="Test task",
        steps=[
            PlanStep(
                step_id="1",
                skill_id="test_skill",
                description="Test step",
                parameters={"param1": "value1"}
            )
        ]
    )
    assert plan.task == "Test task"
    assert len(plan.steps) == 1
    assert plan.steps[0].skill_id == "test_skill"


def test_plan_step_validation():
    """Test PlanStep validation."""
    step = PlanStep(
        step_id="1",
        skill_id="test_skill",
        description="Test step"
    )
    assert step.parameters == {}