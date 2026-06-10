import pytest

from llm_skill_robot.agent.mock_plan_generator import MockPlanGenerator, PlanGenerationError
from llm_skill_robot.core.plan import RobotPlan


def _generate(instruction: str) -> RobotPlan:
    return MockPlanGenerator().generate_plan(
        user_instruction=instruction,
        available_skills=[],
    )


def test_move_to_home_generates_named_target_plan():
    plan = _generate("move to home")

    assert plan.task == "move to home"
    assert len(plan.steps) == 1
    step = plan.steps[0]
    assert step.step_id == "1"
    assert step.skill_id == "move_to_named_target"
    assert step.parameters["target"] == "home"
    assert step.parameters["velocity_scale"] == 0.03
    assert step.parameters["acceleration_scale"] == 0.03


def test_move_to_observe_generates_named_target_plan():
    plan = _generate("move to observe")

    assert len(plan.steps) == 1
    assert plan.steps[0].skill_id == "move_to_named_target"
    assert plan.steps[0].parameters["target"] == "observe"


def test_open_gripper_generates_open_gripper_step():
    plan = _generate("open gripper")

    assert len(plan.steps) == 1
    assert plan.steps[0].step_id == "1"
    assert plan.steps[0].skill_id == "open_gripper"
    assert plan.steps[0].parameters == {}


def test_close_gripper_generates_close_gripper_step():
    plan = _generate("close gripper")

    assert len(plan.steps) == 1
    assert plan.steps[0].step_id == "1"
    assert plan.steps[0].skill_id == "close_gripper"
    assert plan.steps[0].parameters == {}


def test_motion_and_gripper_command_generate_ordered_steps():
    plan = _generate("move to observe and open the gripper")

    assert len(plan.steps) == 2
    assert plan.steps[0].step_id == "1"
    assert plan.steps[0].skill_id == "move_to_named_target"
    assert plan.steps[0].parameters["target"] == "observe"
    assert plan.steps[1].step_id == "2"
    assert plan.steps[1].skill_id == "open_gripper"
    assert plan.steps[1].parameters == {}


def test_unknown_instruction_fails_clearly():
    with pytest.raises(PlanGenerationError, match="Could not generate a RobotPlan"):
        _generate("make me a cup of tea")
