import pytest
from llm_skill_robot.core.skill_library import SkillLibrary
from llm_skill_robot.core.plan import RobotPlan, PlanStep
from llm_skill_robot.core.plan_verifier import PlanVerifier
from llm_skill_robot.core.schemas import VerificationDecision


def test_valid_plan_verification():
    """Test verification of a valid plan."""
    library = SkillLibrary()
    verifier = PlanVerifier(library)

    plan = RobotPlan(
        task="Test task",
        steps=[
            PlanStep(
                step_id="1",
                skill_id="move_to_named_target",
                description="Move to home",
                parameters={"target_name": "home"}
            )
        ]
    )

    result = verifier.verify_plan(plan)
    assert result.decision == VerificationDecision.ALLOW
    assert len(result.reasons) == 0


def test_unknown_skill():
    """Test verification with unknown skill."""
    library = SkillLibrary()
    verifier = PlanVerifier(library)

    plan = RobotPlan(
        task="Test task",
        steps=[
            PlanStep(
                step_id="1",
                skill_id="unknown_skill",
                description="Unknown skill",
                parameters={}
            )
        ]
    )

    result = verifier.verify_plan(plan)
    assert result.decision == VerificationDecision.BLOCK
    assert "Unknown skill: unknown_skill" in result.reasons


def test_missing_required_parameter():
    """Test verification with missing required parameter."""
    library = SkillLibrary()
    verifier = PlanVerifier(library)

    plan = RobotPlan(
        task="Test task",
        steps=[
            PlanStep(
                step_id="1",
                skill_id="pick_object",
                description="Pick object without object_id",
                parameters={}
            )
        ]
    )

    result = verifier.verify_plan(plan)
    assert result.decision == VerificationDecision.BLOCK
    assert any("Missing required parameter 'object_id'" in reason for reason in result.reasons)


def test_invalid_allowed_value():
    """Test verification with invalid allowed value."""
    library = SkillLibrary()
    verifier = PlanVerifier(library)

    plan = RobotPlan(
        task="Test task",
        steps=[
            PlanStep(
                step_id="1",
                skill_id="move_to_named_target",
                description="Move to invalid target",
                parameters={"target_name": "invalid_target"}
            )
        ]
    )

    result = verifier.verify_plan(plan)
    assert result.decision == VerificationDecision.BLOCK
    assert any("Invalid value 'invalid_target'" in reason for reason in result.reasons)