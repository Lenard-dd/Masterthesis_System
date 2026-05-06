import pytest
from llm_skill_robot.core.execution_manager import ExecutionManager
from llm_skill_robot.core.skill_library import SkillLibrary
from llm_skill_robot.core.plan import RobotPlan, PlanStep
from llm_skill_robot.core.schemas import (
    VerificationResult,
    VerificationDecision,
    ValidationDecision,
    ValidationResult,
)
from llm_skill_robot.robot.mock_robot_backend import MockRobotBackend


def test_verified_and_approved_plan_executes_successfully():
    library = SkillLibrary()
    backend = MockRobotBackend()
    manager = ExecutionManager(library, backend)

    plan = RobotPlan(
        task="Pick and place",
        steps=[
            PlanStep(step_id="1", skill_id="move_to_named_target", description="Move", parameters={"target_name": "home"}),
            PlanStep(step_id="2", skill_id="open_gripper", description="Open", parameters={}),
            PlanStep(step_id="3", skill_id="pick_object", description="Pick", parameters={"object_id": "cube_1"}),
            PlanStep(step_id="4", skill_id="place_object", description="Place", parameters={"object_id": "cube_1"}),
        ]
    )

    verification = VerificationResult(decision=VerificationDecision.ALLOW)
    human_validation = ValidationResult(decision=ValidationDecision.APPROVE)
    report = manager.execute_plan(plan, verification, human_validation)

    assert report.status == "SUCCESS"
    assert len(report.step_results) == 4
    assert all(step.status == "SUCCESS" for step in report.step_results)


def test_blocked_verification_prevents_execution():
    library = SkillLibrary()
    backend = MockRobotBackend()
    manager = ExecutionManager(library, backend)

    plan = RobotPlan(task="Blocked", steps=[])
    verification = VerificationResult(decision=VerificationDecision.BLOCK)
    report = manager.execute_plan(plan, verification, "approved")

    assert report.status == "NOT_EXECUTED"
    assert report.step_results == []


def test_rejected_human_validation_prevents_execution():
    library = SkillLibrary()
    backend = MockRobotBackend()
    manager = ExecutionManager(library, backend)

    plan = RobotPlan(task="Rejected", steps=[])
    verification = VerificationResult(decision=VerificationDecision.ALLOW)
    report = manager.execute_plan(plan, verification, "rejected")

    assert report.status == "NOT_EXECUTED"
    assert report.step_results == []


def test_execution_stops_after_failed_step():
    library = SkillLibrary()
    backend = MockRobotBackend()
    manager = ExecutionManager(library, backend)

    plan = RobotPlan(
        task="Fail mid plan",
        steps=[
            PlanStep(step_id="1", skill_id="open_gripper", description="Open", parameters={}),
            PlanStep(step_id="2", skill_id="pick_object", description="Pick", parameters={"object_id": "unknown"}),
            PlanStep(step_id="3", skill_id="close_gripper", description="Close", parameters={}),
        ]
    )

    verification = VerificationResult(decision=VerificationDecision.ALLOW)
    report = manager.execute_plan(plan, verification, "approved")

    assert report.status == "FAILED"
    assert len(report.step_results) == 3
    assert report.step_results[0].status == "SUCCESS"
    assert report.step_results[1].status == "FAILED"
    assert report.step_results[2].status == "SKIPPED"


def test_missing_backend_method_causes_failed_step():
    library = SkillLibrary()

    class BackendWithoutMove:
        pass

    backend = BackendWithoutMove()
    manager = ExecutionManager(library, backend)

    plan = RobotPlan(
        task="Missing method",
        steps=[
            PlanStep(step_id="1", skill_id="move_to_named_target", description="Move", parameters={"target_name": "home"}),
        ]
    )

    verification = VerificationResult(decision=VerificationDecision.ALLOW)
    report = manager.execute_plan(plan, verification, "approved")

    assert report.status == "FAILED"
    assert len(report.step_results) == 1
    assert report.step_results[0].status == "FAILED"
    assert "not available" in report.step_results[0].message
