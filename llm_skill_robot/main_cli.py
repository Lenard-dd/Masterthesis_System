#!/usr/bin/env python3
"""Main CLI for LLM Skill Robot demo."""

from llm_skill_robot.core.skill_library import SkillLibrary
from llm_skill_robot.safety.hardware_mode import load_hardware_mode, print_hardware_mode
from llm_skill_robot.core.plan import RobotPlan, PlanStep
from llm_skill_robot.core.plan_verifier import PlanVerifier
from llm_skill_robot.core.human_validator import HumanValidator
from llm_skill_robot.core.audit_logger import AuditLogger
from llm_skill_robot.robot.mock_robot_backend import MockRobotBackend
from llm_skill_robot.core.execution_manager import ExecutionManager
from llm_skill_robot.core.execution_evaluator import ExecutionEvaluator


def main():
    print_hardware_mode(load_hardware_mode())
    # Initialize components
    skill_library = SkillLibrary()
    plan_verifier = PlanVerifier(skill_library)
    human_validator = HumanValidator()
    audit_logger = AuditLogger()
    robot_backend = MockRobotBackend()
    execution_manager = ExecutionManager(skill_library, robot_backend, audit_logger=audit_logger)
    evaluator = ExecutionEvaluator()

    # Create a hardcoded example plan
    plan = RobotPlan(
        task="Pick and place object",
        steps=[
            PlanStep(
                step_id="1",
                skill_id="move_to_named_target",
                description="Move to pickup position",
                parameters={"target_name": "pickup"}
            ),
            PlanStep(
                step_id="2",
                skill_id="open_gripper",
                description="Open gripper",
                parameters={}
            ),
            PlanStep(
                step_id="3",
                skill_id="pick_object",
                description="Pick up the object",
                parameters={"object_id": "cube_1"}
            ),
            PlanStep(
                step_id="4",
                skill_id="place_object",
                description="Place the object",
                parameters={"object_id": "cube_1"}
            )
        ]
    )

    # Show skill library and plan
    print("Available skills:")
    for skill_id in skill_library.list_skills():
        print(f"- {skill_id}")

    print("\nGenerated plan:")
    for step in plan.steps:
        print(f"{step.step_id}. {step.skill_id} - {step.description} {step.parameters}")

    # Log plan generation
    audit_logger.log_plan_generation(plan)

    # Verify the plan
    verification_result = plan_verifier.verify_plan(plan)
    audit_logger.log_verification(plan, verification_result)

    print(f"\nPlan verification: {verification_result.decision}")
    if verification_result.reasons:
        print("Reasons:", verification_result.reasons)
    if verification_result.warnings:
        print("Warnings:", verification_result.warnings)

    validation_result = None
    if verification_result.decision == "ALLOW":
        validation_result = human_validator.validate_plan(plan)
        audit_logger.log_validation(plan, validation_result)
        print(f"\nHuman validation: {validation_result.decision}")
        if validation_result.comments:
            print(f"Comments: {validation_result.comments}")
    else:
        print("\nPlan blocked by verification. Cannot proceed to human validation.")

    execution_report = None
    evaluation_result = None
    if verification_result.decision == "ALLOW" and validation_result and validation_result.decision == "APPROVE":
        execution_report = execution_manager.execute_plan(plan, verification_result, validation_result)
        evaluation_result = evaluator.evaluate(plan, execution_report)

        print("\nExecution:")
        for step_result in execution_report.step_results:
            print(f"Step {step_result.step_id} {step_result.skill_id}: {step_result.status}")

        print("\nEvaluation:")
        print(f"Task success: {str(evaluation_result['task_success']).lower()}")
        print(f"Summary: {evaluation_result['summary']}")

    if execution_report and audit_logger:
        audit_logger.log_execution_report(execution_report)


if __name__ == "__main__":
    main()
