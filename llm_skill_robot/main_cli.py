#!/usr/bin/env python3
"""Main CLI for LLM Skill Robot demo."""

from llm_skill_robot.core.skill_library import SkillLibrary
from llm_skill_robot.core.plan import RobotPlan, PlanStep
from llm_skill_robot.core.plan_verifier import PlanVerifier
from llm_skill_robot.core.human_validator import HumanValidator
from llm_skill_robot.core.audit_logger import AuditLogger


def main():
    # Initialize components
    skill_library = SkillLibrary()
    plan_verifier = PlanVerifier(skill_library)
    human_validator = HumanValidator()
    audit_logger = AuditLogger()

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
                skill_id="move_to_named_target",
                description="Move to place position",
                parameters={"target_name": "place"}
            ),
            PlanStep(
                step_id="5",
                skill_id="place_object",
                description="Place the object",
                parameters={}
            )
        ]
    )

    # Log plan generation
    audit_logger.log_plan_generation(plan)

    # Verify the plan
    verification_result = plan_verifier.verify_plan(plan)
    audit_logger.log_verification(plan, verification_result)

    print(f"Plan verification: {verification_result.decision}")
    if verification_result.reasons:
        print("Reasons:", verification_result.reasons)
    if verification_result.warnings:
        print("Warnings:", verification_result.warnings)

    if verification_result.decision == "ALLOW":
        # Present to human validator
        validation_result = human_validator.validate_plan(plan)
        audit_logger.log_validation(plan, validation_result)

        print(f"Human validation: {validation_result.decision}")
        if validation_result.comments:
            print(f"Comments: {validation_result.comments}")
    else:
        print("Plan blocked by verification. Cannot proceed to human validation.")


if __name__ == "__main__":
    main()