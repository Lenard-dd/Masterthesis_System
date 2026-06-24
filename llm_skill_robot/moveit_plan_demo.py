#!/usr/bin/env python3
"""Plan-only MoveIt dry-run demo for UR5."""

from llm_skill_robot.core.audit_logger import AuditLogger
from llm_skill_robot.core.human_validator import HumanValidator
from llm_skill_robot.core.plan import PlanStep, RobotPlan
from llm_skill_robot.core.plan_verifier import PlanVerifier
from llm_skill_robot.core.schemas import ValidationDecision
from llm_skill_robot.core.skill_library import SkillLibrary
from llm_skill_robot.robot.ur5_moveit_plan_backend import UR5MoveItPlanBackend
from llm_skill_robot.safety.hardware_mode import load_hardware_mode, print_hardware_mode


def _create_demo_plan() -> RobotPlan:
    return RobotPlan(
        task="Plan-only MoveIt dry run to home",
        steps=[
            PlanStep(
                step_id="1",
                skill_id="move_to_named_target",
                description="Plan to named target home",
                parameters={
                    "target": "home",
                    "velocity_scale": 0.03,
                    "acceleration_scale": 0.03,
                },
            )
        ],
    )


def _print_generated_plan(plan: RobotPlan) -> None:
    print("Generated Plan:")
    for step in plan.steps:
        target = step.parameters.get("target") or step.parameters.get("target_name")
        print(f"{step.step_id}. {step.skill_id} target={target}")


def _print_verification(result) -> None:
    print("\nVerification:")
    print(result.decision.value if hasattr(result.decision, "value") else result.decision)
    if result.reasons:
        print(f"Reasons: {result.reasons}")
    if result.warnings:
        print(f"Warnings: {result.warnings}")


def _print_plan_preview(result: dict) -> None:
    summary = result["summary"]
    print("\nMoveIt Plan Preview:")
    print(f"Plan ID: {result['plan_id']}")
    print(f"Target: {summary.get('target_name')}")
    print(f"Planning group: {summary.get('planning_group')}")
    print(f"Trajectory points: {summary.get('num_trajectory_points')}")
    print("Status: PLAN_ONLY_DRY_RUN")


def main():
    hardware_mode = load_hardware_mode()
    print_hardware_mode(hardware_mode)
    skill_library = SkillLibrary()
    plan_verifier = PlanVerifier(skill_library)
    human_validator = HumanValidator()
    audit_logger = AuditLogger()
    plan_backend = UR5MoveItPlanBackend(dry_run=True, hardware_mode=hardware_mode)

    plan = _create_demo_plan()
    _print_generated_plan(plan)
    audit_logger.log_plan_generation(plan)

    verification_result = plan_verifier.verify_plan(plan)
    audit_logger.log_verification(plan, verification_result)
    _print_verification(verification_result)

    if verification_result.decision != "ALLOW":
        print("\nPlan blocked by verification. No motion plan generated.")
        return

    validation_result = human_validator.validate_plan(plan)
    audit_logger.log_validation(plan, validation_result)

    print("\nHuman Validation:")
    if validation_result.decision == ValidationDecision.APPROVE:
        print("approved")
    else:
        print(validation_result.decision.value.lower())

    if validation_result.decision != ValidationDecision.APPROVE:
        print("\nPlan was not approved. No motion plan generated.")
        return

    step = plan.steps[0]
    plan_result = plan_backend.plan_to_named_target(
        target=step.parameters["target"],
        velocity_scale=step.parameters.get("velocity_scale", 0.03),
        acceleration_scale=step.parameters.get("acceleration_scale", 0.03),
        skill_id=step.skill_id,
        step_id=int(step.step_id),
    )
    _print_plan_preview(plan_result)


if __name__ == "__main__":
    main()
