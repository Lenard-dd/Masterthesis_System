"""ROS 2 MoveIt plan-only demo for UR5."""

import sys

from llm_skill_robot.core.audit_logger import AuditLogger
from llm_skill_robot.core.human_validator import HumanValidator
from llm_skill_robot.core.plan import PlanStep, RobotPlan
from llm_skill_robot.core.plan_verifier import PlanVerifier
from llm_skill_robot.core.schemas import ValidationDecision
from llm_skill_robot.core.skill_library import SkillLibrary
from llm_skill_robot.robot.ur5_moveit_plan_backend import UR5MoveItPlanBackend


def _create_plan() -> RobotPlan:
    return RobotPlan(
        task="Plan-only MoveIt request to home",
        steps=[
            PlanStep(
                step_id="1",
                skill_id="move_to_named_target",
                description="Plan to named target home",
                parameters={
                    "target": "home",
                    "velocity_scale": 0.1,
                    "acceleration_scale": 0.1,
                },
            )
        ],
    )


def _print_plan(plan: RobotPlan) -> None:
    print("Generated Plan:")
    for step in plan.steps:
        target = step.parameters.get("target") or step.parameters.get("target_name")
        print(f"{step.step_id}. {step.skill_id} target={target}")


def _print_plan_summary(result: dict) -> None:
    summary = result["summary"]
    print("\nMoveIt Plan Preview:")
    print(f"Plan ID: {result['plan_id']}")
    print(f"Target: {summary.get('target_name')}")
    print(f"Planning group: {summary.get('planning_group')}")
    print(f"Trajectory points: {summary.get('num_trajectory_points')}")
    print(f"Success: {str(result['success']).lower()}")
    print(f"Message: {result['message']}")
    print("Status: PLAN_ONLY")


def main():
    print("PLAN-ONLY MODE: this demo will not execute robot motion.")

    try:
        import rclpy
    except ImportError:
        print("Error: rclpy not available. Please source ROS 2 setup.", file=sys.stderr)
        sys.exit(1)

    backend = None
    node = None
    try:
        rclpy.init()
        node = rclpy.create_node("ros_moveit_plan_only_demo")

        skill_library = SkillLibrary()
        plan_verifier = PlanVerifier(skill_library)
        human_validator = HumanValidator()
        audit_logger = AuditLogger()
        backend = UR5MoveItPlanBackend(node=node, dry_run=False)

        plan = _create_plan()
        _print_plan(plan)
        audit_logger.log_plan_generation(plan)

        verification_result = plan_verifier.verify_plan(plan)
        audit_logger.log_verification(plan, verification_result)

        print("\nVerification:")
        print(verification_result.decision.value)
        if verification_result.reasons:
            print(f"Reasons: {verification_result.reasons}")
        if verification_result.warnings:
            print(f"Warnings: {verification_result.warnings}")

        if verification_result.decision != "ALLOW":
            print("\nPlan blocked by verification. No MoveIt plan requested.")
            return

        validation_result = human_validator.validate_plan(plan)
        audit_logger.log_validation(plan, validation_result)

        print("\nHuman Validation:")
        print("approved" if validation_result.decision == ValidationDecision.APPROVE else validation_result.decision.value.lower())

        if validation_result.decision != ValidationDecision.APPROVE:
            print("\nPlan was not approved. No MoveIt plan requested.")
            return

        print("\nPLAN-ONLY MODE: this demo will not execute robot motion.")
        step = plan.steps[0]
        plan_result = backend.plan_to_named_target(
            target=step.parameters["target"],
            velocity_scale=step.parameters.get("velocity_scale", 0.1),
            acceleration_scale=step.parameters.get("acceleration_scale", 0.1),
            skill_id=step.skill_id,
            step_id=int(step.step_id),
        )
        _print_plan_summary(plan_result)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    finally:
        if backend is not None:
            backend.shutdown()
        if node is not None:
            node.destroy_node()
        if "rclpy" in locals() and rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
