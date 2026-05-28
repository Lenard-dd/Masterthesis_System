"""ROS 2 RViz/fake-hardware simulated execution demo for UR5."""

import sys
from typing import Any, Dict

import yaml

from llm_skill_robot.core.audit_logger import AuditLogger
from llm_skill_robot.core.execution_mode import is_execution_allowed, load_execution_config
from llm_skill_robot.core.human_validator import HumanValidator
from llm_skill_robot.core.plan import PlanStep, RobotPlan
from llm_skill_robot.core.plan_verifier import PlanVerifier
from llm_skill_robot.core.schemas import ValidationDecision
from llm_skill_robot.core.skill_library import SkillLibrary
from llm_skill_robot.core.trajectory_reviewer import TrajectoryReviewer
from llm_skill_robot.core.trajectory_validator import TrajectoryValidator
from llm_skill_robot.core.trajectory_visualizer import TrajectoryVisualizer
from llm_skill_robot.robot.ur5_moveit_plan_backend import UR5MoveItPlanBackend
from llm_skill_robot.utils import get_config_dir

RVIZ_SIM_MODE_WARNING = (
    "RVIZ_SIM MODE ONLY. This demo is not intended for real robot execution."
)


def _load_safety_config() -> Dict[str, Any]:
    safety_file = get_config_dir() / "safety.yaml"
    if not safety_file.exists():
        return {}
    with open(safety_file, "r") as f:
        return yaml.safe_load(f) or {}


def _create_plan() -> RobotPlan:
    return RobotPlan(
        task="RViz simulated execution to home",
        steps=[
            PlanStep(
                step_id="1",
                skill_id="move_to_named_target",
                description="Plan and simulate moving to home",
                parameters={
                    "target": "home",
                    "velocity_scale": 0.03,
                    "acceleration_scale": 0.03,
                },
            )
        ],
    )


def _print_plan(plan: RobotPlan) -> None:
    print("\nGenerated Plan:")
    for step in plan.steps:
        target = step.parameters.get("target") or step.parameters.get("target_name")
        print(f"{step.step_id}. {step.skill_id} target={target}")


def _print_motion_plan_summary(plan_result: dict) -> None:
    summary = plan_result["summary"]
    print("\nMotion Plan Summary:")
    print(f"Plan ID: {plan_result['plan_id']}")
    print(f"Target: {summary.get('target_name')}")
    print(f"Planning group: {summary.get('planning_group')}")
    print(f"Trajectory points: {summary.get('num_trajectory_points')}")
    print(f"Duration sec: {summary.get('duration_sec')}")
    print(f"Max joint delta rad: {summary.get('max_joint_delta_rad')}")
    print(f"Max velocity abs: {summary.get('max_velocity_abs')}")
    print(f"Planning success: {str(plan_result['success']).lower()}")
    print(f"Message: {plan_result['message']}")


def _print_validation_result(label: str, result) -> None:
    print(f"\n{label}:")
    decision = result["decision"] if isinstance(result, dict) else result.decision.value
    print(decision)
    reasons = result.get("reasons", []) if isinstance(result, dict) else result.reasons
    warnings = result.get("warnings", []) if isinstance(result, dict) else result.warnings
    if reasons:
        print(f"Reasons: {reasons}")
    if warnings:
        print(f"Warnings: {warnings}")


def _audit_event(audit_logger: AuditLogger, event: str, payload: Dict[str, Any]) -> None:
    if hasattr(audit_logger, "_write_entry"):
        audit_logger._write_entry({"event": event, **payload})


def _decision_value(decision) -> str:
    return getattr(decision, "value", decision)


def main():
    print(RVIZ_SIM_MODE_WARNING)

    try:
        import rclpy
    except ImportError:
        print("Error: rclpy not available. Please source ROS 2 setup.", file=sys.stderr)
        sys.exit(1)

    backend = None
    node = None
    try:
        rclpy.init()
        node = rclpy.create_node("ros_rviz_sim_execution_demo")

        skill_library = SkillLibrary()
        safety_config = _load_safety_config()
        execution_config = load_execution_config().model_dump()
        execution_mode = "rviz_sim"
        audit_logger = AuditLogger()
        _audit_event(
            audit_logger,
            "rviz_sim_demo_started",
            {
                "mode": execution_mode,
                "message": RVIZ_SIM_MODE_WARNING,
            },
        )
        _audit_event(
            audit_logger,
            "configs_and_skills_loaded",
            {
                "execution_config": execution_config,
                "safety_config": safety_config,
                "skill_count": len(getattr(skill_library, "skills", {})),
            },
        )

        if not is_execution_allowed(execution_mode, execution_config):
            print("\nExecution config does not allow rviz_sim mode. Exiting safely.")
            _audit_event(
                audit_logger,
                "rviz_sim_execution_not_allowed",
                {"mode": execution_mode, "execution_config": execution_config},
            )
            return

        if execution_config.get("allow_real_robot_execution", False):
            print("\nreal_robot execution is enabled in config. Exiting safely.")
            _audit_event(
                audit_logger,
                "real_robot_execution_config_blocked_demo",
                {"allow_real_robot_execution": True},
            )
            return

        plan_verifier = PlanVerifier(skill_library)
        human_validator = HumanValidator()
        trajectory_validator = TrajectoryValidator(safety_config)
        trajectory_reviewer = TrajectoryReviewer()

        plan = _create_plan()
        _print_plan(plan)
        audit_logger.log_plan_generation(plan)

        verification_result = plan_verifier.verify_plan(plan)
        audit_logger.log_verification(plan, verification_result)
        _print_validation_result("Verification", verification_result)
        if _decision_value(verification_result.decision) == "BLOCK":
            print("\nPlan blocked by verification. No planning or execution requested.")
            _audit_event(
                audit_logger,
                "rviz_sim_demo_stopped",
                {
                    "reason": "verification_blocked",
                    "verification_result": verification_result.model_dump(),
                },
            )
            return

        validation_result = human_validator.validate_plan(plan)
        audit_logger.log_validation(plan, validation_result)
        print("\nHuman Validation:")
        print("approved" if validation_result.decision == ValidationDecision.APPROVE else validation_result.decision.value.lower())
        if validation_result.decision != ValidationDecision.APPROVE:
            print("\nPlan was not approved. No planning or execution requested.")
            _audit_event(
                audit_logger,
                "rviz_sim_demo_stopped",
                {
                    "reason": "human_validation_not_approved",
                    "validation_result": validation_result.model_dump(),
                },
            )
            return

        backend = UR5MoveItPlanBackend(
            node=node,
            dry_run=False,
            mode=execution_mode,
            execution_config=execution_config,
        )
        step = plan.steps[0]
        plan_result = backend.plan_to_named_target(
            target=step.parameters["target"],
            velocity_scale=step.parameters.get("velocity_scale", 0.03),
            acceleration_scale=step.parameters.get("acceleration_scale", 0.03),
            skill_id=step.skill_id,
            step_id=int(step.step_id),
        )
        _print_motion_plan_summary(plan_result)

        trajectory_result = trajectory_validator.validate_motion_plan_summary(
            plan_result["summary"]
        )
        _print_validation_result("Trajectory Validation", trajectory_result)
        _audit_event(
            audit_logger,
            "trajectory_validated",
            {
                "plan_id": plan_result["plan_id"],
                "trajectory_validation": trajectory_result,
            },
        )

        if trajectory_result["decision"] != "ALLOW":
            print("\nTrajectory validation blocked simulated execution.")
            _audit_event(
                audit_logger,
                "rviz_sim_demo_stopped",
                {
                    "reason": "trajectory_validation_blocked",
                    "plan_id": plan_result["plan_id"],
                    "trajectory_validation": trajectory_result,
                },
            )
            return

        _audit_event(
            audit_logger,
            "trajectory_review_shown",
            {"plan_id": plan_result["plan_id"]},
        )
        review_approved = trajectory_reviewer.ask_for_review_approval(plan_result["summary"])
        _audit_event(
            audit_logger,
            "trajectory_review_approved",
            {
                "plan_id": plan_result["plan_id"],
                "approved": review_approved,
            },
        )
        if not review_approved:
            print("\nTrajectory review was not approved. No execution requested.")
            _audit_event(
                audit_logger,
                "rviz_sim_demo_stopped",
                {
                    "reason": "trajectory_review_not_approved",
                    "plan_id": plan_result["plan_id"],
                },
            )
            return

        print("\nVisual Trajectory Preview:")
        if execution_config.get("enable_visual_trajectory_preview", True):
            preview_topic = execution_config.get(
                "visual_preview_topic",
                "/display_planned_path",
            )
            trajectory_visualizer = TrajectoryVisualizer(node=node, topic=preview_topic)
            cached_plan = backend.get_plan(plan_result["plan_id"])
            preview_result = trajectory_visualizer.publish_preview(cached_plan)
            _audit_event(
                audit_logger,
                "visual_trajectory_preview",
                {
                    "plan_id": plan_result["plan_id"],
                    "preview_result": preview_result,
                },
            )

            if preview_result["success"]:
                print(preview_result["message"])
            else:
                print("No raw trajectory available. Please rely on textual trajectory review and MoveIt/RViz planning display.")
        else:
            print("Visual trajectory preview is disabled by execution config.")
            _audit_event(
                audit_logger,
                "visual_trajectory_preview",
                {
                    "plan_id": plan_result["plan_id"],
                    "preview_result": {
                        "success": False,
                        "message": "Visual trajectory preview is disabled by execution config.",
                    },
                },
            )

        print("Please inspect RViz. No motion has been executed yet.")

        print("\nFinal Confirmation:")
        choice = input("\nExecute this plan in RViz/fake hardware only? [y/N] ").strip()
        final_confirmed = choice.lower() == "y"
        _audit_event(
            audit_logger,
            "final_execution_confirmed",
            {
                "plan_id": plan_result["plan_id"],
                "confirmed": final_confirmed,
            },
        )
        if not final_confirmed:
            print("\nFinal confirmation was not provided. No execution requested.")
            _audit_event(
                audit_logger,
                "rviz_sim_demo_stopped",
                {
                    "reason": "final_confirmation_not_provided",
                    "plan_id": plan_result["plan_id"],
                },
            )
            return

        execution_result = backend.execute_cached_plan_simulated(
            plan_result["plan_id"],
            final_confirmed=True,
            trajectory_validation_allowed=trajectory_result["decision"] == "ALLOW",
            trajectory_review_approved=review_approved,
        )
        _audit_event(audit_logger, "rviz_sim_execution_result", execution_result)

        print("\nExecution Report:")
        print(f"Plan ID: {execution_result['plan_id']}")
        print(f"Mode: {execution_result['mode']}")
        print(f"Success: {str(execution_result['success']).lower()}")
        print(f"Message: {execution_result['message']}")

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
