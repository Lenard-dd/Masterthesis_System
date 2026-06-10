"""Natural-language RViz/fake-hardware simulated execution demo for UR5."""

from datetime import datetime, timezone
import sys
from typing import Any, Dict, List

from llm_skill_robot.agent.mock_plan_generator import MockPlanGenerator, PlanGenerationError
from llm_skill_robot.core.audit_logger import AuditLogger
from llm_skill_robot.core.execution_mode import is_execution_allowed, load_execution_config
from llm_skill_robot.core.human_validator import HumanValidator
from llm_skill_robot.core.plan import RobotPlan
from llm_skill_robot.core.plan_verifier import PlanVerifier
from llm_skill_robot.core.schemas import ValidationDecision
from llm_skill_robot.core.skill_library import SkillLibrary
from llm_skill_robot.core.trajectory_reviewer import TrajectoryReviewer
from llm_skill_robot.core.trajectory_validator import TrajectoryValidator
from llm_skill_robot.core.trajectory_visualizer import TrajectoryVisualizer
from llm_skill_robot.robot.mock_robot_backend import MockRobotBackend
from llm_skill_robot.robot.ur5_moveit_plan_backend import UR5MoveItPlanBackend
from llm_skill_robot.ros_rviz_sim_execution_demo import _load_safety_config

NL_RVIZ_SIM_WARNING = (
    "NATURAL LANGUAGE RVIZ_SIM DEMO. Real robot execution is disabled."
)


def _audit_event(audit_logger: AuditLogger, event: str, payload: Dict[str, Any]) -> None:
    if hasattr(audit_logger, "_write_entry"):
        audit_logger._write_entry(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event": event,
                "mode": "rviz_sim",
                "plan_id": None,
                "decision": None,
                "message": None,
                **payload,
            }
        )


def _print_generated_plan(plan: RobotPlan) -> None:
    print("\nGenerated Skill Plan:")
    for step in plan.steps:
        if step.skill_id == "move_to_named_target":
            target = step.parameters.get("target") or step.parameters.get("target_name")
            print(f"{step.step_id}. {step.skill_id} target={target}")
        else:
            print(f"{step.step_id}. {step.skill_id}")


def _print_verification_result(result) -> None:
    print("\nVerification:")
    print(result.decision.value)
    if result.reasons:
        print(f"Reasons: {result.reasons}")
    if result.warnings:
        print(f"Warnings: {result.warnings}")


def _print_motion_plan_summary(plan_result: dict) -> None:
    summary = plan_result["summary"]
    print("\nMotion Plan Summary:")
    print(f"Plan ID: {plan_result['plan_id']}")
    print(f"Target: {summary.get('target_name')}")
    print(f"Planning group: {summary.get('planning_group')}")
    print(f"Trajectory points: {summary.get('num_trajectory_points')}")
    print(f"Duration sec: {summary.get('duration_sec')}")
    print(f"Planning success: {str(plan_result['success']).lower()}")
    print(f"Message: {plan_result['message']}")


def _print_trajectory_validation(result: Dict[str, Any]) -> None:
    print("\nTrajectory Validation:")
    print(result["decision"])
    if result.get("reasons"):
        print(f"Reasons: {result['reasons']}")
    if result.get("warnings"):
        print(f"Warnings: {result['warnings']}")


def _execute_gripper_step(step, mock_backend: MockRobotBackend) -> Dict[str, Any]:
    if step.skill_id == "open_gripper":
        result = mock_backend.open_gripper()
    elif step.skill_id == "close_gripper":
        result = mock_backend.close_gripper()
    else:
        result = {
            "success": False,
            "message": f"Unsupported non-motion skill in this demo: {step.skill_id}",
            "output": {},
        }
    return {
        "step_id": step.step_id,
        "skill_id": step.skill_id,
        "mode": "mock_sim",
        **result,
    }


def _run_motion_step(
    *,
    step,
    backend: UR5MoveItPlanBackend,
    node,
    execution_config: Dict[str, Any],
    trajectory_validator: TrajectoryValidator,
    trajectory_reviewer: TrajectoryReviewer,
    audit_logger: AuditLogger,
) -> Dict[str, Any]:
    target = step.parameters.get("target") or step.parameters.get("target_name")
    plan_result = backend.plan_to_named_target(
        target=target,
        velocity_scale=step.parameters.get("velocity_scale", 0.03),
        acceleration_scale=step.parameters.get("acceleration_scale", 0.03),
        skill_id=step.skill_id,
        step_id=int(step.step_id),
    )
    _print_motion_plan_summary(plan_result)
    _audit_event(
        audit_logger,
        "motion_planned",
        {
            "plan_id": plan_result["plan_id"],
            "step_id": step.step_id,
            "decision": "SUCCESS" if plan_result["success"] else "FAILED",
            "message": plan_result["message"],
            "summary": plan_result["summary"],
        },
    )

    trajectory_result = trajectory_validator.validate_motion_plan_summary(
        plan_result["summary"]
    )
    _print_trajectory_validation(trajectory_result)
    _audit_event(
        audit_logger,
        "trajectory_validated",
        {
            "plan_id": plan_result["plan_id"],
            "step_id": step.step_id,
            "decision": trajectory_result["decision"],
            "message": "Trajectory validation completed.",
            "reasons": trajectory_result.get("reasons", []),
            "warnings": trajectory_result.get("warnings", []),
        },
    )
    if trajectory_result["decision"] != "ALLOW":
        return {
            "success": False,
            "step_id": step.step_id,
            "skill_id": step.skill_id,
            "message": "Trajectory validation blocked RViz/fake hardware execution.",
        }

    print("\nTrajectory Review:")
    review_approved = trajectory_reviewer.ask_for_review_approval(plan_result["summary"])
    _audit_event(
        audit_logger,
        "trajectory_reviewed",
        {
            "plan_id": plan_result["plan_id"],
            "step_id": step.step_id,
            "decision": "APPROVE" if review_approved else "REJECT",
            "message": "Trajectory review approved." if review_approved else "Trajectory review rejected.",
        },
    )
    if not review_approved:
        return {
            "success": False,
            "step_id": step.step_id,
            "skill_id": step.skill_id,
            "message": "Trajectory review was not approved.",
        }

    print("\nVisual Trajectory Preview:")
    preview_topic = execution_config.get("visual_preview_topic", "/display_planned_path")
    trajectory_visualizer = TrajectoryVisualizer(node=node, topic=preview_topic)
    cached_plan = backend.get_plan(plan_result["plan_id"])
    preview_result = trajectory_visualizer.publish_preview(cached_plan)
    _audit_event(
        audit_logger,
        "visual_preview_published",
        {
            "plan_id": plan_result["plan_id"],
            "step_id": step.step_id,
            "decision": "PUBLISHED" if preview_result["success"] else "FAILED",
            "message": preview_result["message"],
            "preview_result": preview_result,
        },
    )
    if preview_result["success"]:
        print(preview_result["message"])
    else:
        print("No raw trajectory available. Please rely on textual trajectory review and MoveIt/RViz planning display.")
    print("Please inspect RViz. No motion has been executed yet.")

    print("\nFinal Confirmation:")
    choice = input("\nExecute this plan in RViz/fake hardware only? [y/N] ").strip()
    final_confirmed = choice.lower() == "y"
    _audit_event(
        audit_logger,
        "final_confirmed",
        {
            "plan_id": plan_result["plan_id"],
            "step_id": step.step_id,
            "decision": "CONFIRMED" if final_confirmed else "DECLINED",
            "message": "Final confirmation provided." if final_confirmed else "Final confirmation not provided.",
            "confirmed": final_confirmed,
        },
    )
    if not final_confirmed:
        return {
            "success": False,
            "step_id": step.step_id,
            "skill_id": step.skill_id,
            "message": "Final confirmation was not provided.",
        }

    execution_result = backend.execute_cached_plan_simulated(
        plan_result["plan_id"],
        final_confirmed=True,
        trajectory_validation_allowed=trajectory_result["decision"] == "ALLOW",
        trajectory_review_approved=review_approved,
    )
    _audit_event(
        audit_logger,
        "simulated_execution_result",
        {
            "plan_id": execution_result["plan_id"],
            "step_id": step.step_id,
            "decision": "SUCCESS" if execution_result["success"] else "FAILED",
            "message": execution_result["message"],
            "execution_result": execution_result,
        },
    )
    return {
        "step_id": step.step_id,
        "skill_id": step.skill_id,
        **execution_result,
    }


def _print_execution_report(step_results: List[Dict[str, Any]]) -> None:
    success = all(result.get("success", False) for result in step_results)
    print("\nExecution Report:")
    print("SUCCESS" if success else "FAILED")
    for result in step_results:
        print(
            f"Step {result.get('step_id')} {result.get('skill_id')}: "
            f"{'SUCCESS' if result.get('success') else 'FAILED'} - {result.get('message')}"
        )


def main() -> None:
    print(NL_RVIZ_SIM_WARNING)
    audit_logger = AuditLogger()

    try:
        import rclpy
    except ImportError:
        print("Error: rclpy not available. Please source ROS 2 setup.", file=sys.stderr)
        print(f"\nAudit log: {audit_logger.log_file.resolve()}")
        sys.exit(1)

    backend = None
    node = None
    try:
        rclpy.init()
        node = rclpy.create_node("ros_nl_rviz_sim_demo")

        skill_library = SkillLibrary()
        safety_config = _load_safety_config()
        execution_config = load_execution_config().model_dump()
        execution_mode = "rviz_sim"
        _audit_event(
            audit_logger,
            "nl_rviz_sim_demo_started",
            {"message": NL_RVIZ_SIM_WARNING},
        )

        if not is_execution_allowed(execution_mode, execution_config):
            print("\nExecution config does not allow rviz_sim mode. Exiting safely.")
            _audit_event(
                audit_logger,
                "rviz_sim_execution_not_allowed",
                {
                    "decision": "BLOCK",
                    "message": "Execution config does not allow rviz_sim mode.",
                },
            )
            return
        if execution_config.get("allow_real_robot_execution", False):
            print("\nreal_robot execution is enabled in config. Exiting safely.")
            _audit_event(
                audit_logger,
                "real_robot_execution_config_blocked_demo",
                {
                    "decision": "BLOCK",
                    "message": "real_robot execution is enabled in config.",
                },
            )
            return

        user_instruction = input("\nEnter a robot task:\n> ").strip()
        plan = MockPlanGenerator().generate_plan(
            user_instruction=user_instruction,
            available_skills=skill_library.list_skills(),
        )
        _print_generated_plan(plan)
        _audit_event(
            audit_logger,
            "plan_generated",
            {
                "decision": "GENERATED",
                "message": "MockPlanGenerator produced a RobotPlan.",
                "plan": plan.model_dump(),
            },
        )

        verification_result = PlanVerifier(skill_library).verify_plan(plan)
        _print_verification_result(verification_result)
        _audit_event(
            audit_logger,
            "plan_verified",
            {
                "decision": verification_result.decision.value,
                "message": "Plan verification completed.",
                "reasons": verification_result.reasons,
                "warnings": verification_result.warnings,
            },
        )
        if verification_result.decision.value != "ALLOW":
            print("\nPlan blocked by verification. No planning or execution requested.")
            return

        validation_result = HumanValidator().validate_plan(plan)
        print("\nHuman Validation:")
        print("approved" if validation_result.decision == ValidationDecision.APPROVE else validation_result.decision.value.lower())
        _audit_event(
            audit_logger,
            "human_validated",
            {
                "decision": validation_result.decision.value,
                "message": "Human validation completed.",
                "comments": validation_result.comments,
            },
        )
        if validation_result.decision != ValidationDecision.APPROVE:
            print("\nPlan was not approved. No planning or execution requested.")
            return

        backend = UR5MoveItPlanBackend(
            node=node,
            dry_run=False,
            mode=execution_mode,
            execution_config=execution_config,
        )
        mock_backend = MockRobotBackend(known_named_targets=["home", "observe", "pickup", "place"])
        trajectory_validator = TrajectoryValidator(safety_config)
        trajectory_reviewer = TrajectoryReviewer()
        step_results = []

        for step in plan.steps:
            if step.skill_id == "move_to_named_target":
                result = _run_motion_step(
                    step=step,
                    backend=backend,
                    node=node,
                    execution_config=execution_config,
                    trajectory_validator=trajectory_validator,
                    trajectory_reviewer=trajectory_reviewer,
                    audit_logger=audit_logger,
                )
            else:
                result = _execute_gripper_step(step, mock_backend)
                print(f"\nSimulated Non-Motion Step:")
                print(f"Step {step.step_id}: {result['message']}")
                _audit_event(
                    audit_logger,
                    "mock_step_executed",
                    {
                        "step_id": step.step_id,
                        "skill_id": step.skill_id,
                        "decision": "SUCCESS" if result["success"] else "FAILED",
                        "message": result["message"],
                        "result": result,
                    },
                )
            step_results.append(result)
            if not result.get("success", False):
                break

        _print_execution_report(step_results)
        _audit_event(
            audit_logger,
            "nl_rviz_sim_execution_report",
            {
                "decision": "SUCCESS" if all(r.get("success", False) for r in step_results) else "FAILED",
                "message": "Natural-language RViz simulation demo completed.",
                "step_results": step_results,
            },
        )

    except PlanGenerationError as e:
        print(f"\nPlan generation failed: {e}")
        _audit_event(
            audit_logger,
            "plan_generation_failed",
            {
                "decision": "FAILED",
                "message": str(e),
            },
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        print(f"\nAudit log: {audit_logger.log_file.resolve()}")
        if backend is not None:
            backend.shutdown()
        if node is not None:
            node.destroy_node()
        if "rclpy" in locals() and rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
