"""LLM RobotPlan generation with RViz/fake-hardware simulated execution."""

import sys

from llm_skill_robot.agent.llm_client import (
    LLMClientConfigurationError,
    OpenAILLMClient,
    load_llm_config,
)
from llm_skill_robot.agent.llm_plan_generator import LLMPlanGenerator
from llm_skill_robot.agent.mock_plan_generator import PlanGenerationError
from llm_skill_robot.core.audit_logger import AuditLogger
from llm_skill_robot.core.execution_mode import is_execution_allowed, load_execution_config
from llm_skill_robot.core.human_validator import HumanValidator
from llm_skill_robot.core.plan_verifier import PlanVerifier
from llm_skill_robot.core.schemas import ValidationDecision, VerificationDecision
from llm_skill_robot.core.skill_library import SkillLibrary
from llm_skill_robot.core.trajectory_reviewer import TrajectoryReviewer
from llm_skill_robot.core.trajectory_validator import TrajectoryValidator
from llm_skill_robot.robot.robotiq_2f140_sim_backend import Robotiq2F140SimBackend
from llm_skill_robot.robot.ur5_moveit_plan_backend import UR5MoveItPlanBackend
from llm_skill_robot.safety.hardware_mode import load_hardware_mode, print_hardware_mode
from llm_skill_robot.perception.perception_factory import (
    create_perception_pipeline,
    load_perception_config,
    perception_context_from_config,
)
from llm_skill_robot.ros_nl_rviz_sim_demo import (
    OBJECT_RELATED_SKILL_IDS,
    PERCEPTION_SKILL_IDS,
    _audit_event,
    _audit_perception_result,
    _create_gripper_joint_state_sim,
    _load_gripper_config,
    _print_perception_step_result,
    _print_gripper_step_result,
    _print_execution_report,
    _print_generated_plan,
    _print_verification_result,
    _run_motion_step,
    _verify_object_step_for_execution,
    execute_non_motion_skill,
)
from llm_skill_robot.ros_rviz_sim_execution_demo import _load_safety_config

LLM_RVIZ_SIM_WARNING = "LLM RVIZ_SIM DEMO. Real robot execution is disabled."


def _available_skill_objects(skill_library: SkillLibrary) -> list:
    return [
        skill_library.get_skill(skill_id)
        for skill_id in skill_library.list_skills()
        if skill_library.get_skill(skill_id) is not None
    ]


def _create_llm_plan_generator():
    llm_config = load_llm_config().get("llm", {})
    llm_client = OpenAILLMClient(
        model=llm_config.get("model"),
        temperature=llm_config.get("temperature"),
        max_retries=llm_config.get("max_retries"),
        timeout_sec=llm_config.get("timeout_sec"),
    )
    return LLMPlanGenerator(llm_client)


def _generate_and_verify_plan(
    *,
    user_instruction: str,
    generator: LLMPlanGenerator,
    skill_library: SkillLibrary,
    plan_verifier: PlanVerifier,
    audit_logger: AuditLogger,
):
    available_skills = _available_skill_objects(skill_library)
    plan = generator.generate_plan(
        user_instruction=user_instruction,
        available_skills=available_skills,
    )
    _print_generated_plan(plan)
    _audit_event(
        audit_logger,
        "llm_plan_generated",
        {
            "decision": "GENERATED",
            "message": "LLMPlanGenerator produced a RobotPlan.",
            "plan": plan.model_dump(),
        },
    )

    verification_result = plan_verifier.verify_plan(
        plan,
        context={"skip_perception_scene_checks": True},
    )
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

    if verification_result.decision != VerificationDecision.BLOCK:
        return plan, verification_result

    choice = input("\nTry LLM repair? [y/N] ").strip().lower()
    _audit_event(
        audit_logger,
        "llm_repair_prompted",
        {
            "decision": "REQUESTED" if choice == "y" else "SKIPPED",
            "message": "LLM repair prompt completed.",
            "verification_result": verification_result.model_dump(),
        },
    )
    if choice != "y":
        return plan, verification_result

    repaired_plan = generator.repair_plan(
        user_instruction=user_instruction,
        available_skills=available_skills,
        previous_plan_or_text=plan,
        verification_result=verification_result,
    )
    _print_generated_plan(repaired_plan)
    repaired_verification = plan_verifier.verify_plan(
        repaired_plan,
        context={"skip_perception_scene_checks": True},
    )
    _print_verification_result(repaired_verification)
    _audit_event(
        audit_logger,
        "llm_plan_repaired",
        {
            "decision": repaired_verification.decision.value,
            "message": "LLM repair completed and repaired plan was verified.",
            "plan": repaired_plan.model_dump(),
            "verification_result": repaired_verification.model_dump(),
        },
    )
    return repaired_plan, repaired_verification


def _execute_plan_steps(
    *,
    plan,
    backend: UR5MoveItPlanBackend,
    node,
    execution_config,
    safety_config,
    audit_logger: AuditLogger,
) -> list:
    gripper_joint_state_sim = _create_gripper_joint_state_sim(node)
    gripper_backend = Robotiq2F140SimBackend(
        _load_gripper_config(),
        joint_state_sim=gripper_joint_state_sim,
    )
    skill_library = SkillLibrary()
    perception_config = load_perception_config()
    perception_pipeline = create_perception_pipeline(
        node=node,
        config=perception_config,
    )
    execution_context = {
        "scene": perception_pipeline.get_latest_scene(),
        "perception_available": False,
        **perception_context_from_config(perception_config),
    }
    trajectory_validator = TrajectoryValidator(safety_config)
    trajectory_reviewer = TrajectoryReviewer()
    step_results = []

    try:
        for step in plan.steps:
            if step.skill_id in OBJECT_RELATED_SKILL_IDS:
                result = _verify_object_step_for_execution(
                    step=step,
                    skill_library=skill_library,
                    execution_context=execution_context,
                )
                if result is None:
                    result = {
                        "success": False,
                        "step_id": step.step_id,
                        "skill_id": step.skill_id,
                        "parameters": dict(step.parameters),
                        "message": (
                            f"{step.skill_id} execution is not implemented in RViz sim "
                            "without a verified perception-backed object pose."
                        ),
                        "mode": "rviz_sim",
                        "output": {"mode": "rviz_sim"},
                    }
                _audit_event(
                    audit_logger,
                    "object_skill_verified",
                    {
                        "step_id": step.step_id,
                        "skill_id": step.skill_id,
                        "parameters": dict(step.parameters),
                        "decision": "SUCCESS" if result["success"] else "BLOCK",
                        "message": result["message"],
                        "result": result,
                    },
                )
            elif step.skill_id == "move_to_named_target":
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
                result = execute_non_motion_skill(
                    step,
                    gripper_backend,
                    context=execution_context,
                    perception_pipeline=perception_pipeline,
                )
                if step.skill_id in PERCEPTION_SKILL_IDS:
                    _print_perception_step_result(result)
                    _audit_perception_result(audit_logger, step, result)
                else:
                    _print_gripper_step_result(result)
                    _audit_event(
                        audit_logger,
                        "gripper_skill_executed",
                        {
                            "step_id": step.step_id,
                            "skill_id": step.skill_id,
                            "parameters": dict(step.parameters),
                            "decision": "SUCCESS" if result["success"] else "FAILED",
                            "message": result["message"],
                            "result": result,
                            "mode": "rviz_sim",
                            "real_gripper_enabled": False,
                        },
                    )
            step_results.append(result)
            if not result.get("success", False) and step.skill_id not in PERCEPTION_SKILL_IDS:
                break
    finally:
        if gripper_joint_state_sim is not None:
            gripper_joint_state_sim.stop()

    return step_results


def main() -> None:
    hardware_mode = load_hardware_mode()
    print_hardware_mode(hardware_mode)
    print(LLM_RVIZ_SIM_WARNING)
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
        node = rclpy.create_node("ros_llm_rviz_sim_demo")

        skill_library = SkillLibrary()
        safety_config = _load_safety_config()
        execution_config = load_execution_config().model_dump()
        execution_mode = "rviz_sim"
        _audit_event(
            audit_logger,
            "llm_rviz_sim_demo_started",
            {"message": LLM_RVIZ_SIM_WARNING},
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
        _audit_event(
            audit_logger,
            "user_instruction",
            {
                "message": "User instruction received for LLM RViz sim demo.",
                "instruction": user_instruction,
            },
        )

        generator = _create_llm_plan_generator()
        plan_verifier = PlanVerifier(skill_library)
        plan, verification_result = _generate_and_verify_plan(
            user_instruction=user_instruction,
            generator=generator,
            skill_library=skill_library,
            plan_verifier=plan_verifier,
            audit_logger=audit_logger,
        )

        if verification_result.decision != VerificationDecision.ALLOW:
            print(
                "\nPlan blocked by verification. "
                "No planning or execution requested."
            )
            return

        validation_result = HumanValidator().validate_plan(plan)
        print("\nHuman Validation:")
        print(
            "approved"
            if validation_result.decision == ValidationDecision.APPROVE
            else validation_result.decision.value.lower()
        )
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
            hardware_mode=hardware_mode,
        )
        step_results = _execute_plan_steps(
            plan=plan,
            backend=backend,
            node=node,
            execution_config=execution_config,
            safety_config=safety_config,
            audit_logger=audit_logger,
        )
        _print_execution_report(step_results)
        _audit_event(
            audit_logger,
            "llm_rviz_sim_execution_report",
            {
                "decision": (
                    "SUCCESS"
                    if all(r.get("success", False) for r in step_results)
                    else "FAILED"
                ),
                "message": "LLM RViz simulation demo completed.",
                "step_results": step_results,
            },
        )

    except LLMClientConfigurationError as e:
        print(f"\nLLM client is not configured: {e}")
        _audit_event(
            audit_logger,
            "llm_client_configuration_failed",
            {"decision": "FAILED", "message": str(e)},
        )
    except PlanGenerationError as e:
        print(f"\nLLM plan generation failed: {e}")
        _audit_event(
            audit_logger,
            "llm_plan_generation_failed",
            {"decision": "FAILED", "message": str(e)},
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
