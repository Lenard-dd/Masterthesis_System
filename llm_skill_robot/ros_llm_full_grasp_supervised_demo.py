"""LLM-planned, fully supervised real UR5 grasp entry point.

The LLM may compose only the reviewed full-grasp skill plan.  Real hardware
execution is delegated to ``FullGraspSupervisedDemoController``, preserving its
state machine, candidate screening, trajectory review, and per-step prompts.
"""

from __future__ import annotations

import sys
from typing import Any

from llm_skill_robot.agent.llm_client import LLMClientConfigurationError
from llm_skill_robot.agent.mock_plan_generator import PlanGenerationError
from llm_skill_robot.core.audit_logger import AuditLogger
from llm_skill_robot.core.human_validator import HumanValidator
from llm_skill_robot.core.plan import RobotPlan
from llm_skill_robot.core.plan_verifier import PlanVerifier
from llm_skill_robot.core.schemas import ValidationDecision, VerificationDecision
from llm_skill_robot.core.skill_library import SkillLibrary
from llm_skill_robot.grasping.grasp_execution_backend import GraspExecutionBackend
from llm_skill_robot.grasping.grasp_verifier import GraspVerifier
from llm_skill_robot.grasping.graspgenx_grasp_planner import GraspGenXGraspPlanner
from llm_skill_robot.grasping.graspgenx_subprocess_adapter import GraspGenXSubprocessAdapter
from llm_skill_robot.perception.object_point_cloud_builder import ObjectPointCloudBuilder
from llm_skill_robot.perception.perception_factory import create_perception_pipeline
from llm_skill_robot.perception.tf_pose_transformer import TFPoseTransformer
from llm_skill_robot.robot.robotiq_2f140_real_backend import Robotiq2F140RealBackend
from llm_skill_robot.robot.ur5_moveit_plan_backend import UR5MoveItPlanBackend
from llm_skill_robot.ros_full_grasp_supervised_demo import (
    FullGraspSupervisedDemoController,
    _load_yaml,
    print_full_grasp_report,
    wait_for_grasp_camera_inputs,
)
from llm_skill_robot.ros_llm_rviz_sim_demo import (
    _available_skill_objects,
    _create_llm_plan_generator,
    _generate_and_verify_plan,
)
from llm_skill_robot.safety.hardware_mode import load_hardware_mode, print_hardware_mode
from llm_skill_robot.safety.real_arm_safety import (
    load_real_arm_safety,
    print_real_arm_safety,
)
from llm_skill_robot.safety.tabletop_safety import TabletopSafety


EXPECTED_REAL_FULL_GRASP_SEQUENCE = [
    "move_to_named_target",
    "detect_object",
    "build_object_point_cloud",
    "generate_grasp_pose",
    "review_grasp_candidate",
    "open_gripper",
    "move_to_pregrasp",
    "approach_grasp",
    "close_gripper",
    "retreat_grasp",
    "verify_grasp",
]


def validated_object_query(plan: RobotPlan) -> str:
    """Return the query only for the exact state-machine-compatible plan shape."""
    actual = [step.skill_id for step in plan.steps]
    if actual != EXPECTED_REAL_FULL_GRASP_SEQUENCE:
        raise ValueError(
            "Real LLM grasp execution requires the exact supervised sequence: "
            + ", ".join(EXPECTED_REAL_FULL_GRASP_SEQUENCE)
        )
    observe = plan.steps[0]
    observe_target = (
        observe.parameters.get("target_name") or observe.parameters.get("target")
    )
    if observe_target != "observe":
        raise ValueError("Object recognition must be preceded by the observe named target.")
    close = plan.steps[8]
    if close.parameters.get("during_contact") is not True:
        raise ValueError("Real contact close requires close_gripper during_contact=true.")
    query = plan.steps[1].parameters.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("detect_object requires a non-empty object query.")
    return query.strip()


def execute_approved_plan(
    plan: RobotPlan,
    controller: FullGraspSupervisedDemoController,
) -> dict[str, Any]:
    """Execute through the existing safe controller, never directly from the LLM."""
    return controller.run(validated_object_query(plan))


def _audit(logger: Any, event: str, **fields: Any) -> None:
    if hasattr(logger, "log_hardware_safety_event"):
        logger.log_hardware_safety_event(event, **fields)


def main() -> None:
    hardware_mode = load_hardware_mode()
    real_arm_safety = load_real_arm_safety()
    print("LLM FULL REAL GRASP SUPERVISED MODE")
    print("This can move the REAL UR5 and close the REAL Robotiq gripper.")
    print("Plan approval does not approve hardware actions.")
    print("Every arm motion still requires EXECUTE; every gripper action requires YES.")
    print_hardware_mode(hardware_mode)
    print_real_arm_safety(real_arm_safety)

    if (
        hardware_mode.mode != "real_full_grasp_supervised"
        or hardware_mode.arm.execution_mode != "full_grasp_supervised"
        or real_arm_safety.execution_mode != "full_grasp_supervised"
    ):
        print(
            "Error: real_full_grasp_supervised/full_grasp_supervised must be enabled.",
            file=sys.stderr,
        )
        return

    audit_logger = AuditLogger(
        "llm_full_grasp_supervised_audit.log", hardware_mode=hardware_mode
    )
    skill_library = SkillLibrary()
    try:
        instruction = input("\nEnter a REAL full-grasp task:\n> ").strip()
        generator = _create_llm_plan_generator(hardware_mode=hardware_mode)
        plan_verifier = PlanVerifier(
            skill_library, hardware_mode=hardware_mode, audit_logger=audit_logger
        )
        plan, verification = _generate_and_verify_plan(
            user_instruction=instruction,
            generator=generator,
            skill_library=skill_library,
            plan_verifier=plan_verifier,
            audit_logger=audit_logger,
        )
        if verification.decision != VerificationDecision.ALLOW:
            print("\nPlan verification blocked execution. No hardware action was requested.")
            return
        try:
            object_query = validated_object_query(plan)
        except ValueError as exc:
            print(f"\nPlan blocked: {exc}")
            _audit(audit_logger, "llm_step_blocked", reason=str(exc))
            return

        validation = HumanValidator().validate_plan(plan)
        _audit(
            audit_logger,
            "llm_real_full_grasp_plan_reviewed",
            accepted=validation.decision == ValidationDecision.APPROVE,
            object_query=object_query,
        )
        if validation.decision != ValidationDecision.APPROVE:
            print("\nPlan was not approved. No hardware action was requested.")
            return
    except (LLMClientConfigurationError, PlanGenerationError) as exc:
        print(f"\nLLM plan generation failed: {exc}", file=sys.stderr)
        _audit(audit_logger, "llm_plan_generation_failed", reason=str(exc))
        return

    try:
        import rclpy
    except ImportError:
        print("Error: rclpy is unavailable. Source the ROS 2 environment first.", file=sys.stderr)
        return

    node = None
    try:
        rclpy.init()
        node = rclpy.create_node("ros_llm_full_grasp_supervised_demo")
        perception_config = _load_yaml("perception.yaml")
        grasping_config = _load_yaml("grasping.yaml").get("grasping", {})
        gripper_config = _load_yaml("gripper.yaml")
        tabletop_safety = TabletopSafety.from_config(_load_yaml("tabletop_safety.yaml"))
        transformer = TFPoseTransformer(node, target_frame="base_link")
        perception = create_perception_pipeline(node=node, config=perception_config)

        print("\nWaiting for D435i RGB, aligned depth, and camera_info messages...")
        diagnostics = wait_for_grasp_camera_inputs(perception.backend, node, rclpy)
        if not diagnostics["ready"]:
            print(f"Camera inputs are not ready: {diagnostics}", file=sys.stderr)
            _audit(audit_logger, "llm_real_full_grasp_camera_not_ready", **diagnostics)
            return

        point_cloud_builder = ObjectPointCloudBuilder(
            grasping_config.get("point_cloud", {})
        )
        grasp_planner = GraspGenXGraspPlanner(
            GraspGenXSubprocessAdapter(
                grasping_config.get("graspgenx_subprocess", {})
            )
        )
        moveit = UR5MoveItPlanBackend(
            node=node,
            dry_run=False,
            execution_mode="real_robot",
            hardware_mode=hardware_mode,
        )
        motion = GraspExecutionBackend(
            moveit,
            real_arm_safety,
            audit_logger=audit_logger,
            tabletop_safety=tabletop_safety,
        )
        gripper = Robotiq2F140RealBackend(
            node=node,
            config=gripper_config,
            hardware_mode=hardware_mode,
            real_arm_safety=real_arm_safety,
            audit_logger=audit_logger,
        )
        controller = FullGraspSupervisedDemoController(
            hardware_mode=hardware_mode,
            real_arm_safety=real_arm_safety,
            perception_pipeline=perception,
            point_cloud_builder=point_cloud_builder,
            grasp_planner=grasp_planner,
            motion_backend=motion,
            gripper_backend=gripper,
            grasp_verifier=GraspVerifier(),
            grasping_config=grasping_config,
            transformer=transformer,
            audit_logger=audit_logger,
            tabletop_safety=tabletop_safety,
        )
        _audit(
            audit_logger,
            "llm_real_full_grasp_execution_started",
            object_query=object_query,
        )
        report = execute_approved_plan(plan, controller)
        print_full_grasp_report(report)
    except KeyboardInterrupt:
        print("\nInterrupted by operator. The sequence stopped.")
        _audit(audit_logger, "llm_real_full_grasp_interrupted")
    except Exception as exc:
        print(f"\nExecution stopped safely: {exc}", file=sys.stderr)
        _audit(audit_logger, "llm_real_full_grasp_failed", reason=str(exc))
    finally:
        print(f"\nAudit log: {audit_logger.log_file.resolve()}")
        if node is not None:
            node.destroy_node()
        if "rclpy" in locals() and rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
