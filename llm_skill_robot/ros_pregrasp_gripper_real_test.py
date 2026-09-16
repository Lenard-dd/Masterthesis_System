"""Real gripper width + restricted real UR5 pregrasp test without contact."""

from __future__ import annotations

from dataclasses import dataclass
import pprint
import sys
from typing import Any

from llm_skill_robot.core.audit_logger import AuditLogger
from llm_skill_robot.grasping.graspgenx_grasp_planner import GraspGenXGraspPlanner, generate_candidates_with_context
from llm_skill_robot.grasping.graspgenx_subprocess_adapter import GraspGenXSubprocessAdapter
from llm_skill_robot.perception.object_point_cloud_builder import ObjectPointCloudBuilder
from llm_skill_robot.perception.perception_factory import create_perception_pipeline
from llm_skill_robot.perception.tf_pose_transformer import TFPoseTransformer
from llm_skill_robot.robot.robotiq_2f140_real_backend import Robotiq2F140RealBackend
from llm_skill_robot.robot.ur5_moveit_plan_backend import UR5MoveItPlanBackend
from llm_skill_robot.ros_graspgenx_gripper_width_test import (
    _candidate_value,
    _load_yaml,
    _select_highest_confidence,
    extract_candidate_width,
)
from llm_skill_robot.ros_graspgenx_pregrasp_real_demo import (
    PregraspRealDemoController,
    print_stop_notice,
    wait_for_camera_messages,
)
from llm_skill_robot.ros_real_arm_safe_motion_test import (
    print_motion_summary,
    validate_execute_confirmation,
)
from llm_skill_robot.ros_real_gripper_test import confirmation_is_yes
from llm_skill_robot.safety.hardware_mode import (
    HardwareMode,
    assert_arm_execution_allowed,
    clamp_or_reject_gripper_command,
    load_hardware_mode,
    print_hardware_mode,
)
from llm_skill_robot.safety.real_arm_safety import (
    load_real_arm_safety,
    print_real_arm_safety,
)


@dataclass
class PregraspGripperResult:
    success: bool
    status: str
    message: str
    width_m: float | None = None


class PregraspGripperRealTestController:
    """Single candidate-width test combined with one restricted pregrasp motion."""

    def __init__(
        self,
        *,
        hardware_mode: HardwareMode,
        real_arm_safety,
        perception_pipeline,
        point_cloud_builder,
        grasp_planner,
        moveit_backend,
        gripper_backend,
        grasping_config: dict | None = None,
        gripper_config: dict | None = None,
        transformer=None,
        audit_logger: AuditLogger | None = None,
    ):
        self.hardware_mode = hardware_mode
        self.real_arm_safety = real_arm_safety
        self.perception_pipeline = perception_pipeline
        self.point_cloud_builder = point_cloud_builder
        self.grasp_planner = grasp_planner
        self.moveit_backend = moveit_backend
        self.gripper_backend = gripper_backend
        self.grasping_config = grasping_config or {}
        self.gripper_config = gripper_config or {}
        self.audit_logger = audit_logger
        self.pregrasp_controller = PregraspRealDemoController(
            hardware_mode=hardware_mode,
            real_arm_safety=real_arm_safety,
            perception_pipeline=perception_pipeline,
            point_cloud_builder=point_cloud_builder,
            grasp_planner=grasp_planner,
            moveit_backend=moveit_backend,
            gripper_backend=None,
            transformer=transformer,
            grasping_config=grasping_config,
            gripper_config=gripper_config,
            audit_logger=audit_logger,
        )

    def run_once(
        self,
        query: str,
        *,
        open_confirmation: str | None = "",
        execute_confirmation: str | None = "",
        set_width_confirmation: str | None = "",
    ) -> PregraspGripperResult:
        if not query.strip():
            return self._blocked("Object query must not be empty.")

        selected_object = self._detect_object(query)
        if selected_object is None:
            return self._blocked("No object detected.")

        point_cloud = self.pregrasp_controller._build_point_cloud(selected_object)
        if not point_cloud.get("success") or not point_cloud.get("point_cloud_path"):
            return self._blocked(
                f"Point-cloud generation failed: {point_cloud.get('message', 'unavailable')}"
            )

        generation = generate_candidates_with_context(
            self.grasp_planner, point_cloud, self.pregrasp_controller.transformer,
        )
        candidates = generation.get("candidates", [])
        if not candidates:
            return self._blocked(
                f"No GraspGenX candidates: {generation.get('message', 'none returned')}"
            )

        width_candidates = self._candidate_width_commands(candidates)
        if not width_candidates:
            reason = getattr(
                self,
                "_last_width_rejection_reason",
                "No candidate has a safe usable gripper_width_m.",
            )
            return self._blocked(reason)

        selection = self.pregrasp_controller.select_restricted_pregrasp_plan(
            [item["candidate"] for item in width_candidates],
            point_cloud,
        )
        if selection is None:
            return self._blocked("No candidate produced a safe reachable pregrasp.")

        candidate = selection.candidate
        width_command = next(
            item["command"]
            for item in width_candidates
            if item["candidate"] is candidate
        )
        width_m = width_command["width_m"]
        print_selected_candidate(candidate)

        if open_confirmation is None:
            open_confirmation = input("Open REAL gripper? Type YES:\n> ").strip()
        self._open_gripper_if_confirmed(open_confirmation)

        plan = selection.plan
        print_motion_summary(plan.summary)

        if execute_confirmation is None:
            execute_confirmation = input("Type EXECUTE to move REAL UR5 to pregrasp:\n> ")
        confirmation_check = validate_execute_confirmation(
            execute_confirmation,
            self.real_arm_safety.confirmation_phrase,
        )
        if not confirmation_check["allowed"]:
            self._audit(
                "real_arm_motion_rejected",
                plan_id=plan.plan_id,
                reason=confirmation_check["reason"],
                real_arm_execution_blocked=True,
            )
            return PregraspGripperResult(
                False,
                "CANCELLED",
                "EXECUTE confirmation was not provided. No real arm motion was executed.",
                width_m,
            )

        arm_result = self.moveit_backend.execute_cached_plan_real(plan.plan_id, final_confirmed=True)
        self._audit(
            "real_arm_motion_executed" if arm_result.get("success") else "real_arm_motion_failed",
            plan_id=plan.plan_id,
            execution_success=bool(arm_result.get("success")),
            message=arm_result.get("message"),
        )
        if not arm_result.get("success"):
            return PregraspGripperResult(
                False,
                "FAILED",
                str(arm_result.get("message", "Pregrasp execution failed.")),
                width_m,
            )

        if set_width_confirmation is None:
            set_width_confirmation = input(
                f"Set REAL gripper to candidate width {width_m:.4f} m? Type YES:\n> "
            ).strip()
        if not confirmation_is_yes(set_width_confirmation):
            return PregraspGripperResult(
                True,
                "ARM_ONLY",
                "Pregrasp reached. Candidate-width gripper command was not confirmed.",
                width_m,
            )

        gripper_result = self.gripper_backend.set_gripper_width(
            width_m=width_command["width_m"],
            speed_mps=width_command["speed_mps"],
            force_N=width_command["force_N"],
            confirmed=True,
        )
        self._audit(
            "real_gripper_candidate_width_set",
            real_gripper_command_requested=True,
            real_gripper_command_sent=bool(gripper_result.get("success")),
            width_m=width_m,
            message=gripper_result.get("message"),
        )
        if not gripper_result.get("success"):
            return PregraspGripperResult(
                False,
                "FAILED",
                str(gripper_result.get("message", "Gripper width command failed.")),
                width_m,
            )
        return PregraspGripperResult(
            True,
            "SUCCESS",
            "Pregrasp reached and candidate width command sent.",
            width_m,
        )

    def _detect_object(self, query: str):
        result = self.perception_pipeline.detect_object(query.strip(), require_pose=False)
        selected = _select_highest_confidence(result.get("objects", []))
        if selected is not None:
            self._audit("real_camera_used", real_camera_used=True)
        return selected

    def _open_gripper_if_confirmed(self, confirmation: str) -> None:
        if not confirmation_is_yes(confirmation):
            return
        if not self.real_arm_safety.gripper.get("allow_open_gripper", False):
            print("Real gripper open is disabled by real_arm_safety.yaml.")
            return
        speed = float(self.gripper_config.get("default_speed_mps", 0.02))
        force = float(self.gripper_config.get("default_force_N", 10.0))
        self._audit("real_gripper_command_requested", real_gripper_command_requested=True)
        result = self.gripper_backend.open_gripper(
            speed_mps=speed,
            force_N=force,
            confirmed=True,
        )
        self._audit(
            "real_gripper_opened_before_pregrasp",
            real_gripper_command_requested=True,
            real_gripper_command_sent=bool(result.get("success")),
            message=result.get("message"),
        )

    def _blocked(self, message: str) -> PregraspGripperResult:
        self._audit(
            "pregrasp_gripper_real_test_blocked",
            reason=message,
            real_arm_execution_blocked=True,
        )
        return PregraspGripperResult(False, "BLOCKED", message)

    def _audit(self, event: str, **fields: Any) -> None:
        if self.audit_logger is not None:
            self.audit_logger.log_hardware_safety_event(event, **fields)

    def _candidate_width_commands(self, candidates: list[Any]) -> list[dict[str, Any]]:
        usable = []
        rejected = []
        speed = float(self.gripper_config.get("default_speed_mps", 0.02))
        force = float(self.gripper_config.get("default_force_N", 10.0))
        valid_candidates = [
            candidate
            for candidate in candidates
            if _candidate_value(candidate, "valid", True) is not False
        ]
        valid_candidates = sorted(
            valid_candidates,
            key=lambda candidate: float(_candidate_value(candidate, "score", 0.0) or 0.0),
            reverse=True,
        )
        for candidate in valid_candidates:
            candidate_id = _candidate_value(candidate, "candidate_id", _candidate_value(candidate, "id"))
            try:
                command = validate_candidate_width_for_pregrasp_test(
                    candidate,
                    self.hardware_mode,
                    speed_mps=speed,
                    force_N=force,
                )
            except (ValueError, PermissionError) as exc:
                rejected.append({"candidate_id": candidate_id, "reason": str(exc)})
                continue
            usable.append({"candidate": candidate, "command": command})
        if rejected:
            print("\nCandidate width screening rejected:")
            for item in rejected[:20]:
                print(f"- candidate={item['candidate_id']} reason={item['reason']}")
            self._last_width_rejection_reason = rejected[-1]["reason"]
        self._audit(
            "pregrasp_gripper_width_screening",
            usable_count=len(usable),
            rejected=rejected,
        )
        return usable


def validate_candidate_width_for_pregrasp_test(
    candidate: Any,
    hardware_mode: HardwareMode,
    speed_mps: float,
    force_N: float,
) -> dict[str, float]:
    width_m = extract_candidate_width(candidate)
    safe_close_width = float(hardware_mode.gripper.safe_close_width_m)
    if width_m < safe_close_width:
        raise ValueError(
            f"gripper_width_m {width_m} is below safe_close_width_m {safe_close_width}."
        )
    return clamp_or_reject_gripper_command(
        width_m=width_m,
        speed_mps=speed_mps,
        force_N=force_N,
        mode=hardware_mode,
    )


def print_selected_candidate(candidate: Any) -> None:
    print("\nSelected GraspGenX candidate:")
    print(f"candidate_id: {_candidate_value(candidate, 'candidate_id', _candidate_value(candidate, 'id'))}")
    print(f"score: {_candidate_value(candidate, 'score')}")
    print(f"gripper_width_m: {_candidate_value(candidate, 'gripper_width_m')}")
    print(f"approach_vector: {_candidate_value(candidate, 'approach_vector')}")
    print(f"closing_axis: {_candidate_value(candidate, 'closing_axis')}")
    print(f"width_axis: {_candidate_value(candidate, 'width_axis')}")
    print(f"gripper_width_axis: {_candidate_value(candidate, 'gripper_width_axis')}")
    print("pose:")
    pprint.pprint(_candidate_value(candidate, "pose"), sort_dicts=False)


def main() -> None:
    try:
        import rclpy
    except ImportError:
        print("Error: rclpy not available. Please source ROS 2 setup.", file=sys.stderr)
        return

    try:
        hardware_mode = load_hardware_mode()
        real_arm_safety = load_real_arm_safety()
        perception_config = _load_yaml("perception.yaml")
        grasping_document = _load_yaml("grasping.yaml")
        gripper_document = _load_yaml("gripper.yaml")
        assert_arm_execution_allowed(hardware_mode)
    except Exception as exc:
        print(f"BLOCKED: failed to start pregrasp gripper test: {exc}")
        return

    print("REAL UR5 PREGRASP + REAL GRIPPER WIDTH TEST")
    print("This may move the REAL robot to pregrasp and command the REAL gripper.")
    print_hardware_mode(hardware_mode)
    print_real_arm_safety(real_arm_safety)

    node = None
    moveit_backend = None
    try:
        rclpy.init()
        node = rclpy.create_node("ros_pregrasp_gripper_real_test")
        perception_pipeline = create_perception_pipeline(node=node, config=perception_config)
        grasping_config = grasping_document.get("grasping", {})
        point_cloud_builder = ObjectPointCloudBuilder(grasping_config.get("point_cloud", {}))
        grasp_planner = GraspGenXGraspPlanner(
            GraspGenXSubprocessAdapter(grasping_config.get("graspgenx_subprocess", {}))
        )
        moveit_backend = UR5MoveItPlanBackend(
            node=node,
            dry_run=False,
            execution_mode="real_robot",
            hardware_mode=hardware_mode,
        )
        gripper_backend = Robotiq2F140RealBackend(
            node=node,
            config=gripper_document,
            hardware_mode=hardware_mode,
        )
        logger = AuditLogger("pregrasp_gripper_real_test_audit.log", hardware_mode=hardware_mode)
        controller = PregraspGripperRealTestController(
            hardware_mode=hardware_mode,
            real_arm_safety=real_arm_safety,
            perception_pipeline=perception_pipeline,
            point_cloud_builder=point_cloud_builder,
            grasp_planner=grasp_planner,
            moveit_backend=moveit_backend,
            gripper_backend=gripper_backend,
            grasping_config=grasping_config,
            gripper_config=gripper_document.get("gripper", {}),
            transformer=TFPoseTransformer(node, target_frame="base_link"),
            audit_logger=logger,
        )

        print("\nWaiting for D435i RGB/depth/camera_info messages...")
        wait_for_camera_messages(perception_pipeline.backend, node, rclpy)

        query = input("Enter object query:\n> ").strip()
        result = controller.run_once(
            query,
            open_confirmation=None,
            execute_confirmation=None,
            set_width_confirmation=None,
        )
        print(f"{result.status}: {result.message}")
    finally:
        print_stop_notice()
        if moveit_backend is not None:
            moveit_backend.shutdown()
        if node is not None:
            try:
                node.destroy_node()
            except Exception:
                pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
