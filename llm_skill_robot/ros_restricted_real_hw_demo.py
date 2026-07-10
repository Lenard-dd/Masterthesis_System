"""Unified restricted real-hardware demo for D435i, GraspGenX, UR5, and gripper."""

from __future__ import annotations

from dataclasses import dataclass, field
import pprint
import sys
from typing import Any

from llm_skill_robot.core.audit_logger import AuditLogger
from llm_skill_robot.grasping.graspgenx_grasp_planner import GraspGenXGraspPlanner
from llm_skill_robot.grasping.graspgenx_subprocess_adapter import GraspGenXSubprocessAdapter
from llm_skill_robot.perception.object_point_cloud_builder import ObjectPointCloudBuilder
from llm_skill_robot.perception.perception_factory import create_perception_pipeline
from llm_skill_robot.perception.tf_pose_transformer import TFPoseTransformer
from llm_skill_robot.robot.robotiq_2f140_real_backend import Robotiq2F140RealBackend
from llm_skill_robot.robot.ur5_moveit_plan_backend import UR5MoveItPlanBackend
from llm_skill_robot.ros_check_real_arm import run_status_checks, print_status_report
from llm_skill_robot.ros_graspgenx_gripper_width_test import (
    _candidate_value,
    _load_yaml,
    _select_highest_confidence,
)
from llm_skill_robot.ros_graspgenx_pregrasp_real_demo import (
    PregraspRealDemoController,
    PregraspSelection,
    print_stop_notice,
    wait_for_camera_messages,
)
from llm_skill_robot.ros_pregrasp_gripper_real_test import (
    print_selected_candidate,
    validate_candidate_width_for_pregrasp_test,
)
from llm_skill_robot.ros_real_arm_safe_motion_test import (
    LastMotionPlan,
    execute_last_plan_for_target,
    plan_named_target,
    print_motion_summary,
    validate_execute_confirmation,
    validate_execution_precheck,
)
from llm_skill_robot.ros_real_gripper_test import confirmation_is_yes
from llm_skill_robot.safety.hardware_mode import (
    HardwareMode,
    assert_arm_execution_allowed,
    load_hardware_mode,
    print_hardware_mode,
)
from llm_skill_robot.safety.real_arm_safety import (
    RealArmSafetyConfig,
    load_real_arm_safety,
    print_real_arm_safety,
)


FORBIDDEN_MENU_TERMS = ("approach_grasp", "close_gripper", "lift_object", "pick_object")


@dataclass
class DemoActionResult:
    success: bool
    status: str
    message: str
    output: dict[str, Any] = field(default_factory=dict)


@dataclass
class RestrictedRealDemoState:
    object_query: str | None = None
    selected_object: Any | None = None
    point_cloud: dict[str, Any] | None = None
    candidates: list[Any] = field(default_factory=list)
    selected_grasp_plan: PregraspSelection | None = None
    last_planned_trajectory: LastMotionPlan | None = None
    candidate_width_command: dict[str, float] | None = None


class RestrictedRealHardwareDemoController:
    """Menu-callable state machine for incremental restricted real hardware tests."""

    def __init__(
        self,
        *,
        hardware_mode: HardwareMode,
        real_arm_safety: RealArmSafetyConfig,
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
        self.state = RestrictedRealDemoState()
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

    def detect_object(self, query: str) -> DemoActionResult:
        if not query.strip():
            return self._blocked("Object query must not be empty.")
        self._audit("restricted_real_hw_detect_requested", query=query.strip())
        result = self.perception_pipeline.detect_object(query.strip(), require_pose=False)
        selected = _select_highest_confidence(result.get("objects", []))
        if selected is None:
            return self._blocked("No object detected.")
        self.state.object_query = query.strip()
        self.state.selected_object = selected
        self.state.point_cloud = None
        self.state.candidates = []
        self.state.selected_grasp_plan = None
        self.state.last_planned_trajectory = None
        self.state.candidate_width_command = None
        self._audit("real_camera_used", real_camera_used=True, query=query.strip())
        return DemoActionResult(True, "SUCCESS", "Object detected.", {"object": selected})

    def build_object_point_cloud(self) -> DemoActionResult:
        if self.state.selected_object is None:
            return self._blocked("Detect an object before building a point cloud.")
        point_cloud = self.pregrasp_controller._build_point_cloud(self.state.selected_object)
        if not point_cloud.get("success") or not point_cloud.get("point_cloud_path"):
            return self._blocked(
                f"Point-cloud generation failed: {point_cloud.get('message', 'unavailable')}"
            )
        self.state.point_cloud = point_cloud
        self.state.candidates = []
        self.state.selected_grasp_plan = None
        self.state.last_planned_trajectory = None
        self.state.candidate_width_command = None
        self._audit(
            "restricted_real_hw_point_cloud_built",
            point_cloud_path=point_cloud.get("point_cloud_path"),
        )
        return DemoActionResult(True, "SUCCESS", "Object point cloud built.", point_cloud)

    def generate_grasp_candidates(self) -> DemoActionResult:
        if self.state.point_cloud is None:
            return self._blocked("Build an object point cloud before generating grasps.")
        result = self.grasp_planner.generate_candidates(self.state.point_cloud["point_cloud_path"])
        candidates = result.get("candidates", [])
        if not candidates:
            return self._blocked(f"No GraspGenX candidates: {result.get('message', 'none returned')}")
        self.state.candidates = candidates
        self.state.selected_grasp_plan = None
        self.state.last_planned_trajectory = None
        self.state.candidate_width_command = None
        self._audit("restricted_real_hw_grasp_candidates_generated", candidate_count=len(candidates))
        return DemoActionResult(
            True,
            "SUCCESS",
            f"Generated {len(candidates)} GraspGenX candidate(s).",
            {"candidates": candidates},
        )

    def review_selected_grasp_candidate(self) -> DemoActionResult:
        if self.state.point_cloud is None:
            return self._blocked("Build an object point cloud before reviewing a grasp.")
        if not self.state.candidates:
            return self._blocked("Generate GraspGenX candidates before review.")
        width_screened = self._candidate_width_commands(self.state.candidates)
        if not width_screened:
            return self._blocked("No candidate has a safe usable gripper_width_m.")
        selection = self.pregrasp_controller.select_restricted_pregrasp_plan(
            [item["candidate"] for item in width_screened],
            self.state.point_cloud,
        )
        if selection is None:
            return self._blocked("No candidate produced a safe reachable pregrasp.")
        width_command = next(
            item["command"]
            for item in width_screened
            if item["candidate"] is selection.candidate
        )
        self.state.selected_grasp_plan = selection
        self.state.last_planned_trajectory = selection.plan
        self.state.candidate_width_command = width_command
        print_selected_candidate(selection.candidate)
        print_motion_summary(selection.plan.summary)
        self._audit(
            "restricted_real_hw_grasp_candidate_reviewed",
            selected_candidate_id=_candidate_value(selection.candidate, "candidate_id", "unknown"),
            selected_variant=selection.variant,
            plan_id=selection.plan.plan_id,
        )
        return DemoActionResult(
            True,
            "SUCCESS",
            "Selected GraspGenX candidate reviewed and pregrasp planned.",
            {"selection": selection, "width_command": width_command},
        )

    def plan_named_target(self, target_name: str) -> DemoActionResult:
        velocity_scale, acceleration_scale = self._restricted_scales()
        plan = plan_named_target(
            self.moveit_backend,
            target_name,
            self.real_arm_safety,
            self._logger(),
            velocity_scale,
            acceleration_scale,
        )
        if plan is None:
            return DemoActionResult(False, "BLOCKED", f"Could not plan to {target_name}.")
        self.state.last_planned_trajectory = plan
        return DemoActionResult(True, "SUCCESS", f"Planned {target_name}.", {"plan": plan})

    def execute_named_target(self, target_name: str, confirmation: str | None = None) -> DemoActionResult:
        if confirmation is None:
            confirmation = input(f"Type EXECUTE to move REAL UR5 to {target_name}:\n> ")
        result = execute_last_plan_for_target(
            self.moveit_backend,
            self.state.last_planned_trajectory,
            target_name,
            self.real_arm_safety,
            self._logger(),
        ) if confirmation is None else self._execute_plan_with_confirmation(
            self.state.last_planned_trajectory,
            target_name,
            confirmation,
        )
        return self._result_from_backend(result)

    def open_real_gripper(self, confirmation: str | None = None) -> DemoActionResult:
        if confirmation is None:
            confirmation = input("Open REAL gripper? Type YES:\n> ").strip()
        if not confirmation_is_yes(confirmation):
            self._audit("real_gripper_step_requires_confirmation", real_gripper_step_requires_confirmation=True)
            return DemoActionResult(False, "CANCELLED", "Exact YES confirmation is required.")
        if not self.real_arm_safety.gripper.get("allow_open_gripper", False):
            return self._blocked("Opening the real gripper is disabled by real_arm_safety.yaml.")
        speed, force = self._gripper_speed_force()
        self._audit("real_gripper_command_requested", real_gripper_command_requested=True)
        result = self.gripper_backend.open_gripper(
            speed_mps=speed,
            force_N=force,
            confirmed=True,
        )
        return self._gripper_result(result, "Real gripper open command completed.")

    def move_to_pregrasp(self, confirmation: str | None = None) -> DemoActionResult:
        if self.state.selected_grasp_plan is None or self.state.last_planned_trajectory is None:
            return self._blocked("Review a valid GraspPlan before moving to pregrasp.")
        plan = self.state.selected_grasp_plan.plan
        precheck = validate_execution_precheck(plan, self.real_arm_safety)
        if not precheck["allowed"]:
            return self._blocked(precheck["reason"])
        print_motion_summary(plan.summary)
        if confirmation is None:
            confirmation = input("Type EXECUTE to move REAL UR5 to pregrasp:\n> ")
        confirmation_check = validate_execute_confirmation(
            confirmation,
            self.real_arm_safety.confirmation_phrase,
        )
        if not confirmation_check["allowed"]:
            self._audit(
                "real_arm_step_requires_confirmation",
                real_arm_step_requires_confirmation=True,
                plan_id=plan.plan_id,
                skill_id=plan.skill_id,
            )
            return DemoActionResult(False, "CANCELLED", confirmation_check["reason"])
        result = self.moveit_backend.execute_cached_plan_real(plan.plan_id, final_confirmed=True)
        self._audit(
            "real_arm_motion_executed" if result.get("success") else "real_arm_motion_failed",
            plan_id=plan.plan_id,
            skill_id=plan.skill_id,
            target_name=plan.target_name,
            execution_success=bool(result.get("success")),
            message=result.get("message"),
        )
        return self._result_from_backend(result)

    def set_real_gripper_to_candidate_width(
        self,
        confirmation: str | None = None,
    ) -> DemoActionResult:
        if self.state.selected_grasp_plan is None or self.state.candidate_width_command is None:
            return self._blocked("Review a valid GraspPlan before setting candidate width.")
        width_m = self.state.candidate_width_command["width_m"]
        if confirmation is None:
            confirmation = input(
                f"Set REAL gripper to candidate width {width_m:.4f} m? Type YES:\n> "
            ).strip()
        if not confirmation_is_yes(confirmation):
            self._audit("real_gripper_step_requires_confirmation", real_gripper_step_requires_confirmation=True)
            return DemoActionResult(False, "CANCELLED", "Exact YES confirmation is required.")
        result = self.gripper_backend.set_gripper_width(
            width_m=width_m,
            speed_mps=self.state.candidate_width_command["speed_mps"],
            force_N=self.state.candidate_width_command["force_N"],
            confirmed=True,
        )
        return self._gripper_result(result, "Candidate-width gripper command completed.")

    def check_gripper_status(self) -> DemoActionResult:
        getter = getattr(self.gripper_backend, "get_gripper_state", None)
        if not callable(getter):
            return DemoActionResult(False, "UNKNOWN", "Real gripper state interface is unavailable.")
        result = getter()
        return self._result_from_backend(result)

    def block_forbidden_internal_action(self, action_name: str) -> DemoActionResult:
        if action_name in FORBIDDEN_MENU_TERMS:
            return self._blocked_forbidden_action(action_name)
        return DemoActionResult(False, "UNKNOWN", f"Unknown internal action {action_name}.")

    def state_summary(self) -> dict[str, Any]:
        selection = self.state.selected_grasp_plan
        return {
            "object_query": self.state.object_query,
            "selected_object": self.state.selected_object,
            "point_cloud_path": (
                self.state.point_cloud or {}
            ).get("point_cloud_path"),
            "candidate_count": len(self.state.candidates),
            "selected_candidate_id": (
                _candidate_value(selection.candidate, "candidate_id", "unknown")
                if selection is not None
                else None
            ),
            "last_plan_id": (
                self.state.last_planned_trajectory.plan_id
                if self.state.last_planned_trajectory is not None
                else None
            ),
            "candidate_width_m": (
                self.state.candidate_width_command or {}
            ).get("width_m"),
        }

    def _candidate_width_commands(self, candidates: list[Any]) -> list[dict[str, Any]]:
        usable = []
        speed, force = self._gripper_speed_force()
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
            try:
                command = validate_candidate_width_for_pregrasp_test(
                    candidate,
                    self.hardware_mode,
                    speed_mps=speed,
                    force_N=force,
                )
            except (ValueError, PermissionError) as exc:
                self._audit(
                    "restricted_real_hw_candidate_width_rejected",
                    candidate_id=_candidate_value(candidate, "candidate_id", "unknown"),
                    reason=str(exc),
                )
                continue
            usable.append({"candidate": candidate, "command": command})
        return usable

    def _execute_plan_with_confirmation(
        self,
        plan: LastMotionPlan | None,
        target_name: str,
        confirmation: str,
    ) -> dict[str, Any]:
        if plan is None:
            return {"success": False, "status": "BLOCKED", "message": "No valid plan exists. Plan first."}
        if plan.target_name != target_name:
            return {
                "success": False,
                "status": "BLOCKED",
                "message": f"Last plan is for {plan.target_name}, not {target_name}.",
            }
        precheck = validate_execution_precheck(plan, self.real_arm_safety)
        if not precheck["allowed"]:
            return {"success": False, "status": "BLOCKED", "message": precheck["reason"]}
        confirmation_check = validate_execute_confirmation(
            confirmation,
            self.real_arm_safety.confirmation_phrase,
        )
        if not confirmation_check["allowed"]:
            self._audit(
                "real_arm_step_requires_confirmation",
                real_arm_step_requires_confirmation=True,
                plan_id=plan.plan_id,
                skill_id=plan.skill_id,
            )
            return {"success": False, "status": "CANCELLED", "message": confirmation_check["reason"]}
        result = self.moveit_backend.execute_cached_plan_real(plan.plan_id, final_confirmed=True)
        self._audit(
            "real_arm_motion_executed" if result.get("success") else "real_arm_motion_failed",
            plan_id=plan.plan_id,
            skill_id=plan.skill_id,
            target_name=plan.target_name,
            execution_success=bool(result.get("success")),
            message=result.get("message"),
        )
        return result

    def _gripper_result(self, result: dict[str, Any], default_message: str) -> DemoActionResult:
        sent = bool(result.get("success"))
        self._audit(
            "real_gripper_command_sent" if sent else "real_gripper_command_failed",
            real_gripper_command_requested=True,
            real_gripper_command_sent=sent,
            message=result.get("message"),
        )
        return self._result_from_backend(result, default_message=default_message)

    def _result_from_backend(
        self,
        result: dict[str, Any],
        default_message: str = "Action completed.",
    ) -> DemoActionResult:
        success = bool(result.get("success"))
        return DemoActionResult(
            success,
            "SUCCESS" if success else str(result.get("status", "FAILED")),
            str(result.get("message", default_message)),
            dict(result.get("output", {})) if isinstance(result.get("output", {}), dict) else {},
        )

    def _blocked_forbidden_action(self, action_name: str) -> DemoActionResult:
        self._audit(
            "unsafe_llm_step_blocked",
            unsafe_llm_step_blocked=True,
            skill_id=action_name,
            reason=f"{action_name} is forbidden in restricted real hardware demo.",
            real_arm_execution_blocked=True,
        )
        return DemoActionResult(False, "BLOCKED", f"{action_name} is forbidden.")

    def _blocked(self, message: str) -> DemoActionResult:
        self._audit(
            "restricted_real_hw_demo_blocked",
            reason=message,
            real_arm_execution_blocked=True,
        )
        return DemoActionResult(False, "BLOCKED", message)

    def _restricted_scales(self) -> tuple[float, float]:
        preview_config = self.grasping_config.get("plan_only_preview", {}) or {}
        velocity_scale = min(
            float(preview_config.get("velocity_scale", 0.05)),
            self.real_arm_safety.limits.max_velocity_scale,
        )
        acceleration_scale = min(
            float(preview_config.get("acceleration_scale", 0.05)),
            self.real_arm_safety.limits.max_acceleration_scale,
        )
        return velocity_scale, acceleration_scale

    def _gripper_speed_force(self) -> tuple[float, float]:
        return (
            float(self.gripper_config.get("default_speed_mps", 0.02)),
            float(self.gripper_config.get("default_force_N", 10.0)),
        )

    def _logger(self) -> AuditLogger:
        return self.audit_logger or AuditLogger(hardware_mode=self.hardware_mode)

    def _audit(self, event: str, **fields: Any) -> None:
        if self.audit_logger is not None:
            self.audit_logger.log_hardware_safety_event(event, **fields)


def print_full_safety_summary(hardware_mode, real_arm_safety, gripper_config: dict) -> None:
    print("RESTRICTED REAL HARDWARE DEMO")
    print("This may move the REAL UR5 only to safe named targets or pregrasp.")
    print("No approach, contact close, lift, or full pick is available.")
    print_hardware_mode(hardware_mode)
    print_real_arm_safety(real_arm_safety)
    print("\nReal gripper limits:")
    print(
        f"width=[{hardware_mode.gripper.min_width_m:.3f}, "
        f"{hardware_mode.gripper.max_width_m:.3f}] m, "
        f"speed<={hardware_mode.gripper.max_speed_mps:.3f} m/s, "
        f"force<={hardware_mode.gripper.max_force_N:.1f} N"
    )
    driver = gripper_config.get("driver", {}) if isinstance(gripper_config, dict) else {}
    print(f"Gripper driver interface: {driver.get('interface_type', 'not configured')}")
    print("Arm confirmation phrase: EXECUTE")
    print("Gripper confirmation phrase: YES")


def menu_text() -> str:
    return (
        "\nMenu:\n"
        "1. Check real arm status\n"
        "2. Check real gripper status\n"
        "3. Move to safe_home\n"
        "4. Move to observe\n"
        "5. Detect object\n"
        "6. Build object point cloud\n"
        "7. Generate GraspGenX candidates\n"
        "8. Review selected grasp candidate\n"
        "9. Open real gripper\n"
        "10. Move real UR5 to pregrasp\n"
        "11. Set real gripper to candidate width\n"
        "12. Return to safe_home\n"
        "s. Show current demo state\n"
        "q. Quit"
    )


def print_menu() -> None:
    print(menu_text())


def run_menu(controller, node, rclpy_module, hardware_mode, real_arm_safety) -> None:
    while True:
        print_menu()
        choice = input("Select: ").strip().lower()
        if choice == "q":
            break
        if choice == "s":
            pprint.pprint(controller.state_summary(), sort_dicts=False)
        elif choice == "1":
            print_status_report(run_status_checks(node, hardware_mode, real_arm_safety))
        elif choice == "2":
            print(controller.check_gripper_status().message)
        elif choice == "3":
            plan = controller.plan_named_target("safe_home")
            print(f"{plan.status}: {plan.message}")
            if plan.success:
                print(controller.execute_named_target("safe_home", confirmation=None).message)
        elif choice == "4":
            plan = controller.plan_named_target("observe")
            print(f"{plan.status}: {plan.message}")
            if plan.success:
                print(controller.execute_named_target("observe", confirmation=None).message)
        elif choice == "5":
            query = input("Enter object query:\n> ")
            print(controller.detect_object(query).message)
        elif choice == "6":
            print(controller.build_object_point_cloud().message)
        elif choice == "7":
            print(controller.generate_grasp_candidates().message)
        elif choice == "8":
            print(controller.review_selected_grasp_candidate().message)
        elif choice == "9":
            print(controller.open_real_gripper(confirmation=None).message)
        elif choice == "10":
            print(controller.move_to_pregrasp(confirmation=None).message)
        elif choice == "11":
            print(controller.set_real_gripper_to_candidate_width(confirmation=None).message)
        elif choice == "12":
            plan = controller.plan_named_target("safe_home")
            print(f"{plan.status}: {plan.message}")
            if plan.success:
                print(controller.execute_named_target("safe_home", confirmation=None).message)
        else:
            print("Unknown selection.")
        rclpy_module.spin_once(node, timeout_sec=0.01)


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
        print(f"BLOCKED: failed to start restricted real hardware demo: {exc}")
        return

    print_full_safety_summary(
        hardware_mode,
        real_arm_safety,
        gripper_document.get("gripper", {}),
    )

    node = None
    moveit_backend = None
    try:
        rclpy.init()
        node = rclpy.create_node("ros_restricted_real_hw_demo")
        perception_pipeline = create_perception_pipeline(node=node, config=perception_config)
        grasping_config = grasping_document.get("grasping", {})
        gripper_config = gripper_document.get("gripper", {})
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
        logger = AuditLogger("restricted_real_hw_demo_audit.log", hardware_mode=hardware_mode)
        controller = RestrictedRealHardwareDemoController(
            hardware_mode=hardware_mode,
            real_arm_safety=real_arm_safety,
            perception_pipeline=perception_pipeline,
            point_cloud_builder=point_cloud_builder,
            grasp_planner=grasp_planner,
            moveit_backend=moveit_backend,
            gripper_backend=gripper_backend,
            grasping_config=grasping_config,
            gripper_config=gripper_config,
            transformer=TFPoseTransformer(node, target_frame="base_link"),
            audit_logger=logger,
        )

        print("\nWaiting for D435i RGB/depth/camera_info messages...")
        wait_for_camera_messages(perception_pipeline.backend, node, rclpy)
        run_menu(controller, node, rclpy, hardware_mode, real_arm_safety)
    except KeyboardInterrupt:
        print("\nCancelled by user.")
    except Exception as exc:
        print(f"ERROR: restricted real hardware demo failed: {exc}")
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
