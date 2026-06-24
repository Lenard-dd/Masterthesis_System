"""Unified real-perception, reviewed-gripper, and plan-only MoveIt demo."""

from __future__ import annotations

from dataclasses import dataclass, field
import pprint
import sys
from typing import Any

from llm_skill_robot.core.audit_logger import AuditLogger
from llm_skill_robot.core.trajectory_validator import TrajectoryValidator
from llm_skill_robot.core.trajectory_visualizer import TrajectoryVisualizer
from llm_skill_robot.grasping.graspgenx_grasp_planner import GraspGenXGraspPlanner
from llm_skill_robot.grasping.graspgenx_subprocess_adapter import GraspGenXSubprocessAdapter
from llm_skill_robot.perception.object_point_cloud_builder import ObjectPointCloudBuilder
from llm_skill_robot.perception.perception_factory import create_perception_pipeline
from llm_skill_robot.robot.robotiq_2f140_real_backend import Robotiq2F140RealBackend
from llm_skill_robot.robot.ur5_moveit_plan_backend import UR5MoveItPlanBackend
from llm_skill_robot.ros_graspgenx_gripper_width_test import (
    _candidate_value,
    _load_yaml,
    _select_highest_confidence,
    command_candidate_width_if_confirmed,
    extract_candidate_width,
)
from llm_skill_robot.ros_graspgenx_plan_only_demo import (
    _publish_sequence_preview,
    assert_plan_only_mode,
    create_preview_poses,
    plan_pose_sequence,
)
from llm_skill_robot.ros_real_gripper_test import confirmation_is_yes
from llm_skill_robot.safety.hardware_mode import HardwareMode, load_hardware_mode, print_hardware_mode


ARM_BLOCK_MESSAGE = "BLOCKED: real arm execution is disabled by hardware_mode.yaml."


@dataclass
class HardwareDemoState:
    detection_result: dict | None = None
    selected_object: Any | None = None
    point_cloud: dict | None = None
    candidates: list[Any] = field(default_factory=list)
    selected_candidate: Any | None = None
    candidate_reviewed: bool = False
    grasp_plan: dict[str, dict] | None = None
    last_gripper_result: dict | None = None
    last_plan_result: dict | None = None

    def reset_after_detection(self) -> None:
        self.point_cloud = None
        self.candidates = []
        self.selected_candidate = None
        self.candidate_reviewed = False
        self.grasp_plan = None
        self.last_gripper_result = None
        self.last_plan_result = None


class HardwareSafeDemoController:
    """Ordered HW6 state machine. It contains no real-arm execution call."""

    def __init__(
        self,
        *,
        hardware_mode: HardwareMode,
        perception_pipeline,
        point_cloud_builder,
        grasp_planner,
        gripper_backend,
        moveit_backend,
        trajectory_validator,
        trajectory_visualizer=None,
        gripper_config: dict | None = None,
        preview_config: dict | None = None,
        audit_logger: AuditLogger | None = None,
    ):
        assert_plan_only_mode(hardware_mode)
        self.hardware_mode = hardware_mode
        self.perception_pipeline = perception_pipeline
        self.point_cloud_builder = point_cloud_builder
        self.grasp_planner = grasp_planner
        self.gripper_backend = gripper_backend
        self.moveit_backend = moveit_backend
        self.trajectory_validator = trajectory_validator
        self.trajectory_visualizer = trajectory_visualizer
        self.gripper_config = gripper_config or {}
        self.preview_config = preview_config or {}
        self.audit_logger = audit_logger
        self.state = HardwareDemoState()

    def detect_object(self, query: str) -> dict:
        if not query.strip():
            return self._blocked("Object query must not be empty.")
        result = self.perception_pipeline.detect_object(query.strip(), require_pose=False)
        selected = _select_highest_confidence(result.get("objects", []))
        self.state.detection_result = result
        self.state.selected_object = selected
        self.state.reset_after_detection()
        if selected is None:
            return self._blocked("No object detected.", status=result.get("status", "NOT_FOUND"))
        self._audit("real_camera_used", real_camera_used=True)
        return {
            "success": True,
            "status": "SUCCESS",
            "message": "Highest-confidence object selected.",
            "object": selected,
        }

    def build_object_point_cloud(self) -> dict:
        if self.state.selected_object is None:
            return self._blocked("Detect an object before building a point cloud.")
        camera_backend = getattr(self.perception_pipeline, "backend", None)
        depth_getter = getattr(camera_backend, "_get_latest_depth_image", None)
        result = self.point_cloud_builder.build(
            self.state.selected_object,
            depth_getter() if callable(depth_getter) else None,
            getattr(camera_backend, "latest_camera_info_msg", None),
            frame=getattr(camera_backend, "config", {}).get("depth_frame"),
        )
        if not result.get("success", False) or not result.get("point_cloud_path"):
            self.state.point_cloud = None
            return self._blocked(
                f"Point-cloud generation stopped: {result.get('message', 'data unavailable')}",
                status=result.get("status", "NOT_AVAILABLE"),
            )
        self.state.point_cloud = result
        self.state.candidates = []
        self.state.selected_candidate = None
        self.state.candidate_reviewed = False
        self.state.grasp_plan = None
        return result

    def generate_grasp_candidates(self) -> dict:
        if not self.state.point_cloud or not self.state.point_cloud.get("point_cloud_path"):
            return self._blocked("Build an object point cloud before generating grasps.")
        result = self.grasp_planner.generate_candidates(
            self.state.point_cloud["point_cloud_path"]
        )
        candidates = result.get("candidates", [])
        if not candidates:
            self.state.candidates = []
            return self._blocked(
                f"No GraspGenX candidates: {result.get('message', 'none returned')}",
                status=result.get("status", "NOT_FOUND"),
            )
        self.state.candidates = list(candidates)
        self.state.selected_candidate = None
        self.state.candidate_reviewed = False
        self.state.grasp_plan = None
        return {
            "success": True,
            "status": "SUCCESS",
            "message": f"Generated {len(candidates)} GraspGenX candidate(s).",
            "candidates": candidates,
        }

    def review_best_candidate(self, confirmation: str) -> dict:
        if not self.state.candidates:
            return self._blocked("Generate GraspGenX candidates before review.")
        candidate = self.grasp_planner.select_best_candidate(self.state.candidates)
        if candidate is None:
            return self._blocked("No valid GraspGenX candidate is available for review.")
        print("\nBest GraspGenX candidate:")
        pprint.pprint(candidate, sort_dicts=False)
        if not confirmation_is_yes(confirmation):
            self.state.selected_candidate = None
            self.state.candidate_reviewed = False
            self.state.grasp_plan = None
            self._audit(
                "grasp_candidate_reviewed",
                grasp_candidate_reviewed=True,
                grasp_candidate_approved=False,
            )
            return self._blocked("Candidate review requires exact YES approval.", status="CANCELLED")
        try:
            grasp_plan = create_preview_poses(
                candidate,
                self.preview_config,
                fallback_frame=(self.state.point_cloud or {}).get("frame") or "base_link",
            )
        except ValueError as exc:
            return self._blocked(f"Candidate pose is not usable: {exc}")
        self.state.selected_candidate = candidate
        self.state.candidate_reviewed = True
        self.state.grasp_plan = grasp_plan
        self._audit(
            "grasp_candidate_reviewed",
            grasp_candidate_reviewed=True,
            grasp_candidate_approved=True,
        )
        return {
            "success": True,
            "status": "APPROVED",
            "message": "Best candidate reviewed and approved.",
            "candidate": candidate,
            "grasp_plan": grasp_plan,
        }

    def test_real_gripper_width(self, confirmation: str) -> dict:
        if not self.state.candidate_reviewed or self.state.selected_candidate is None:
            return self._blocked("Review and approve a grasp candidate before testing width.")
        speed = float(self.gripper_config.get("default_speed_mps", 0.02))
        force = float(self.gripper_config.get("default_force_N", 10.0))
        self._audit(
            "real_gripper_command_requested",
            real_gripper_command_requested=True,
        )
        result = command_candidate_width_if_confirmed(
            self.state.selected_candidate,
            self.gripper_backend,
            self.hardware_mode,
            confirmation,
            speed,
            force,
        )
        self.state.last_gripper_result = result
        output = result.get("output", {}) if isinstance(result, dict) else {}
        self._audit(
            "real_gripper_command_result",
            real_gripper_command_requested=True,
            real_gripper_command_sent=bool(output.get("command_sent", False)),
        )
        return result

    def preview_moveit_plan_only(self) -> dict:
        if not self.state.candidate_reviewed or self.state.grasp_plan is None:
            return self._blocked("Review a candidate and create its GraspPlan before preview.")
        if getattr(self.moveit_backend, "execution_mode", None) != "plan_only":
            return self.request_arm_execution()
        sequence = plan_pose_sequence(
            self.moveit_backend,
            self.state.grasp_plan,
            self.trajectory_validator,
            velocity_scale=float(self.preview_config.get("velocity_scale", 0.05)),
            acceleration_scale=float(self.preview_config.get("acceleration_scale", 0.05)),
        )
        preview = None
        if sequence.get("success", False) and self.trajectory_visualizer is not None:
            preview = _publish_sequence_preview(
                self.moveit_backend, sequence, self.trajectory_visualizer
            )
        result = {
            "success": bool(sequence.get("success", False)),
            "status": "PLAN_ONLY",
            "message": "MoveIt plan-only sequence completed; no arm motion was executed.",
            "sequence": sequence,
            "preview": preview,
        }
        self.state.last_plan_result = result
        return result

    def request_arm_execution(self) -> dict:
        self._audit("real_arm_execution_blocked", real_arm_execution_blocked=True)
        print(ARM_BLOCK_MESSAGE)
        return self._blocked(ARM_BLOCK_MESSAGE, status="BLOCKED")

    def show_state(self) -> dict:
        gripper_state = None
        getter = getattr(self.gripper_backend, "get_gripper_state", None)
        if callable(getter):
            gripper_state = getter()
        return {
            "success": True,
            "status": "STATE",
            "hardware_mode": self.hardware_mode.mode,
            "object_detected": self.state.selected_object is not None,
            "point_cloud_available": bool(self.state.point_cloud),
            "num_candidates": len(self.state.candidates),
            "candidate_reviewed": self.state.candidate_reviewed,
            "grasp_plan_available": self.state.grasp_plan is not None,
            "gripper_state": gripper_state,
            "last_gripper_result": self.state.last_gripper_result,
            "last_plan_result": self.state.last_plan_result,
        }

    @staticmethod
    def _blocked(message: str, status: str = "BLOCKED") -> dict:
        return {"success": False, "status": status, "message": message}

    def _audit(self, event: str, **fields: Any) -> None:
        if self.audit_logger is not None:
            self.audit_logger.log_hardware_safety_event(event, **fields)


def _print_result(result: dict) -> None:
    print("\nResult:")
    pprint.pprint(result, sort_dicts=False)


def main() -> None:
    try:
        import rclpy
    except ImportError:
        print("Error: rclpy not available. Please source ROS 2 setup.", file=sys.stderr)
        return

    hardware_mode = load_hardware_mode()
    try:
        assert_plan_only_mode(hardware_mode)
    except PermissionError as exc:
        print(f"Safety error: {exc}")
        return
    print("HARDWARE-SAFE PERCEPTION + GRIPPER DEMO")
    print_hardware_mode(hardware_mode)
    print(ARM_BLOCK_MESSAGE)

    perception_config = _load_yaml("perception.yaml")
    grasping_document = _load_yaml("grasping.yaml")
    gripper_document = _load_yaml("gripper.yaml")
    safety_config = _load_yaml("safety.yaml")
    grasping_config = grasping_document.get("grasping", {})
    gripper_config = gripper_document.get("gripper", {})

    node = None
    moveit_backend = None
    try:
        rclpy.init()
        node = rclpy.create_node("ros_hw_perception_gripper_demo")
        moveit_backend = UR5MoveItPlanBackend(
            node=node,
            dry_run=False,
            execution_mode="plan_only",
            hardware_mode=hardware_mode,
        )
        controller = HardwareSafeDemoController(
            hardware_mode=hardware_mode,
            perception_pipeline=create_perception_pipeline(
                node=node, config=perception_config
            ),
            point_cloud_builder=ObjectPointCloudBuilder(
                grasping_config.get("point_cloud", {})
            ),
            grasp_planner=GraspGenXGraspPlanner(
                GraspGenXSubprocessAdapter(
                    grasping_config.get("graspgenx_subprocess", {})
                )
            ),
            gripper_backend=Robotiq2F140RealBackend(
                node=node,
                config=gripper_document,
                hardware_mode=hardware_mode,
            ),
            moveit_backend=moveit_backend,
            trajectory_validator=TrajectoryValidator(safety_config),
            trajectory_visualizer=TrajectoryVisualizer(node=node),
            gripper_config=gripper_config,
            preview_config=grasping_config.get("plan_only_preview", {}),
            audit_logger=AuditLogger(hardware_mode=hardware_mode),
        )

        while rclpy.ok():
            choice = input(
                "\n1. Detect object\n"
                "2. Build object point cloud\n"
                "3. Generate GraspGenX candidates\n"
                "4. Review best candidate\n"
                "5. Test real gripper width from selected candidate\n"
                "6. Preview MoveIt plan only\n"
                "7. Show state\n"
                "q. Quit\n> "
            ).strip().lower()
            if choice == "q":
                break
            if choice == "1":
                result = controller.detect_object(input("Enter object query:\n> "))
            elif choice == "2":
                result = controller.build_object_point_cloud()
            elif choice == "3":
                result = controller.generate_grasp_candidates()
            elif choice == "4":
                if controller.state.candidates:
                    best = controller.grasp_planner.select_best_candidate(
                        controller.state.candidates
                    )
                    print("Best candidate preview:")
                    pprint.pprint(best, sort_dicts=False)
                response = input("Approve this candidate? Type YES to continue: ")
                result = controller.review_best_candidate(response)
            elif choice == "5":
                try:
                    width = extract_candidate_width(controller.state.selected_candidate)
                except (TypeError, ValueError):
                    width = None
                prompt = (
                    f"Set REAL gripper to candidate width {width} m? Type YES to continue: "
                )
                result = controller.test_real_gripper_width(input(prompt))
            elif choice == "6":
                result = controller.preview_moveit_plan_only()
            elif choice == "7":
                result = controller.show_state()
            else:
                result = {"success": False, "status": "INVALID", "message": "Unknown menu choice."}
            _print_result(result)
    except KeyboardInterrupt:
        print("\nInterrupted. No further hardware commands will be sent.")
    except Exception as exc:
        print(f"\nStopped safely after an error: {exc}", file=sys.stderr)
        print("No real arm command was sent.", file=sys.stderr)
    finally:
        if moveit_backend is not None:
            moveit_backend.shutdown()
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
