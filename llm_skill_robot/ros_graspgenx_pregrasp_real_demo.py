"""Restricted real-UR5 pregrasp execution demo using D435i, SAM3, and GraspGenX."""

from __future__ import annotations

from dataclasses import dataclass
import pprint
import sys
import time
from typing import Any

from llm_skill_robot.core.audit_logger import AuditLogger
from llm_skill_robot.grasping.graspgenx_grasp_planner import GraspGenXGraspPlanner
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
)
from llm_skill_robot.ros_graspgenx_plan_only_demo import (
    moveit_stage_planning_kwargs,
    planning_variant_kwargs,
    prepare_candidate_preview,
)
from llm_skill_robot.ros_real_arm_safe_motion_test import (
    LastMotionPlan,
    build_restricted_motion_summary,
    print_motion_summary,
    validate_execute_confirmation,
    validate_execution_precheck,
)
from llm_skill_robot.ros_real_gripper_test import confirmation_is_yes
from llm_skill_robot.safety.hardware_mode import (
    assert_arm_execution_allowed,
    load_hardware_mode,
    print_hardware_mode,
)
from llm_skill_robot.safety.real_arm_safety import (
    is_real_motion_skill_allowed,
    load_real_arm_safety,
    print_real_arm_safety,
)


@dataclass
class PregraspDemoResult:
    success: bool
    status: str
    message: str
    plan: LastMotionPlan | None = None


@dataclass
class PregraspSelection:
    candidate: Any
    variant: str
    pregrasp_pose: dict[str, Any]
    plan: LastMotionPlan
    attempts: list[dict[str, Any]]


class PregraspRealDemoController:
    """One-shot state machine that can execute only the selected pregrasp pose."""

    def __init__(
        self,
        *,
        hardware_mode,
        real_arm_safety,
        perception_pipeline,
        point_cloud_builder,
        grasp_planner,
        moveit_backend,
        gripper_backend=None,
        transformer=None,
        grasping_config: dict | None = None,
        gripper_config: dict | None = None,
        audit_logger: AuditLogger | None = None,
    ):
        self.hardware_mode = hardware_mode
        self.real_arm_safety = real_arm_safety
        self.perception_pipeline = perception_pipeline
        self.point_cloud_builder = point_cloud_builder
        self.grasp_planner = grasp_planner
        self.moveit_backend = moveit_backend
        self.gripper_backend = gripper_backend
        self.transformer = transformer
        self.grasping_config = grasping_config or {}
        self.preview_config = self.grasping_config.get("plan_only_preview", {})
        self.gripper_config = gripper_config or {}
        self.audit_logger = audit_logger

    def run_once(
        self,
        query: str,
        *,
        open_gripper_confirmation: str | None = "",
        execute_confirmation: str | None = "",
    ) -> PregraspDemoResult:
        if not query.strip():
            return self._blocked("Object query must not be empty.")
        if not self._pregrasp_skill_allowed():
            return self._blocked("move_to_pregrasp is not allowed by real_arm_safety.yaml.")

        self._audit("pregrasp_real_demo_started", query=query)

        detection = self.perception_pipeline.detect_object(query.strip(), require_pose=False)
        selected_object = _select_highest_confidence(detection.get("objects", []))
        if selected_object is None:
            return self._blocked("No object detected.")

        point_cloud = self._build_point_cloud(selected_object)
        if not point_cloud.get("success") or not point_cloud.get("point_cloud_path"):
            return self._blocked(
                f"Point-cloud generation failed: {point_cloud.get('message', 'unavailable')}"
            )

        generation = self.grasp_planner.generate_candidates(point_cloud["point_cloud_path"])
        candidates = generation.get("candidates", [])
        if not candidates:
            return self._blocked(
                f"No GraspGenX candidates: {generation.get('message', 'none returned')}"
            )
        selection = self.select_restricted_pregrasp_plan(candidates, point_cloud)
        if selection is None:
            return self._blocked("No GraspGenX candidate produced a safe reachable pregrasp.")
        plan = selection.plan
        print(
            "\nSelected pregrasp candidate: "
            f"id={_candidate_value(selection.candidate, 'candidate_id', 'unknown')} "
            f"score={_candidate_value(selection.candidate, 'score')} "
            f"variant={selection.variant}"
        )

        print_motion_summary(plan.summary)
        if execute_confirmation is None:
            execute_confirmation = input("Type EXECUTE to move the REAL UR5 to pregrasp:\n> ")
        confirmation_check = validate_execute_confirmation(
            execute_confirmation,
            self.real_arm_safety.confirmation_phrase,
        )
        if not confirmation_check["allowed"]:
            self._audit(
                "real_arm_motion_rejected",
                plan_id=plan.plan_id,
                skill_id=plan.skill_id,
                target_name=plan.target_name,
                reason=confirmation_check["reason"],
                real_arm_execution_blocked=True,
            )
            return PregraspDemoResult(
                False,
                "CANCELLED",
                "EXECUTE confirmation was not provided. No real arm motion was executed.",
                plan,
            )

        if open_gripper_confirmation is None:
            open_gripper_confirmation = input("Open real gripper before motion? Type YES:\n> ").strip()
        self._maybe_open_gripper(open_gripper_confirmation)
        self._audit(
            "real_arm_motion_confirmed",
            plan_id=plan.plan_id,
            skill_id=plan.skill_id,
            target_name=plan.target_name,
        )
        result = self.moveit_backend.execute_cached_plan_real(plan.plan_id, final_confirmed=True)
        event = "real_arm_motion_executed" if result.get("success") else "real_arm_motion_failed"
        self._audit(
            event,
            plan_id=plan.plan_id,
            skill_id=plan.skill_id,
            target_name=plan.target_name,
            execution_success=bool(result.get("success")),
            message=result.get("message"),
        )
        return PregraspDemoResult(
            bool(result.get("success")),
            "SUCCESS" if result.get("success") else "FAILED",
            str(result.get("message", "")),
            plan,
        )

    def select_restricted_pregrasp_plan(
        self,
        candidates: list[Any],
        point_cloud: dict,
        *,
        preserve_candidate_order: bool = False,
        report_rejections: bool = True,
    ) -> PregraspSelection | None:
        screening_config = self.preview_config.get("candidate_screening", {}) or {}
        max_candidates = max(1, int(screening_config.get("max_candidates", 12)))
        variants = list(
            screening_config.get(
                "variants", ["relaxed_orientation", "full_pose", "position_only"]
            )
        )
        if not screening_config.get("allow_position_only_fallback", True):
            variants = [variant for variant in variants if variant != "position_only"]
        valid_candidates = [
            candidate
            for candidate in candidates
            if _candidate_value(candidate, "valid", True) is not False
        ]
        if not preserve_candidate_order:
            valid_candidates = sorted(
                valid_candidates,
                key=lambda candidate: float(_candidate_value(candidate, "score", 0.0) or 0.0),
                reverse=True,
            )
        valid_candidates = valid_candidates[:max_candidates]

        attempts: list[dict[str, Any]] = []
        for candidate_index, candidate in enumerate(valid_candidates, start=1):
            candidate_id = _candidate_value(candidate, "candidate_id", f"candidate_{candidate_index}")
            try:
                poses = prepare_candidate_preview(
                    candidate,
                    point_cloud,
                    self.preview_config,
                    target_frame="base_link",
                    transformer=self.transformer,
                )
            except ValueError as exc:
                attempts.append(
                    {
                        "candidate_id": candidate_id,
                        "variant": "prepare",
                        "accepted": False,
                        "reason": str(exc),
                    }
                )
                continue

            pregrasp_pose = poses["pregrasp"]
            if pregrasp_pose.get("frame") != "base_link":
                attempts.append(
                    {
                        "candidate_id": candidate_id,
                        "variant": "prepare",
                        "accepted": False,
                        "reason": "pregrasp pose is not in base_link",
                    }
                )
                continue
            workspace_check = validate_pregrasp_workspace(pregrasp_pose, self.preview_config)
            if not workspace_check["allowed"]:
                attempts.append(
                    {
                        "candidate_id": candidate_id,
                        "variant": "workspace",
                        "accepted": False,
                        "reason": workspace_check["reason"],
                    }
                )
                continue

            for variant in variants:
                try:
                    plan = self._plan_pregrasp(
                        pregrasp_pose,
                        variant=variant,
                        variant_kwargs=planning_variant_kwargs(variant, screening_config),
                    )
                except ValueError as exc:
                    attempts.append(
                        {
                            "candidate_id": candidate_id,
                            "variant": variant,
                            "accepted": False,
                            "reason": str(exc),
                        }
                    )
                    continue
                if plan is None:
                    attempts.append(
                        {
                            "candidate_id": candidate_id,
                            "variant": variant,
                            "accepted": False,
                            "reason": "MoveIt pregrasp planning failed",
                        }
                    )
                    continue
                precheck = validate_execution_precheck(plan, self.real_arm_safety)
                if not precheck["allowed"]:
                    attempts.append(
                        {
                            "candidate_id": candidate_id,
                            "variant": variant,
                            "accepted": False,
                            "reason": precheck["reason"],
                            "plan_id": plan.plan_id,
                        }
                    )
                    continue
                attempts.append(
                    {
                        "candidate_id": candidate_id,
                        "variant": variant,
                        "accepted": True,
                        "plan_id": plan.plan_id,
                    }
                )
                self._audit(
                    "pregrasp_candidate_screening",
                    selected_candidate_id=candidate_id,
                    selected_variant=variant,
                    attempts=attempts,
                )
                return PregraspSelection(candidate, variant, pregrasp_pose, plan, attempts)

        self._audit("pregrasp_candidate_screening_failed", attempts=attempts)
        if report_rejections:
            print("\nPregrasp candidate screening rejected all candidates:")
            for attempt in attempts[:20]:
                print(
                    "- "
                    f"candidate={attempt.get('candidate_id')} "
                    f"variant={attempt.get('variant')} "
                    f"reason={attempt.get('reason')}"
                )
        return None

    def _build_point_cloud(self, selected_object: Any) -> dict:
        camera_backend = getattr(self.perception_pipeline, "backend", None)
        depth_getter = getattr(camera_backend, "_get_latest_depth_image", None)
        return self.point_cloud_builder.build(
            selected_object,
            depth_getter() if callable(depth_getter) else None,
            getattr(camera_backend, "latest_camera_info_msg", None),
            frame=getattr(camera_backend, "config", {}).get("depth_frame", "base_link"),
        )

    def _plan_pregrasp(
        self,
        pregrasp_pose: dict,
        variant: str = "default",
        variant_kwargs: dict | None = None,
    ) -> LastMotionPlan | None:
        velocity_scale = min(
            float(self.preview_config.get("velocity_scale", 0.05)),
            self.real_arm_safety.limits.max_velocity_scale,
        )
        acceleration_scale = min(
            float(self.preview_config.get("acceleration_scale", 0.05)),
            self.real_arm_safety.limits.max_acceleration_scale,
        )
        self._audit(
            "real_arm_motion_requested",
            skill_id="move_to_pregrasp",
            target_name="selected_pregrasp",
        )
        result = self.moveit_backend.plan_to_pose(
            frame=pregrasp_pose["frame"],
            position=pregrasp_pose["position"],
            orientation=pregrasp_pose["orientation"],
            velocity_scale=velocity_scale,
            acceleration_scale=acceleration_scale,
            skill_id="move_to_pregrasp",
            **{
                **moveit_stage_planning_kwargs(self.preview_config, "pregrasp"),
                **(variant_kwargs or {}),
            },
        )
        summary = build_restricted_motion_summary(
            result.get("summary", {}),
            target_name="selected_pregrasp",
            velocity_scale=velocity_scale,
            acceleration_scale=acceleration_scale,
        )
        self._audit(
            "real_arm_motion_planned",
            skill_id="move_to_pregrasp",
            target_name="selected_pregrasp",
            variant=variant,
            plan_id=result.get("plan_id"),
            planning_success=bool(result.get("success")),
            summary=summary,
        )
        if not result.get("success"):
            return None
        return LastMotionPlan(
            plan_id=str(result["plan_id"]),
            target_name="selected_pregrasp",
            skill_id="move_to_pregrasp",
            summary=summary,
        )

    def _maybe_open_gripper(self, confirmation: str) -> None:
        if self.gripper_backend is None:
            return
        if not confirmation_is_yes(confirmation):
            print("Real gripper open was not confirmed. Skipping gripper command.")
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

    def _pregrasp_skill_allowed(self) -> bool:
        result = is_real_motion_skill_allowed("move_to_pregrasp", self.real_arm_safety)
        return bool(result.get("allowed"))

    def _blocked(self, message: str, plan: LastMotionPlan | None = None) -> PregraspDemoResult:
        self._audit(
            "real_arm_motion_rejected",
            reason=message,
            real_arm_execution_blocked=True,
        )
        return PregraspDemoResult(False, "BLOCKED", message, plan)

    def _audit(self, event: str, **fields: Any) -> None:
        if self.audit_logger is not None:
            self.audit_logger.log_hardware_safety_event(event, **fields)


def validate_pregrasp_workspace(
    pose: dict[str, Any],
    preview_config: dict | None = None,
) -> dict[str, Any]:
    """Check a conservative base_link workspace box for pregrasp testing."""
    workspace = (preview_config or {}).get("real_pregrasp_workspace", {}) or {}
    position = pose.get("position", {})
    try:
        x = float(position["x"])
        y = float(position["y"])
        z = float(position["z"])
    except (KeyError, TypeError, ValueError) as exc:
        return {"allowed": False, "reason": f"Invalid pregrasp position: {exc}"}

    x_limits = workspace.get("x_m", [0.15, 0.85])
    y_limits = workspace.get("y_m", [-0.65, 0.65])
    z_limits = workspace.get("z_m", [0.05, 0.75])
    if not float(x_limits[0]) <= x <= float(x_limits[1]):
        return {"allowed": False, "reason": f"pregrasp x={x:.3f} is outside workspace."}
    if not float(y_limits[0]) <= y <= float(y_limits[1]):
        return {"allowed": False, "reason": f"pregrasp y={y:.3f} is outside workspace."}
    if not float(z_limits[0]) <= z <= float(z_limits[1]):
        return {"allowed": False, "reason": f"pregrasp z={z:.3f} is outside workspace."}
    return {"allowed": True, "reason": "Pregrasp pose is inside the configured workspace."}


def wait_for_camera_messages(backend, node, rclpy_module, timeout_sec: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if (
            getattr(backend, "latest_rgb_msg", None) is not None
            and getattr(backend, "latest_depth_msg", None) is not None
            and getattr(backend, "latest_camera_info_msg", None) is not None
        ):
            return
        rclpy_module.spin_once(node, timeout_sec=0.1)


def print_stop_notice() -> None:
    print("No approach, contact, close, or lift was executed.")


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
        print(f"BLOCKED: failed to start restricted pregrasp demo: {exc}")
        return

    print("REAL UR5 GRASPGENX PREGRASP DEMO")
    print("This may move the REAL robot to pregrasp only.")
    print_hardware_mode(hardware_mode)
    print_real_arm_safety(real_arm_safety)

    node = None
    moveit_backend = None
    try:
        rclpy.init()
        node = rclpy.create_node("ros_graspgenx_pregrasp_real_demo")
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
        logger = AuditLogger("pregrasp_real_demo_audit.log", hardware_mode=hardware_mode)
        controller = PregraspRealDemoController(
            hardware_mode=hardware_mode,
            real_arm_safety=real_arm_safety,
            perception_pipeline=perception_pipeline,
            point_cloud_builder=point_cloud_builder,
            grasp_planner=grasp_planner,
            moveit_backend=moveit_backend,
            gripper_backend=gripper_backend,
            transformer=TFPoseTransformer(node, target_frame="base_link"),
            grasping_config=grasping_config,
            gripper_config=gripper_document.get("gripper", {}),
            audit_logger=logger,
        )

        print("\nWaiting for D435i RGB/depth/camera_info messages...")
        wait_for_camera_messages(perception_pipeline.backend, node, rclpy)

        query = input("Enter object query:\n> ").strip()
        result = controller.run_once(
            query,
            open_gripper_confirmation=None,
            execute_confirmation=None,
        )
        print(f"{result.status}: {result.message}")
        if result.plan is not None:
            print("pregrasp plan:")
            pprint.pprint(result.plan.summary, sort_dicts=False)
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
