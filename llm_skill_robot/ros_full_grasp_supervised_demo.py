"""Fully supervised, step-by-step real UR5 and Robotiq grasp demonstration."""

from __future__ import annotations

from pathlib import Path
import pprint
import sys
import time
from typing import Any, Callable

import yaml

from llm_skill_robot.core.audit_logger import AuditLogger
from llm_skill_robot.core.grasp_motion_policy import GraspMotionPolicy
from llm_skill_robot.grasping.full_grasp_state_machine import FullGraspStateMachine
from llm_skill_robot.grasping.grasp_execution_backend import GraspExecutionBackend
from llm_skill_robot.grasping.grasp_verifier import GraspVerifier
from llm_skill_robot.grasping.graspgenx_grasp_planner import GraspGenXGraspPlanner
from llm_skill_robot.grasping.graspgenx_subprocess_adapter import GraspGenXSubprocessAdapter
from llm_skill_robot.perception.object_point_cloud_builder import ObjectPointCloudBuilder
from llm_skill_robot.perception.perception_factory import create_perception_pipeline
from llm_skill_robot.perception.tf_pose_transformer import TFPoseTransformer
from llm_skill_robot.robot.robotiq_2f140_real_backend import Robotiq2F140RealBackend
from llm_skill_robot.robot.ur5_moveit_plan_backend import UR5MoveItPlanBackend
from llm_skill_robot.ros_graspgenx_plan_only_demo import prepare_candidate_preview
from llm_skill_robot.ros_graspgenx_plan_only_demo import (
    moveit_stage_planning_kwargs,
    planning_variant_kwargs,
)
from llm_skill_robot.ros_graspgenx_pregrasp_real_demo import PregraspRealDemoController
from llm_skill_robot.ros_restricted_real_hw_demo import RestrictedRealHardwareDemoController
from llm_skill_robot.safety.hardware_mode import load_hardware_mode, print_hardware_mode
from llm_skill_robot.safety.real_arm_safety import (
    load_real_arm_safety,
    print_real_arm_safety,
    validate_full_grasp_step,
    validate_motion_summary,
)
from llm_skill_robot.safety.tabletop_safety import TabletopSafety
from llm_skill_robot.utils import get_config_dir


def _load_yaml(filename: str) -> dict:
    path = get_config_dir() / filename
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _value(value: Any, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def camera_readiness(backend: Any) -> dict[str, Any]:
    """Return the exact camera inputs required to build a grasp point cloud."""
    config = getattr(backend, "config", {}) or {}
    config = config if isinstance(config, dict) else {}
    depth_getter = getattr(backend, "_get_latest_depth_image", None)
    depth_frame = depth_getter() if callable(depth_getter) else None
    return {
        "rgb_received": getattr(backend, "latest_rgb_msg", None) is not None,
        "aligned_depth_received": getattr(backend, "latest_depth_msg", None) is not None,
        "depth_convertible": depth_frame is not None,
        "camera_info_received": getattr(backend, "latest_camera_info_msg", None) is not None,
        "rgb_topic": config.get("rgb_topic"),
        "aligned_depth_topic": config.get("depth_topic"),
        "camera_info_topic": config.get("camera_info_topic"),
        "depth_frame": config.get("depth_frame"),
    }


def wait_for_grasp_camera_inputs(
    backend: Any,
    node: Any,
    rclpy_module: Any,
    timeout_sec: float = 10.0,
) -> dict[str, Any]:
    """Spin until all point-cloud inputs are usable, returning diagnostics either way."""
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        diagnostics = camera_readiness(backend)
        if diagnostics["rgb_received"] and diagnostics["depth_convertible"] and diagnostics["camera_info_received"]:
            diagnostics["ready"] = True
            return diagnostics
        rclpy_module.spin_once(node, timeout_sec=0.1)
    diagnostics = camera_readiness(backend)
    diagnostics["ready"] = False
    return diagnostics


def print_full_grasp_report(report: dict[str, Any]) -> None:
    """Print a compact operator report; detailed payloads remain in the audit log."""
    print("\nFINAL FULL-GRASP REPORT")
    print(f"Status: {report.get('status', 'UNKNOWN')}")
    if report.get("abort_reason"):
        print(f"Stopped safely: {report['abort_reason']}")
    print(f"Object: {report.get('object_id') or 'not selected'}")
    candidate_id = report.get("candidate_id") or "not selected"
    score = report.get("candidate_score")
    score_text = f"{float(score):.3f}" if isinstance(score, (int, float)) else "n/a"
    print(f"Candidate: {candidate_id} (score: {score_text})")

    results = report.get("results", {}) or {}
    point_cloud = results.get("build_object_point_cloud")
    if isinstance(point_cloud, dict):
        print(
            "Point cloud: "
            f"{point_cloud.get('status', 'UNKNOWN')} — {point_cloud.get('message', '')}"
        )
    for step in (
        "move_to_observe",
        "open_gripper",
        "move_to_pregrasp",
        "approach_grasp",
        "close_for_grasp",
        "retreat_grasp",
        "lift_object",
        "verify_grasp",
    ):
        result = results.get(step)
        if not isinstance(result, dict):
            continue
        if step == "verify_grasp":
            print(f"Verification: {result.get('status', 'UNCERTAIN')} — {result.get('message', '')}")
            continue
        outcome = "SUCCESS" if result.get("success") else "BLOCKED/FAILED"
        print(f"{step}: {outcome} — {result.get('message', '')}")
    if report.get("audit_log_path"):
        print(f"Audit log: {report['audit_log_path']}")


class FullGraspSupervisedDemoController:
    """Coordinate one full grasp while requiring separate approval for each action."""

    def __init__(
        self,
        *,
        hardware_mode,
        real_arm_safety,
        perception_pipeline,
        point_cloud_builder,
        grasp_planner,
        motion_backend,
        gripper_backend,
        grasp_verifier: GraspVerifier,
        grasping_config: dict | None = None,
        transformer=None,
        state_machine: FullGraspStateMachine | None = None,
        audit_logger=None,
        input_fn: Callable[[str], str] = input,
        pose_builder: Callable[..., dict] = prepare_candidate_preview,
        tabletop_safety: TabletopSafety | None = None,
    ) -> None:
        self.hardware_mode = hardware_mode
        self.real_arm_safety = real_arm_safety
        self.perception_pipeline = perception_pipeline
        self.point_cloud_builder = point_cloud_builder
        self.grasp_planner = grasp_planner
        self.motion_backend = motion_backend
        self.gripper_backend = gripper_backend
        self.grasp_verifier = grasp_verifier
        self.grasping_config = grasping_config or {}
        self.transformer = transformer
        self.state = state_machine or FullGraspStateMachine()
        self.audit_logger = audit_logger
        self.input_fn = input_fn
        self.pose_builder = pose_builder
        self.tabletop_safety = tabletop_safety
        self.results: dict[str, Any] = {}
        self.abort_reason: str | None = None

    def run(self, query: str, *, retreat_mode: str = "retreat") -> dict:
        """Run one supervised sequence; no confirmation is reused across steps."""
        if self.real_arm_safety.execution_mode != "full_grasp_supervised":
            return self._abort("real_arm_safety must use full_grasp_supervised mode.")
        if not query.strip():
            return self._abort("Object query must not be empty.")
        if not self._move_to_observe():
            return self._report()

        detection = self.perception_pipeline.detect_object(query.strip(), require_pose=False)
        self.results["detect_object"] = detection
        selected_object = self._select_object(detection)
        if selected_object is None:
            return self._abort("No object detected.")
        object_id = str(_value(selected_object, "object_id", ""))
        if not object_id:
            return self._abort("Detected object has no object_id.")
        self.state.update_context(selected_object_id=object_id)
        if not self._transition(self.state.OBJECT_DETECTED, "Object detected."):
            return self._report()

        point_cloud = self._build_point_cloud(selected_object)
        self.results["build_object_point_cloud"] = point_cloud
        if not point_cloud.get("success") or not point_cloud.get("point_cloud_path"):
            return self._abort(f"Point cloud failed: {point_cloud.get('message', 'unavailable')}")
        self.state.update_context(point_cloud_path=str(point_cloud["point_cloud_path"]))
        if not self._transition(self.state.POINT_CLOUD_BUILT, "Object point cloud built."):
            return self._report()

        generation = self.grasp_planner.generate_candidates(point_cloud["point_cloud_path"])
        selection = self.select_safe_grasp_candidate(
            generation.get("candidates", []), point_cloud
        )
        if not selection["success"]:
            return self._abort(selection["message"])
        candidate = selection["candidate"]
        score = selection["score"]
        candidate_id = selection["candidate_id"]
        grasp_plan = selection["grasp_plan"]
        self.state.update_context(
            selected_grasp_candidate_id=candidate_id,
            selected_grasp_score=score,
        )
        if not self._transition(self.state.GRASP_GENERATED, "GraspPlan generated."):
            return self._report()
        print("\nSelected grasp candidate for review:")
        pprint.pprint({"candidate_id": candidate_id, "score": score, "plan": grasp_plan}, sort_dicts=False)
        if self.input_fn("Type YES to approve this grasp candidate:\n> ") != "YES":
            return self._abort("Grasp candidate review was not approved.")

        if not self._open_gripper():
            return self._report()
        if not self._transition(self.state.PREGRASP_PLANNED, "Pregrasp step approved for planning."):
            return self._report()
        if not self._execute_arm_step("move_to_pregrasp", object_id, grasp_plan):
            return self._report()
        if not self._transition(self.state.AT_PREGRASP, "Reached pregrasp."):
            return self._report()

        if not self._transition(self.state.APPROACH_PLANNED, "Approach step approved for planning."):
            return self._report()
        if not self._execute_arm_step("approach_grasp", object_id, grasp_plan):
            return self._report()
        if not self._transition(self.state.AT_GRASP_POSE, "Reached grasp pose."):
            return self._report()

        if not self._close_for_grasp():
            return self._report()
        if not self._transition(self.state.GRIPPER_CLOSED, "Contact close completed."):
            return self._report()

        if not self._transition(self.state.RETREAT_PLANNED, "Retreat/lift step approved for planning."):
            return self._report()
        retreat_skill = "lift_object" if retreat_mode == "lift" else "retreat_grasp"
        if not self._execute_arm_step(retreat_skill, object_id, grasp_plan):
            return self._report()
        if not self._transition(self.state.RETREATED, f"{retreat_skill} completed."):
            return self._report()

        perception_recheck = self.perception_pipeline.detect_object(query.strip(), require_pose=False)
        gripper_state = self.gripper_backend.get_gripper_state()
        verification = self.grasp_verifier.verify_grasp(
            object_id, gripper_state=gripper_state, perception_result=perception_recheck
        )
        self.results["verify_grasp"] = verification
        if not self._transition(self.state.GRASP_VERIFIED, verification["message"]):
            return self._report()
        self._transition(self.state.DONE, "Supervised grasp sequence completed.")
        return self._report()

    def select_safe_grasp_candidate(
        self,
        candidates: list[Any],
        point_cloud: dict[str, Any],
    ) -> dict[str, Any]:
        """Select and prepare one candidate using the demo's complete safety screen."""
        eligible = self._eligible_candidates(candidates)
        if not eligible:
            return {
                "success": False,
                "message": "No valid GraspGenX candidate meets the minimum grasp score.",
            }
        screening = self._screen_pregrasp_candidates(eligible, point_cloud)
        if screening is None:
            return {
                "success": False,
                "message": "No candidate produced a safe, reachable full grasp trajectory.",
            }
        candidate, selected_variant, attempts = screening
        score = float(_value(candidate, "score", 0.0))
        candidate_id = str(_value(candidate, "candidate_id", "unknown"))
        try:
            poses = self._prepare_poses(candidate, point_cloud)
        except Exception as exc:
            return {
                "success": False,
                "message": f"Could not prepare base_link GraspPlan: {exc}",
            }
        grasp_plan = {
            "pregrasp_pose": poses.get("pregrasp"),
            "grasp_pose": poses.get("grasp"),
            "retreat_pose": poses.get("retreat"),
            "grasp_score": score,
            "candidate_id": candidate_id,
        }
        if not all(self._pose_is_base_link(grasp_plan[key]) for key in (
            "pregrasp_pose", "grasp_pose", "retreat_pose"
        )):
            return {
                "success": False,
                "message": "GraspPlan poses must all be in base_link.",
            }
        self._configure_motion_planning(selected_variant)
        screening_result = {
            "success": True,
            "selected_candidate_id": candidate_id,
            "selected_variant": selected_variant,
            "attempt_count": len(attempts),
        }
        self.results["candidate_screening"] = screening_result
        return {
            "success": True,
            "message": "Selected a safe, reachable full-grasp candidate.",
            "candidate": candidate,
            "candidate_id": candidate_id,
            "score": score,
            "selected_variant": selected_variant,
            "attempts": attempts,
            "poses": poses,
            "grasp_plan": grasp_plan,
            "candidate_screening": screening_result,
        }

    def _open_gripper(self) -> bool:
        self._show_action("open_gripper", "YES")
        if self.input_fn("Type YES to open the REAL gripper:\n> ") != "YES":
            self._abort("Open-gripper confirmation was rejected.")
            return False
        result = self.gripper_backend.open_gripper(confirmed=True)
        self.results["open_gripper"] = result
        if not result.get("success", False):
            self._abort(f"open_gripper failed: {result.get('message', 'unknown error')}")
            return False
        return True

    def _move_to_observe(self) -> bool:
        self._show_action("move_to_observe", "EXECUTE")
        confirmation = self.input_fn(
            "Type EXECUTE to move the REAL UR5 to the observation pose:\n> "
        )
        if confirmation != "EXECUTE":
            self._abort("move_to_observe confirmation was rejected.")
            return False
        method = getattr(self.motion_backend, "move_to_observe", None)
        if not callable(method):
            self._abort("Motion backend does not implement move_to_observe.")
            return False
        result = method(confirmation=confirmation)
        self.results["move_to_observe"] = result
        if not result.get("success", False):
            self._abort(f"move_to_observe failed: {result.get('message', 'unknown error')}")
            return False
        return True

    def _execute_arm_step(self, skill_id: str, object_id: str, grasp_plan: dict) -> bool:
        self._show_action(skill_id, "EXECUTE")
        confirmation = self.input_fn("Type EXECUTE to move the REAL UR5:\n> ")
        if confirmation != "EXECUTE":
            self._abort(f"{skill_id} confirmation was rejected.")
            return False
        method = getattr(self.motion_backend, skill_id, None)
        if not callable(method):
            self._abort(f"Motion backend does not implement {skill_id}.")
            return False
        result = method(object_id, {object_id: grasp_plan}, confirmation=confirmation)
        self.results[skill_id] = result
        if not result.get("success", False):
            self._abort(f"{skill_id} failed: {result.get('message', 'unknown error')}")
            return False
        return True

    def _close_for_grasp(self) -> bool:
        self._show_action("close_for_grasp", "YES")
        confirmation = self.input_fn("Type YES to close the REAL gripper on the object:\n> ")
        if confirmation != "YES":
            self._abort("Contact-close confirmation was rejected.")
            return False
        result = self.gripper_backend.close_for_grasp(confirmation=confirmation)
        self.results["close_for_grasp"] = result
        if not result.get("success", False):
            self._abort(f"close_for_grasp failed: {result.get('message', 'unknown error')}")
            return False
        return True

    def _build_point_cloud(self, selected_object: Any) -> dict:
        camera = getattr(self.perception_pipeline, "backend", None)
        depth_getter = getattr(camera, "_get_latest_depth_image", None)
        config = getattr(camera, "config", {}) or {}
        config = config if isinstance(config, dict) else {}
        result = self.point_cloud_builder.build(
            selected_object,
            depth_getter() if callable(depth_getter) else None,
            getattr(camera, "latest_camera_info_msg", None),
            frame=config.get("depth_frame", "base_link"),
        )
        if not result.get("success"):
            result["camera_diagnostics"] = camera_readiness(camera)
        return result

    def _select_object(self, detection: dict) -> Any | None:
        objects = detection.get("objects", []) if isinstance(detection, dict) else []
        return max(objects, key=lambda value: float(_value(value, "confidence", 0.0) or 0.0), default=None)

    def _eligible_candidates(self, candidates: list[Any]) -> list[Any]:
        minimum = float(self.real_arm_safety.grasp.get("min_grasp_score", 0.0))
        return sorted([
            candidate for candidate in candidates
            if _value(candidate, "valid", True) is not False
            and float(_value(candidate, "score", float("-inf"))) >= minimum
        ], key=lambda value: float(_value(value, "score", 0.0)), reverse=True)

    def _screen_pregrasp_candidates(
        self,
        candidates: list[Any],
        point_cloud: dict[str, Any],
    ) -> tuple[Any, str, list[dict[str, Any]]] | None:
        """Reuse restricted-demo plan-only screening before selecting a real grasp."""
        preview = self.grasping_config.get("plan_only_preview", {}) or {}
        screening_config = preview.get("candidate_screening", {}) or {}
        if not screening_config.get("enabled", False):
            candidate = candidates[0] if candidates else None
            return (candidate, "default", []) if candidate is not None else None

        moveit_backend = getattr(self.motion_backend, "moveit_backend", None)
        if moveit_backend is None:
            self._audit(
                "full_grasp_candidate_screening_failed",
                reason="Grasp motion backend does not expose MoveIt plan-only support.",
            )
            return None
        preferred_candidates = self._rank_grasp_preferences(candidates, point_cloud, moveit_backend)
        if not preferred_candidates:
            self._audit(
                "full_grasp_candidate_screening_failed",
                reason="No candidate satisfied the configured grasp orientation preferences.",
            )
            return None
        selector = PregraspRealDemoController(
            hardware_mode=self.hardware_mode,
            real_arm_safety=self.real_arm_safety,
            perception_pipeline=self.perception_pipeline,
            point_cloud_builder=self.point_cloud_builder,
            grasp_planner=self.grasp_planner,
            moveit_backend=moveit_backend,
            transformer=self.transformer,
            grasping_config=self.grasping_config,
            audit_logger=self.audit_logger,
        )
        all_attempts: list[dict[str, Any]] = []
        for candidate in preferred_candidates:
            selection = selector.select_restricted_pregrasp_plan(
                [candidate],
                point_cloud,
                preserve_candidate_order=True,
                report_rejections=False,
            )
            if selection is None:
                all_attempts.append({
                    "candidate_id": str(_value(candidate, "candidate_id", "unknown")),
                    "stage": "pregrasp",
                    "accepted": False,
                    "reason": "No safe reachable pregrasp trajectory.",
                })
                continue
            all_attempts.extend(selection.attempts)
            full_check = self._screen_grasp_and_retreat_targets(
                candidate, point_cloud, moveit_backend, selection.variant
            )
            all_attempts.extend(full_check["attempts"])
            if full_check["allowed"]:
                self._audit(
                    "full_grasp_candidate_screening",
                    selected_candidate_id=str(_value(candidate, "candidate_id", "unknown")),
                    selected_variant=selection.variant,
                    attempts=all_attempts,
                )
                return candidate, selection.variant, all_attempts
        self._audit("full_grasp_candidate_screening_failed", attempts=all_attempts)
        return None

    def _screen_grasp_and_retreat_targets(
        self,
        candidate: Any,
        point_cloud: dict[str, Any],
        moveit_backend: Any,
        selected_variant: str,
    ) -> dict[str, Any]:
        """Reject candidates whose grasp or retreat targets have no plan-only IK."""
        try:
            poses = self._prepare_poses(candidate, point_cloud)
        except Exception as exc:
            return {"allowed": False, "attempts": [{
                "candidate_id": str(_value(candidate, "candidate_id", "unknown")),
                "stage": "pose_preparation", "accepted": False, "reason": str(exc),
            }]}
        if not all(self._pose_is_base_link(poses.get(name)) for name in ("pregrasp", "grasp", "retreat")):
            return {"allowed": False, "attempts": [{
                "candidate_id": str(_value(candidate, "candidate_id", "unknown")),
                "stage": "pose_frame", "accepted": False, "reason": "All grasp poses must be in base_link.",
            }]}

        tabletop_check = self._validate_tabletop_poses(poses)
        if not tabletop_check["allowed"]:
            return {"allowed": False, "attempts": [{
                "candidate_id": str(_value(candidate, "candidate_id", "unknown")),
                "stage": "tabletop_clearance", "accepted": False,
                "reason": tabletop_check["reason"],
            }]}

        preview = self.grasping_config.get("plan_only_preview", {}) or {}
        screening = preview.get("candidate_screening", {}) or {}
        velocity = min(float(preview.get("velocity_scale", 0.05)), self.real_arm_safety.limits.max_velocity_scale)
        acceleration = min(float(preview.get("acceleration_scale", 0.05)), self.real_arm_safety.limits.max_acceleration_scale)
        variant_options = (
            planning_variant_kwargs(selected_variant, screening)
            if selected_variant in {"full_pose", "relaxed_orientation", "position_only"} else {}
        )
        checks = (
            ("approach_grasp", "grasp", "pregrasp", "grasp", {**moveit_stage_planning_kwargs(preview, "grasp"), **variant_options}),
            ("retreat_grasp", "retreat", "grasp", "retreat", moveit_stage_planning_kwargs(preview, "retreat")),
        )
        attempts: list[dict[str, Any]] = []
        candidate_id = str(_value(candidate, "candidate_id", "unknown"))
        score = float(_value(candidate, "score", 0.0))
        for skill_id, target_name, reference_name, stage, options in checks:
            target = poses[target_name]
            reference = poses[reference_name]
            distance = self._pose_distance(reference, target)
            try:
                result = moveit_backend.plan_to_pose(
                    frame=target["frame"], position=target["position"], orientation=target["orientation"],
                    velocity_scale=velocity, acceleration_scale=acceleration,
                    skill_id=skill_id, **options,
                )
            except Exception as exc:
                attempts.append({"candidate_id": candidate_id, "stage": stage, "accepted": False, "reason": f"MoveIt exception: {exc}"})
                return {"allowed": False, "attempts": attempts}
            if not isinstance(result, dict) or not result.get("success", False):
                message = result.get("message", "MoveIt planning failed.") if isinstance(result, dict) else "MoveIt planning failed."
                attempts.append({"candidate_id": candidate_id, "stage": stage, "accepted": False, "reason": str(message)})
                return {"allowed": False, "attempts": attempts}
            summary_builder = getattr(self.motion_backend, "_motion_summary", None)
            if callable(summary_builder):
                summary = summary_builder(result.get("summary", {}), skill_id, target, velocity, acceleration, distance, score)
            else:
                source = dict(result.get("summary", {}) or {})
                summary = {**source, "skill_id": skill_id, "target_pose": target, "distance_m": distance, "grasp_score": score, "velocity_scale": velocity, "acceleration_scale": acceleration}
            trajectory = self.motion_backend.trajectory_validator.validate_motion_plan_summary(summary)
            safety = validate_motion_summary(summary, self.real_arm_safety)
            full_step = validate_full_grasp_step(skill_id, summary, self.real_arm_safety)
            if trajectory.get("decision") != "ALLOW" or not safety.get("allowed", False) or not full_step.get("allowed", False):
                reasons = list(trajectory.get("reasons", []))
                if not safety.get("allowed", False):
                    reasons.append(safety["reason"])
                if not full_step.get("allowed", False):
                    reasons.append(full_step["reason"])
                attempts.append({"candidate_id": candidate_id, "stage": stage, "accepted": False, "reason": "; ".join(dict.fromkeys(reasons))})
                return {"allowed": False, "attempts": attempts}
            attempts.append({"candidate_id": candidate_id, "stage": stage, "accepted": True, "plan_id": result.get("plan_id")})
        return {"allowed": True, "attempts": attempts}

    def _validate_tabletop_poses(self, poses: dict[str, Any]) -> dict[str, Any]:
        if self.tabletop_safety is None:
            return {"allowed": True}
        plan = {
            "pregrasp_pose": poses.get("pregrasp"),
            "grasp_pose": poses.get("grasp"),
            "retreat_pose": poses.get("retreat"),
        }
        return self.tabletop_safety.validate_grasp_plan(plan)

    def _prepare_poses(
        self, candidate: Any, point_cloud: dict[str, Any]
    ) -> dict[str, Any]:
        """Build common poses and apply configured approach-axis backoff."""
        poses = self.pose_builder(
            candidate,
            point_cloud,
            self.grasping_config.get("plan_only_preview", {}),
            "base_link",
            self.transformer,
        )
        if self.tabletop_safety is not None:
            poses = self.tabletop_safety.apply_grasp_contact_backoff(poses)
            if self.tabletop_safety.grasp_contact_backoff_m > 0.0:
                self.results["grasp_contact_backoff"] = {
                    "enabled": True,
                    "distance_m": self.tabletop_safety.grasp_contact_backoff_m,
                    "direction": "toward_pregrasp_along_approach_axis",
                }
        return poses

    @staticmethod
    def _pose_distance(first: dict[str, Any], second: dict[str, Any]) -> float:
        return sum((float(first["position"][axis]) - float(second["position"][axis])) ** 2 for axis in ("x", "y", "z")) ** 0.5

    def _rank_grasp_preferences(
        self,
        candidates: list[Any],
        point_cloud: dict[str, Any],
        moveit_backend: Any,
    ) -> list[Any]:
        """Rank candidates using the restricted-demo top-down/face/yaw preferences."""
        preference_controller = RestrictedRealHardwareDemoController(
            hardware_mode=self.hardware_mode,
            real_arm_safety=self.real_arm_safety,
            perception_pipeline=self.perception_pipeline,
            point_cloud_builder=self.point_cloud_builder,
            grasp_planner=self.grasp_planner,
            moveit_backend=moveit_backend,
            gripper_backend=self.gripper_backend,
            grasping_config=self.grasping_config,
            transformer=self.transformer,
            audit_logger=self.audit_logger,
        )
        preference_controller.state.point_cloud = point_cloud
        ranked = preference_controller._apply_grasp_preferences(
            [{"candidate": candidate} for candidate in candidates]
        )
        diagnostics = [
            {
                "candidate_id": str(_value(item["candidate"], "candidate_id", "unknown")),
                "preference_score": item["preference"].get("score"),
                "reason": item["preference"].get("reason"),
            }
            for item in ranked
        ]
        self.results["grasp_preference_ranking"] = {
            "accepted_count": len(ranked),
            "ranked_candidates": diagnostics,
        }
        return [item["candidate"] for item in ranked]

    def _configure_motion_planning(self, selected_variant: str) -> None:
        """Use the same MoveIt options that passed the plan-only pregrasp screen."""
        setter = getattr(self.motion_backend, "set_planning_kwargs", None)
        if not callable(setter):
            return
        preview = self.grasping_config.get("plan_only_preview", {}) or {}
        screening = preview.get("candidate_screening", {}) or {}
        variant_options = (
            planning_variant_kwargs(selected_variant, screening)
            if selected_variant in {"full_pose", "relaxed_orientation", "position_only"}
            else {}
        )
        policy = GraspMotionPolicy.from_grasping_config(
            self.grasping_config,
            max_velocity_scale=self.real_arm_safety.limits.max_velocity_scale,
            max_acceleration_scale=self.real_arm_safety.limits.max_acceleration_scale,
        ).with_pregrasp_variant(variant_options)
        for skill_id, options in policy.planning_kwargs_by_skill.items():
            setter(skill_id, options)

    @staticmethod
    def _pose_is_base_link(pose: Any) -> bool:
        return isinstance(pose, dict) and (pose.get("frame") or pose.get("frame_id")) == "base_link"

    def _transition(self, next_state: str, reason: str) -> bool:
        transition = self.state.transition(next_state, reason)
        if transition["allowed"]:
            return True
        self._abort(transition["reason"])
        return False

    def _abort(self, reason: str) -> dict:
        self.abort_reason = reason
        self.state.abort(reason)
        self._audit("full_grasp_supervised_aborted", reason=reason)
        return self._report()

    def _show_action(self, action: str, confirmation: str) -> None:
        print(f"\nCurrent state: {self.state.current_state()}")
        print(f"Next action: {action}")
        print("Safety: one real hardware step only; no confirmation is reused.")
        print(f"Required confirmation: {confirmation}")

    def _audit(self, event: str, **fields: Any) -> None:
        if self.audit_logger is not None and hasattr(self.audit_logger, "log_hardware_safety_event"):
            self.audit_logger.log_hardware_safety_event(event, **fields)

    def _report(self) -> dict:
        report = {
            "status": self.state.current_state(),
            "abort_reason": self.abort_reason,
            "object_id": self.state.selected_object_id,
            "candidate_id": self.state.selected_grasp_candidate_id,
            "candidate_score": self.state.selected_grasp_score,
            "results": dict(self.results),
            "state_context": self.state.context(),
            "audit_log_path": str(getattr(self.audit_logger, "log_file", "")) or None,
        }
        return report


def main() -> None:
    try:
        import rclpy
    except ImportError:
        print("Error: rclpy is unavailable. Source the ROS 2 environment first.", file=sys.stderr)
        return

    hardware_mode = load_hardware_mode()
    real_arm_safety = load_real_arm_safety()
    if hardware_mode.arm.execution_mode != "full_grasp_supervised" or real_arm_safety.execution_mode != "full_grasp_supervised":
        print("Error: full_grasp_supervised must be enabled in hardware_mode.yaml and real_arm_safety.yaml.", file=sys.stderr)
        return
    print("FULL REAL GRASP SUPERVISED MODE")
    print("This can move the REAL UR5 and close the REAL gripper.")
    print("Each step requires confirmation.")
    print_hardware_mode(hardware_mode)
    print_real_arm_safety(real_arm_safety)

    perception_config = _load_yaml("perception.yaml")
    grasping_config = _load_yaml("grasping.yaml").get("grasping", {})
    tabletop_safety = TabletopSafety.from_config(_load_yaml("tabletop_safety.yaml"))
    gripper_config = _load_yaml("gripper.yaml")
    audit_logger = AuditLogger("full_grasp_supervised_audit.log", hardware_mode=hardware_mode)
    node = None
    try:
        rclpy.init()
        node = rclpy.create_node("ros_full_grasp_supervised_demo")
        transformer = TFPoseTransformer(node, target_frame="base_link")
        perception = create_perception_pipeline(node=node, config=perception_config)
        print("\nWaiting for D435i RGB, aligned depth, and camera_info messages...")
        camera_diagnostics = wait_for_grasp_camera_inputs(perception.backend, node, rclpy)
        if not camera_diagnostics["ready"]:
            print("Error: D435i point-cloud inputs are not ready; grasp demo will not start.", file=sys.stderr)
            pprint.pprint(camera_diagnostics, sort_dicts=False)
            print(
                "Check the configured aligned-depth and camera-info topics, then restart the demo.",
                file=sys.stderr,
            )
            return
        point_cloud_builder = ObjectPointCloudBuilder(grasping_config.get("point_cloud", {}))
        planner = GraspGenXGraspPlanner(GraspGenXSubprocessAdapter(grasping_config.get("graspgenx_subprocess", {})))
        moveit = UR5MoveItPlanBackend(node=node, dry_run=False, execution_mode="real_robot", hardware_mode=hardware_mode)
        motion = GraspExecutionBackend(
            moveit, real_arm_safety, audit_logger=audit_logger,
            tabletop_safety=tabletop_safety,
        )
        gripper = Robotiq2F140RealBackend(node=node, config=gripper_config, hardware_mode=hardware_mode, real_arm_safety=real_arm_safety, audit_logger=audit_logger)
        controller = FullGraspSupervisedDemoController(
            hardware_mode=hardware_mode,
            real_arm_safety=real_arm_safety,
            perception_pipeline=perception,
            point_cloud_builder=point_cloud_builder,
            grasp_planner=planner,
            motion_backend=motion,
            gripper_backend=gripper,
            grasp_verifier=GraspVerifier(),
            grasping_config=grasping_config,
            transformer=transformer,
            audit_logger=audit_logger,
            tabletop_safety=tabletop_safety,
        )
        report = controller.run(input("Enter object query:\n> "))
        print_full_grasp_report(report)
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
