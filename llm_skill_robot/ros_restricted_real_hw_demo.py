"""Unified restricted real-hardware demo for D435i, GraspGenX, UR5, and gripper."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import pprint
import sys
from typing import Any

from llm_skill_robot.core.audit_logger import AuditLogger
from llm_skill_robot.grasping.grasp_execution_backend import GraspExecutionBackend
from llm_skill_robot.grasping.graspgenx_grasp_planner import GraspGenXGraspPlanner, generate_candidates_with_context
from llm_skill_robot.grasping.graspgenx_subprocess_adapter import GraspGenXSubprocessAdapter
from llm_skill_robot.grasping.parallel_jaw_symmetry import canonicalize_sparse_parallel_jaw_candidates
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
from llm_skill_robot.ros_graspgenx_plan_only_demo import (
    moveit_stage_planning_kwargs,
    planning_variant_kwargs,
    prepare_candidate_preview,
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
    validate_contact_close_allowed,
    validate_full_grasp_step,
    validate_motion_summary,
)
from llm_skill_robot.safety.tabletop_safety import TabletopSafety


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
    grasp_plan: dict[str, Any] | None = None
    at_pregrasp: bool = False
    at_grasp_pose: bool = False
    gripper_closed: bool = False


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
        tabletop_safety: TabletopSafety | None = None,
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
        self.transformer = transformer
        self.audit_logger = audit_logger
        self.tabletop_safety = tabletop_safety
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
        self.grasp_motion = GraspExecutionBackend(
            moveit_backend,
            real_arm_safety,
            audit_logger=audit_logger,
            tabletop_safety=tabletop_safety,
            review_approval_provider=lambda _summary: (
                input("Approve the reviewed trajectory preview? Type YES:\n> ").strip()
                == "YES"
            ),
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
        self._reset_grasp_execution_state()
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
        self._reset_grasp_execution_state()
        self._audit(
            "restricted_real_hw_point_cloud_built",
            point_cloud_path=point_cloud.get("point_cloud_path"),
        )
        return DemoActionResult(True, "SUCCESS", "Object point cloud built.", point_cloud)

    def generate_grasp_candidates(self) -> DemoActionResult:
        if self.state.point_cloud is None:
            return self._blocked("Build an object point cloud before generating grasps.")
        result = generate_candidates_with_context(
            self.grasp_planner, self.state.point_cloud, self.transformer,
        )
        candidates = result.get("candidates", [])
        if not candidates:
            return self._blocked(f"No GraspGenX candidates: {result.get('message', 'none returned')}")
        self.state.candidates = candidates
        self.state.selected_grasp_plan = None
        self.state.last_planned_trajectory = None
        self.state.candidate_width_command = None
        self._reset_grasp_execution_state()
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
        preferred = self._apply_grasp_preferences(width_screened)
        if not preferred:
            return self._blocked("No candidate satisfies the configured grasp preferences.")
        selection = None
        selected_preferred_item = None
        full_grasp_plan = None
        screening_attempts: list[dict[str, Any]] = []
        for item in preferred:
            candidate = item["candidate"]
            candidate_selection = self.pregrasp_controller.select_restricted_pregrasp_plan(
                [item["candidate"]],
                self.state.point_cloud,
            )
            if candidate_selection is None:
                screening_attempts.append({
                    "candidate_id": _candidate_value(candidate, "candidate_id", "unknown"),
                    "stage": "pregrasp",
                    "accepted": False,
                    "reason": "No safe reachable pregrasp trajectory.",
                })
                continue
            try:
                candidate_grasp_plan = self._build_full_grasp_plan(candidate_selection.candidate)
            except (KeyError, TypeError, ValueError) as exc:
                screening_attempts.append({
                    "candidate_id": _candidate_value(candidate, "candidate_id", "unknown"),
                    "stage": "pose_preparation",
                    "accepted": False,
                    "reason": str(exc),
                })
                continue
            full_check = self._screen_grasp_and_retreat_targets(
                candidate_grasp_plan, candidate_selection.variant
            )
            screening_attempts.extend(full_check["attempts"])
            if not full_check["allowed"]:
                continue
            selection = candidate_selection
            selected_preferred_item = item
            full_grasp_plan = candidate_grasp_plan
            break
        if selection is None:
            self._audit("full_grasp_candidate_screening_failed", attempts=screening_attempts)
            return self._blocked("No candidate produced safe reachable pregrasp, grasp, and retreat trajectories.")
        width_command = selected_preferred_item["command"]
        self.state.selected_grasp_plan = selection
        self.state.last_planned_trajectory = selection.plan
        self.state.candidate_width_command = width_command
        self.state.grasp_plan = full_grasp_plan
        self._configure_full_grasp_motion_planning(selection.variant)
        self._audit(
            "full_grasp_candidate_screening",
            selected_candidate_id=self.state.grasp_plan["candidate_id"],
            selected_variant=selection.variant,
            attempts=screening_attempts,
        )
        print_selected_candidate(selection.candidate)
        preference = grasp_preference_score(
            selection.candidate,
            self._grasp_preference_config(),
        )
        print("\nGrasp preference diagnostics:")
        print(f"score: {preference.get('score')}")
        print(f"reason: {preference.get('reason')}")
        print("object planar_principal_axes:")
        pprint.pprint((self.state.point_cloud or {}).get("planar_principal_axes"), sort_dicts=False)
        print_motion_summary(selection.plan.summary)
        self._audit(
            "restricted_real_hw_grasp_candidate_reviewed",
            selected_candidate_id=_candidate_value(selection.candidate, "candidate_id", "unknown"),
            selected_variant=selection.variant,
            plan_id=selection.plan.plan_id,
            grasp_preference_score=preference.get("score"),
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

    def approach_grasp(
        self, confirmation: str | None = None, trajectory_review_approved: bool | None = None
    ) -> DemoActionResult:
        if not self.state.at_pregrasp:
            return self._blocked("Move to pregrasp before approaching the grasp pose.")
        result = self._execute_full_grasp_motion(
            "approach_grasp", confirmation, trajectory_review_approved
        )
        if result.success:
            self.state.at_grasp_pose = True
        return result

    def close_real_gripper_for_grasp(self, confirmation: str | None = None) -> DemoActionResult:
        if not self.state.at_grasp_pose:
            return self._blocked("Approach the grasp pose before closing on the object.")
        allowed = validate_contact_close_allowed(self.real_arm_safety)
        if not allowed["allowed"]:
            return self._blocked(allowed["reason"])
        if confirmation is None:
            confirmation = input("Type YES to close the REAL gripper on the object:\n> ").strip()
        if not confirmation_is_yes(confirmation):
            self._audit("real_gripper_step_requires_confirmation", real_gripper_step_requires_confirmation=True)
            return DemoActionResult(False, "CANCELLED", "Exact YES confirmation is required.")
        result = self.gripper_backend.close_for_grasp(confirmation=confirmation)
        outcome = self._gripper_result(result, "Real grasp contact-close command completed.")
        if outcome.success:
            self.state.gripper_closed = True
        return outcome

    def retreat_grasp(
        self, confirmation: str | None = None, trajectory_review_approved: bool | None = None
    ) -> DemoActionResult:
        if not self.state.gripper_closed:
            return self._blocked("Close the gripper on the object before retreating.")
        return self._execute_full_grasp_motion(
            "retreat_grasp", confirmation, trajectory_review_approved
        )

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
        outcome = self._result_from_backend(result)
        if outcome.success:
            self.state.at_pregrasp = True
            self.state.at_grasp_pose = False
            self.state.gripper_closed = False
        return outcome

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
            "full_grasp_ready": self.state.grasp_plan is not None,
            "at_pregrasp": self.state.at_pregrasp,
            "at_grasp_pose": self.state.at_grasp_pose,
            "gripper_closed": self.state.gripper_closed,
        }

    def _execute_full_grasp_motion(
        self, skill_id: str, confirmation: str | None, trajectory_review_approved: bool | None
    ) -> DemoActionResult:
        if self.state.selected_object is None or self.state.grasp_plan is None:
            return self._blocked("Review a valid GraspPlan before executing this motion.")
        object_id = str(_candidate_value(self.state.selected_object, "object_id", ""))
        method = getattr(self.grasp_motion, skill_id)
        result = method(
            object_id,
            self.state.grasp_plan,
            confirmation=confirmation,
            trajectory_review_approved=trajectory_review_approved,
        )
        return self._result_from_backend(result)

    def _reset_grasp_execution_state(self) -> None:
        self.state.grasp_plan = None
        self.state.at_pregrasp = False
        self.state.at_grasp_pose = False
        self.state.gripper_closed = False

    def _build_full_grasp_plan(self, candidate: Any) -> dict[str, Any]:
        poses = prepare_candidate_preview(
            candidate,
            self.state.point_cloud,
            self.grasping_config.get("plan_only_preview", {}),
            "base_link",
            self.transformer,
        )
        if self.tabletop_safety is not None:
            width = _candidate_value(candidate, "gripper_width_m", None)
            poses = self.tabletop_safety.apply_grasp_contact_backoff(
                poses, gripper_width_m=width,
            )
            if self.tabletop_safety.grasp_contact_backoff_m > 0.0:
                self._audit(
                    "grasp_contact_backoff_applied",
                    distance_m=self.tabletop_safety.grasp_contact_backoff_m,
                    gripper_width_m=width,
                    direction="toward_pregrasp_along_approach_axis",
                )
        return {
            "pregrasp_pose": poses["pregrasp"],
            "grasp_pose": poses["grasp"],
            "retreat_pose": poses["retreat"],
            "grasp_score": float(_candidate_value(candidate, "score", 0.0)),
            "candidate_id": _candidate_value(candidate, "candidate_id", "unknown"),
        }

    def _screen_grasp_and_retreat_targets(
        self, grasp_plan: dict[str, Any], selected_variant: str
    ) -> dict[str, Any]:
        """Require plan-only IK for the same grasp and retreat stages as full demo."""
        preview = self.grasping_config.get("plan_only_preview", {}) or {}
        screening = preview.get("candidate_screening", {}) or {}
        velocity, acceleration = self._restricted_scales()
        variant_options = (
            planning_variant_kwargs(selected_variant, screening)
            if selected_variant in {"full_pose", "relaxed_orientation", "position_only"}
            else {}
        )
        checks = (
            ("approach_grasp", "grasp_pose", "pregrasp_pose", "grasp", {
                **moveit_stage_planning_kwargs(preview, "grasp"), **variant_options,
            }),
            ("retreat_grasp", "retreat_pose", "grasp_pose", "retreat",
             moveit_stage_planning_kwargs(preview, "retreat")),
        )
        attempts: list[dict[str, Any]] = []
        candidate_id = grasp_plan["candidate_id"]
        if self.tabletop_safety is not None:
            tabletop = self.tabletop_safety.validate_grasp_plan(grasp_plan)
            if not tabletop["allowed"]:
                return {
                    "allowed": False,
                    "attempts": [{
                        "candidate_id": candidate_id,
                        "stage": "tabletop_clearance",
                        "accepted": False,
                        "reason": tabletop["reason"],
                    }],
                }
        for skill_id, target_key, reference_key, stage, options in checks:
            target = grasp_plan[target_key]
            reference = grasp_plan[reference_key]
            distance = math.sqrt(sum(
                (float(target["position"][axis]) - float(reference["position"][axis])) ** 2
                for axis in ("x", "y", "z")
            ))
            try:
                result = self.moveit_backend.plan_to_pose(
                    frame=target["frame"], position=target["position"],
                    orientation=target["orientation"], velocity_scale=velocity,
                    acceleration_scale=acceleration, skill_id=skill_id, **options,
                )
            except Exception as exc:
                attempts.append({"candidate_id": candidate_id, "stage": stage,
                                 "accepted": False, "reason": f"MoveIt exception: {exc}"})
                return {"allowed": False, "attempts": attempts}
            if not isinstance(result, dict) or not result.get("success", False):
                message = result.get("message", "MoveIt planning failed.") if isinstance(result, dict) else "MoveIt planning failed."
                attempts.append({"candidate_id": candidate_id, "stage": stage,
                                 "accepted": False, "reason": str(message)})
                return {"allowed": False, "attempts": attempts}
            summary = self.grasp_motion._motion_summary(
                result.get("summary", {}), skill_id, target, velocity, acceleration,
                distance, grasp_plan["grasp_score"],
            )
            trajectory = self.grasp_motion.trajectory_validator.validate_motion_plan_summary(summary)
            safety = validate_motion_summary(summary, self.real_arm_safety)
            full_step = validate_full_grasp_step(skill_id, summary, self.real_arm_safety)
            if trajectory["decision"] != "ALLOW" or not safety["allowed"] or not full_step["allowed"]:
                reasons = list(trajectory["reasons"])
                if not safety["allowed"]:
                    reasons.append(safety["reason"])
                if not full_step["allowed"]:
                    reasons.append(full_step["reason"])
                attempts.append({"candidate_id": candidate_id, "stage": stage,
                                 "accepted": False, "reason": "; ".join(dict.fromkeys(reasons))})
                return {"allowed": False, "attempts": attempts}
            attempts.append({"candidate_id": candidate_id, "stage": stage,
                             "accepted": True, "plan_id": result.get("plan_id")})
        return {"allowed": True, "attempts": attempts}

    def _configure_full_grasp_motion_planning(self, selected_variant: str) -> None:
        """Use the same per-stage MoveIt settings as the full grasp demo.

        In particular, the final approach and retreat use the configured Pilz
        LIN planner rather than falling back to the generic MoveIt planner.
        """
        preview = self.grasping_config.get("plan_only_preview", {}) or {}
        screening = preview.get("candidate_screening", {}) or {}
        variant_options = (
            planning_variant_kwargs(selected_variant, screening)
            if selected_variant in {"full_pose", "relaxed_orientation", "position_only"}
            else {}
        )
        self.grasp_motion.set_planning_kwargs(
            "approach_grasp", {
                **moveit_stage_planning_kwargs(preview, "grasp"),
                **variant_options,
            }
        )
        self.grasp_motion.set_planning_kwargs(
            "retreat_grasp", moveit_stage_planning_kwargs(preview, "retreat")
        )

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

    def _apply_grasp_preferences(
        self,
        width_screened: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        config = self._grasp_preference_config()
        source_items = {
            id(item["candidate"]): item for item in width_screened
        }
        primaries, fallbacks = canonicalize_sparse_parallel_jaw_candidates(
            [item["candidate"] for item in width_screened],
            config.get("current_gripper_closing_axis"),
        ) if config.get("parallel_jaw_symmetry_enabled", True) else (
            [item["candidate"] for item in width_screened], []
        )
        by_candidate_id = {
            str(_candidate_value(item["candidate"], "candidate_id", "")): item
            for item in width_screened
        }
        normalized_items = []
        for candidate in primaries + fallbacks:
            metadata = _candidate_value(candidate, "metadata", {}) or {}
            source_id = str(
                metadata.get("parallel_jaw_symmetry_base_candidate_id")
                or _candidate_value(candidate, "candidate_id", "")
            )
            source = source_items.get(id(candidate)) or by_candidate_id.get(source_id, {})
            normalized_items.append({**source, "candidate": candidate})
        scored = []
        rejected = []
        for item in normalized_items:
            preference = grasp_preference_score(item["candidate"], config)
            if not preference["allowed"]:
                rejected.append(
                    {
                        "candidate_id": _candidate_value(
                            item["candidate"], "candidate_id", "unknown"
                        ),
                        "reason": preference["reason"],
                    }
                )
                continue
            scored.append({**item, "preference": preference})
        if rejected:
            self._audit(
                "restricted_real_hw_grasp_preference_rejected",
                rejected=rejected,
            )
        scored.sort(
            key=lambda item: (
                1 if (_candidate_value(item["candidate"], "metadata", {}) or {}).get(
                    "parallel_jaw_symmetry_role"
                ) == "fallback" else 0,
                -float(item["preference"].get("score", 0.0)),
                -float(_candidate_value(item["candidate"], "score", 0.0) or 0.0),
            )
        )
        self._audit(
            "restricted_real_hw_grasp_preferences_applied",
            accepted_count=len(scored),
            rejected_count=len(rejected),
            ranked_candidate_ids=[
                _candidate_value(item["candidate"], "candidate_id", "unknown")
                for item in scored
            ],
        )
        return scored

    def _grasp_preference_config(self) -> dict[str, Any]:
        preview_config = self.grasping_config.get("plan_only_preview", {}) or {}
        screening_config = preview_config.get("candidate_screening", {}) or {}
        config = dict(screening_config.get("grasp_preferences", {}) or {})
        point_cloud_axes = (self.state.point_cloud or {}).get("planar_principal_axes")
        if point_cloud_axes and "face_reference_axes" not in config:
            config["face_reference_axes"] = point_cloud_axes
        current_axis = self._current_gripper_closing_axis(config)
        if current_axis is not None and "current_gripper_closing_axis" not in config:
            config["current_gripper_closing_axis"] = current_axis
        return config

    def _current_gripper_closing_axis(self, config: dict[str, Any]) -> list[float] | None:
        configured_axis = config.get("current_gripper_closing_axis")
        if configured_axis is not None:
            return _vector_from_value(configured_axis)
        frame_name = str(config.get("current_gripper_frame") or "").strip()
        if not frame_name or self.transformer is None:
            return None
        lookup = getattr(self.transformer, "_lookup_transform", None)
        if not callable(lookup):
            return None
        transform = lookup(frame_name, timeout_sec=0.2)
        if transform is None:
            return None
        rotation = transform.transform.rotation
        axis = _vector_from_value(config.get("current_gripper_axis", [0.0, 0.0, 1.0]))
        if axis is None:
            return None
        return list(
            _rotate_vector_by_quaternion(
                axis,
                (rotation.x, rotation.y, rotation.z, rotation.w),
            )
        )

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


def grasp_preference_score(candidate: Any, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Score tabletop grasp preferences without changing the hardware safety gate."""
    config = config or {}
    if not config.get("enabled", True):
        return {"allowed": True, "score": 0.0, "reason": "Grasp preferences disabled."}

    score = 0.0
    reasons = []
    if config.get("prefer_top_down", True):
        max_tilt_deg = float(config.get("max_tilt_deg", 30.0))
        top_down = _top_down_score(candidate, max_tilt_deg)
        if not top_down["allowed"]:
            return top_down
        score += float(config.get("top_down_weight", 1.0)) * top_down["score"]
        reasons.append(top_down["reason"])

    if config.get("prefer_face_aligned_yaw", True):
        max_deviation_deg = float(config.get("max_face_yaw_deviation_deg", 25.0))
        face = _face_alignment_score(
            candidate,
            max_deviation_deg,
            reference_axes=config.get("face_reference_axes"),
        )
        if not face["allowed"]:
            return face
        score += float(config.get("face_alignment_weight", 1.0)) * face["score"]
        reasons.append(face["reason"])

    if config.get("prefer_current_gripper_yaw", False):
        current = _current_gripper_yaw_score(
            candidate,
            config.get("current_gripper_closing_axis"),
            float(config.get("max_preferred_yaw_change_deg", 90.0)),
        )
        if not current["allowed"]:
            return current
        score += float(config.get("current_yaw_weight", 1.0)) * current["score"]
        reasons.append(current["reason"])

    return {
        "allowed": True,
        "score": score,
        "reason": "; ".join(reasons) if reasons else "Candidate satisfies grasp preferences.",
    }


def _current_gripper_yaw_score(
    candidate: Any,
    current_axis: Any | None,
    max_preferred_yaw_change_deg: float,
) -> dict[str, Any]:
    if current_axis is None:
        return {
            "allowed": True,
            "score": 0.0,
            "reason": "No current gripper closing axis available.",
        }
    candidate_axis = _candidate_vector(
        candidate,
        ("closing_axis_base", "closing_axis", "width_axis", "gripper_width_axis"),
    )
    if candidate_axis is None:
        candidate_axis = _yaw_axis_from_candidate_pose(candidate)
    candidate_xy = _normalize_vector([
        *(candidate_axis or [0.0, 0.0, 0.0])[:2],
        0.0,
    ])
    current = _vector_from_value(current_axis)
    current_xy = (
        _normalize_vector([current[0], current[1], 0.0])
        if current is not None
        else None
    )
    if candidate_xy is None or current_xy is None:
        return {
            "allowed": True,
            "score": 0.0,
            "reason": "Current yaw preference could not compare planar axes.",
        }
    dot = abs(candidate_xy[0] * current_xy[0] + candidate_xy[1] * current_xy[1])
    yaw_change_deg = math.degrees(math.acos(_clamp(dot, -1.0, 1.0)))
    max_yaw = max(float(max_preferred_yaw_change_deg), 1e-6)
    return {
        "allowed": True,
        "score": max(0.0, 1.0 - yaw_change_deg / max_yaw),
        "reason": f"current gripper yaw change {yaw_change_deg:.1f} deg.",
    }


def _top_down_score(candidate: Any, max_tilt_deg: float) -> dict[str, Any]:
    vector = _candidate_vector(
        candidate,
        ("approach_vector_base", "approach_vector", "grasp_approach_axis"),
    )
    if vector is None:
        return {"allowed": True, "score": 0.0, "reason": "No approach vector available."}
    normalized = _normalize_vector(vector)
    if normalized is None:
        return {"allowed": False, "score": 0.0, "reason": "Invalid approach vector."}
    vertical_alignment = abs(normalized[2])
    tilt_deg = math.degrees(math.acos(_clamp(vertical_alignment, -1.0, 1.0)))
    if tilt_deg > max_tilt_deg:
        return {
            "allowed": False,
            "score": 0.0,
            "reason": f"top_down tilt {tilt_deg:.1f} deg exceeds {max_tilt_deg:.1f} deg.",
        }
    return {
        "allowed": True,
        "score": 1.0 - tilt_deg / max(max_tilt_deg, 1e-6),
        "reason": f"top_down tilt {tilt_deg:.1f} deg.",
    }


def _face_alignment_score(
    candidate: Any,
    max_deviation_deg: float,
    reference_axes: Any | None = None,
) -> dict[str, Any]:
    axis = _candidate_vector(
        candidate,
        ("closing_axis_base", "closing_axis", "width_axis", "gripper_width_axis"),
    )
    if axis is None:
        axis = _yaw_axis_from_candidate_pose(candidate)
    if axis is None:
        return {"allowed": True, "score": 0.0, "reason": "No closing/yaw axis available."}
    normalized = _normalize_vector(axis)
    if normalized is None:
        return {"allowed": False, "score": 0.0, "reason": "Invalid closing/yaw axis."}
    xy_norm = math.hypot(normalized[0], normalized[1])
    if xy_norm < 1e-6:
        return {"allowed": True, "score": 0.0, "reason": "Closing/yaw axis is vertical."}
    deviation = _axis_deviation_from_reference_axes(normalized, reference_axes)
    if deviation > max_deviation_deg:
        return {
            "allowed": False,
            "score": 0.0,
            "reason": (
                f"face yaw deviation {deviation:.1f} deg exceeds "
                f"{max_deviation_deg:.1f} deg."
            ),
        }
    return {
        "allowed": True,
        "score": 1.0 - deviation / max(max_deviation_deg, 1e-6),
        "reason": f"face yaw deviation {deviation:.1f} deg.",
    }


def _candidate_vector(candidate: Any, names: tuple[str, ...]) -> list[float] | None:
    for name in names:
        value = _candidate_value(candidate, name)
        if value is None:
            continue
        if isinstance(value, dict):
            try:
                return [float(value[axis]) for axis in ("x", "y", "z")]
            except (KeyError, TypeError, ValueError):
                return None
        try:
            values = list(value)
        except TypeError:
            return None
        if len(values) >= 3:
            try:
                return [float(values[0]), float(values[1]), float(values[2])]
            except (TypeError, ValueError):
                return None
    return None


def _yaw_axis_from_candidate_pose(candidate: Any) -> list[float] | None:
    pose = _candidate_value(candidate, "pose")
    if not isinstance(pose, dict):
        return None
    orientation = pose.get("orientation")
    if not isinstance(orientation, dict):
        return None
    try:
        x = float(orientation.get("x", 0.0))
        y = float(orientation.get("y", 0.0))
        z = float(orientation.get("z", 0.0))
        w = float(orientation.get("w", 1.0))
    except (TypeError, ValueError):
        return None
    # Rotate local X by the quaternion; this is enough to penalize diagonal yaw.
    return [
        1.0 - 2.0 * (y * y + z * z),
        2.0 * (x * y + z * w),
        2.0 * (x * z - y * w),
    ]


def _axis_deviation_from_orthogonal_faces(angle_deg: float) -> float:
    wrapped = abs(((angle_deg + 45.0) % 90.0) - 45.0)
    return wrapped


def _axis_deviation_from_reference_axes(
    axis: list[float],
    reference_axes: Any | None = None,
) -> float:
    if not reference_axes:
        angle_deg = math.degrees(math.atan2(axis[1], axis[0]))
        return _axis_deviation_from_orthogonal_faces(angle_deg)
    deviations = []
    for reference in reference_axes:
        ref = _vector_from_value(reference)
        if ref is None:
            continue
        ref_norm = _normalize_vector(ref)
        if ref_norm is None:
            continue
        axis_xy = _normalize_vector([axis[0], axis[1], 0.0])
        ref_xy = _normalize_vector([ref_norm[0], ref_norm[1], 0.0])
        if axis_xy is None or ref_xy is None:
            continue
        dot = abs(axis_xy[0] * ref_xy[0] + axis_xy[1] * ref_xy[1])
        deviations.append(math.degrees(math.acos(_clamp(dot, -1.0, 1.0))))
    if not deviations:
        angle_deg = math.degrees(math.atan2(axis[1], axis[0]))
        return _axis_deviation_from_orthogonal_faces(angle_deg)
    return min(deviations)


def _vector_from_value(value: Any) -> list[float] | None:
    if isinstance(value, dict):
        try:
            return [float(value[axis]) for axis in ("x", "y", "z")]
        except (KeyError, TypeError, ValueError):
            return None
    try:
        values = list(value)
    except TypeError:
        return None
    if len(values) < 3:
        return None
    try:
        return [float(values[0]), float(values[1]), float(values[2])]
    except (TypeError, ValueError):
        return None


def _normalize_vector(vector: list[float]) -> list[float] | None:
    norm = math.sqrt(sum(float(item) * float(item) for item in vector[:3]))
    if norm <= 1e-9:
        return None
    return [float(item) / norm for item in vector[:3]]


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _rotate_vector_by_quaternion(
    vector: list[float],
    quaternion: tuple[float, float, float, float],
) -> tuple[float, float, float]:
    qx, qy, qz, qw = quaternion
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm <= 1e-9:
        return float(vector[0]), float(vector[1]), float(vector[2])
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
    vx, vy, vz = float(vector[0]), float(vector[1]), float(vector[2])
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return (
        vx + qw * tx + (qy * tz - qz * ty),
        vy + qw * ty + (qz * tx - qx * tz),
        vz + qw * tz + (qx * ty - qy * tx),
    )


def print_full_safety_summary(hardware_mode, real_arm_safety, gripper_config: dict) -> None:
    print("REAL HARDWARE GRASP MENU")
    print("Each real arm or gripper action is individually reviewed and confirmed.")
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
        "12. Approach grasp pose (review + EXECUTE)\n"
        "13. Close real gripper on object (YES)\n"
        "14. Retreat with grasped object (review + EXECUTE)\n"
        "15. Return to safe_home\n"
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
            print(controller.approach_grasp().message)
        elif choice == "13":
            print(controller.close_real_gripper_for_grasp(confirmation=None).message)
        elif choice == "14":
            print(controller.retreat_grasp().message)
        elif choice == "15":
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
        tabletop_safety = TabletopSafety.from_config(_load_yaml("tabletop_safety.yaml"))
        assert_arm_execution_allowed(hardware_mode)
        if (
            hardware_mode.arm.execution_mode != "full_grasp_supervised"
            or real_arm_safety.execution_mode != "full_grasp_supervised"
        ):
            raise PermissionError(
                "This menu requires full_grasp_supervised in hardware_mode.yaml "
                "and real_arm_safety.yaml."
            )
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
            real_arm_safety=real_arm_safety,
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
            tabletop_safety=tabletop_safety,
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
