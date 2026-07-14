"""Single-step, supervised real-UR5 grasp motion execution.

This backend intentionally has no ``execute_grasp_sequence`` method.  Callers
must invoke and confirm one motion at a time, and may use
``FullGraspStateMachine`` to enforce the larger grasp workflow.
"""

from __future__ import annotations

from typing import Any, Callable

from llm_skill_robot.core.trajectory_reviewer import TrajectoryReviewer
from llm_skill_robot.core.trajectory_validator import TrajectoryValidator
from llm_skill_robot.safety.real_arm_safety import (
    RealArmSafetyConfig,
    require_execute_confirmation,
    validate_full_grasp_step,
    validate_motion_summary,
    validate_named_target,
)
from llm_skill_robot.safety.tabletop_safety import TabletopSafety


class GraspExecutionBackend:
    """Plan, review, confirm, and execute exactly one real grasp motion."""

    def __init__(
        self,
        moveit_backend: Any,
        real_arm_safety: RealArmSafetyConfig,
        *,
        trajectory_validator: TrajectoryValidator | None = None,
        trajectory_reviewer: TrajectoryReviewer | None = None,
        review_approval_provider: Callable[[dict[str, Any]], bool] | None = None,
        confirmation_provider: Callable[[str], str] | None = None,
        audit_logger: Any = None,
        planning_kwargs_by_skill: dict[str, dict[str, Any]] | None = None,
        tabletop_safety: TabletopSafety | None = None,
    ) -> None:
        self.moveit_backend = moveit_backend
        self.real_arm_safety = real_arm_safety
        self.trajectory_validator = trajectory_validator or TrajectoryValidator(
            {
                "trajectory_limits": {
                    # ``max_velocity_scale`` is a MoveIt request scaling factor,
                    # not an absolute joint-speed limit in rad/s.  The plan
                    # summary's ``max_velocity_abs`` is measured in rad/s, so
                    # comparing the two would reject a correctly scaled plan.
                    # RealArmSafety still validates the requested velocity and
                    # acceleration scales below; this validator continues to
                    # require a successful, non-empty trajectory.
                    "max_joint_delta_rad": None,
                    "max_velocity_abs": None,
                }
            }
        )
        self.trajectory_reviewer = trajectory_reviewer or TrajectoryReviewer()
        self.review_approval_provider = review_approval_provider
        self.confirmation_provider = confirmation_provider or input
        self.audit_logger = audit_logger
        self.planning_kwargs_by_skill = {
            str(skill_id): dict(options)
            for skill_id, options in (planning_kwargs_by_skill or {}).items()
            if isinstance(options, dict)
        }
        self.tabletop_safety = tabletop_safety

    def set_planning_kwargs(self, skill_id: str, options: dict[str, Any]) -> None:
        """Set MoveIt planning options for one safe, single-motion skill."""
        if not isinstance(options, dict):
            raise ValueError("MoveIt planning options must be a mapping.")
        self.planning_kwargs_by_skill[str(skill_id)] = dict(options)

    def move_to_pregrasp(self, object_id: str, grasp_plans: Any, **kwargs: Any) -> dict:
        """Execute the reviewed pregrasp pose for one selected object."""
        return self._execute_pose_step(
            "move_to_pregrasp", object_id, grasp_plans, "pregrasp_pose", **kwargs
        )

    def move_to_observe(self, **kwargs: Any) -> dict:
        """Move to the configured observation target as one supervised motion."""
        return self._execute_named_target_step("move_to_observe", "observe", **kwargs)

    def approach_grasp(self, object_id: str, grasp_plans: Any, **kwargs: Any) -> dict:
        """Execute one reviewed, limited approach from pregrasp to grasp pose."""
        return self._execute_pose_step(
            "approach_grasp", object_id, grasp_plans, "grasp_pose", **kwargs
        )

    def retreat_grasp(self, object_id: str, grasp_plans: Any, **kwargs: Any) -> dict:
        """Execute one reviewed retreat to the grasp plan's retreat pose."""
        return self._execute_pose_step(
            "retreat_grasp", object_id, grasp_plans, "retreat_pose", **kwargs
        )

    def lift_object(
        self,
        object_id: str,
        grasp_plans: Any,
        lift_distance_m: float | None = None,
        **kwargs: Any,
    ) -> dict:
        """Plan a single upward lift from the current pose, capped by safety limits."""
        plan = self._resolve_grasp_plan(object_id, grasp_plans)
        if plan is None:
            return self._blocked("lift_object requires a valid GraspPlan.")

        requested_distance = (
            self.real_arm_safety.limits.max_lift_distance_m
            if lift_distance_m is None
            else lift_distance_m
        )
        try:
            requested_distance = float(requested_distance)
        except (TypeError, ValueError):
            return self._blocked("lift_distance_m must be numeric.")
        if requested_distance < 0:
            return self._blocked("lift_distance_m must be non-negative.")
        distance = min(requested_distance, self.real_arm_safety.limits.max_lift_distance_m)

        current_pose_getter = getattr(self.moveit_backend, "get_current_pose", None)
        if not callable(current_pose_getter):
            return self._blocked("lift_object requires MoveIt current-pose support.")
        current_pose = current_pose_getter()
        pose = self._normalise_pose(current_pose)
        if pose is None:
            return self._blocked("lift_object could not obtain a valid current pose.")
        if pose["frame"] != "base_link":
            return self._blocked("lift_object requires current pose frame == base_link.")
        pose["position"]["z"] += distance
        return self._plan_review_confirm_execute(
            "lift_object",
            object_id,
            pose,
            distance_m=distance,
            grasp_score=self._value(plan, "grasp_score"),
            **kwargs,
        )

    def _execute_pose_step(
        self,
        skill_id: str,
        object_id: str,
        grasp_plans: Any,
        pose_name: str,
        **kwargs: Any,
    ) -> dict:
        plan = self._resolve_grasp_plan(object_id, grasp_plans)
        if plan is None:
            return self._blocked(f"{skill_id} requires a valid GraspPlan.")
        pose = self._normalise_pose(self._value(plan, pose_name))
        if pose is None:
            return self._blocked(f"{skill_id} requires GraspPlan.{pose_name}.")
        if pose["frame"] != "base_link":
            return self._blocked(f"{skill_id} requires target pose frame == base_link.")
        tabletop_check = self._validate_tabletop_plan(plan, skill_id)
        if not tabletop_check["allowed"]:
            self._audit("grasp_motion_validation_failed", skill_id=skill_id, object_id=object_id, reasons=[tabletop_check["reason"]])
            return self._blocked(tabletop_check["reason"])
        distance_m = kwargs.pop("distance_m", None)
        if skill_id in {"approach_grasp", "retreat_grasp"} and distance_m is None:
            reference_name = (
                "pregrasp_pose" if skill_id == "approach_grasp" else "grasp_pose"
            )
            reference = self._normalise_pose(self._value(plan, reference_name))
            if reference is None or reference["frame"] != "base_link":
                return self._blocked(
                    f"{skill_id} requires GraspPlan.{reference_name} in base_link "
                    "to calculate its distance."
                )
            distance_m = self._pose_distance(reference, pose)
        return self._plan_review_confirm_execute(
            skill_id,
            object_id,
            pose,
            distance_m=distance_m,
            grasp_score=self._value(plan, "grasp_score"),
            **kwargs,
        )

    def _validate_tabletop_plan(self, plan: Any, skill_id: str) -> dict[str, Any]:
        if self.tabletop_safety is None:
            return {"allowed": True}
        return self.tabletop_safety.validate_grasp_plan(plan, motion_skill=skill_id)

    def _execute_named_target_step(self, skill_id: str, target_name: str, **kwargs: Any) -> dict:
        target_check = validate_named_target(target_name, self.real_arm_safety)
        if not target_check["allowed"]:
            return self._blocked(target_check["reason"])
        velocity = self._safe_scale(kwargs.get("velocity_scale"), self.real_arm_safety.limits.max_velocity_scale)
        acceleration = self._safe_scale(kwargs.get("acceleration_scale"), self.real_arm_safety.limits.max_acceleration_scale)
        self._audit("grasp_motion_plan_requested", skill_id=skill_id, target_name=target_name)
        try:
            plan_result = self.moveit_backend.plan_to_named_target(
                target_name,
                velocity_scale=velocity,
                acceleration_scale=acceleration,
                skill_id=skill_id,
                **self.planning_kwargs_by_skill.get(skill_id, {}),
            )
        except Exception as exc:
            self._audit("grasp_motion_failed", skill_id=skill_id, target_name=target_name, reason=str(exc))
            return self._blocked(f"MoveIt planning failed: {exc}")
        if not isinstance(plan_result, dict) or not plan_result.get("success", False):
            message = plan_result.get("message", "MoveIt planning failed.") if isinstance(plan_result, dict) else "MoveIt planning failed."
            return self._blocked(str(message))

        summary = self._motion_summary(
            plan_result.get("summary", {}), skill_id,
            {"named_target": target_name}, velocity, acceleration, None, None,
        )
        summary["target_name"] = target_name
        validation = self.trajectory_validator.validate_motion_plan_summary(summary)
        safety = validate_motion_summary(summary, self.real_arm_safety)
        full_step = validate_full_grasp_step(skill_id, summary, self.real_arm_safety)
        if validation.get("decision") != "ALLOW" or not safety.get("allowed", False) or not full_step.get("allowed", False):
            reasons = list(validation.get("reasons", []))
            if not safety.get("allowed", False):
                reasons.append(safety["reason"])
            if not full_step.get("allowed", False):
                reasons.append(full_step["reason"])
            self._audit("grasp_motion_validation_failed", skill_id=skill_id, target_name=target_name, reasons=reasons, summary=summary)
            return self._blocked("; ".join(dict.fromkeys(reasons)) or "Trajectory safety validation failed.", summary=summary)
        if not self._review_trajectory(summary, kwargs.get("trajectory_review_approved")):
            return self._blocked("Trajectory review was not approved.", summary=summary)
        confirmation = kwargs.get("confirmation")
        if confirmation is None:
            confirmation = self.confirmation_provider("Type EXECUTE to execute this single real grasp motion:\n> ")
        if not require_execute_confirmation(confirmation, self.real_arm_safety.confirmation_phrase):
            return self._blocked("Exact EXECUTE confirmation is required; no motion was executed.", summary=summary)
        result = self.moveit_backend.execute_cached_plan_real(str(plan_result["plan_id"]), final_confirmed=True)
        if not isinstance(result, dict) or not result.get("success", False):
            message = result.get("message", "MoveIt real execution failed.") if isinstance(result, dict) else "MoveIt real execution failed."
            self._audit("grasp_motion_failed", skill_id=skill_id, target_name=target_name, reason=message, summary=summary)
            return self._blocked(str(message), summary=summary)
        self._audit("grasp_motion_executed", skill_id=skill_id, target_name=target_name, plan_id=plan_result["plan_id"], summary=summary)
        return {"success": True, "message": str(result.get("message", "Motion executed.")), "summary": summary, "output": result.get("output", {})}

    def _plan_review_confirm_execute(
        self,
        skill_id: str,
        object_id: str,
        pose: dict[str, Any],
        *,
        distance_m: float | None,
        grasp_score: float | None = None,
        confirmation: str | None = None,
        trajectory_review_approved: bool | None = None,
        velocity_scale: float | None = None,
        acceleration_scale: float | None = None,
        **_unused: Any,
    ) -> dict:
        velocity = self._safe_scale(velocity_scale, self.real_arm_safety.limits.max_velocity_scale)
        acceleration = self._safe_scale(
            acceleration_scale, self.real_arm_safety.limits.max_acceleration_scale
        )
        self._audit("grasp_motion_plan_requested", skill_id=skill_id, object_id=object_id, target_pose=pose)
        try:
            plan_result = self.moveit_backend.plan_to_pose(
                frame=pose["frame"],
                position=pose["position"],
                orientation=pose["orientation"],
                velocity_scale=velocity,
                acceleration_scale=acceleration,
                skill_id=skill_id,
                **self.planning_kwargs_by_skill.get(skill_id, {}),
            )
        except Exception as exc:
            self._audit("grasp_motion_failed", skill_id=skill_id, object_id=object_id, reason=str(exc))
            return self._blocked(f"MoveIt planning failed: {exc}")
        if not isinstance(plan_result, dict) or not plan_result.get("success", False):
            message = plan_result.get("message", "MoveIt planning failed.") if isinstance(plan_result, dict) else "MoveIt planning failed."
            self._audit("grasp_motion_failed", skill_id=skill_id, object_id=object_id, reason=message)
            return self._blocked(str(message))

        summary = self._motion_summary(
            plan_result.get("summary", {}),
            skill_id,
            pose,
            velocity,
            acceleration,
            distance_m,
            grasp_score,
        )
        self._audit("grasp_motion_plan_success", skill_id=skill_id, object_id=object_id, plan_id=plan_result.get("plan_id"), summary=summary)

        validation = self.trajectory_validator.validate_motion_plan_summary(summary)
        safety = validate_motion_summary(summary, self.real_arm_safety)
        full_step = validate_full_grasp_step(skill_id, summary, self.real_arm_safety)
        if (
            validation.get("decision") != "ALLOW"
            or not safety.get("allowed", False)
            or not full_step.get("allowed", False)
        ):
            reasons = list(validation.get("reasons", []))
            if not safety.get("allowed", False):
                reasons.append(safety["reason"])
            if not full_step.get("allowed", False):
                reasons.append(full_step["reason"])
            self._audit("grasp_motion_validation_failed", skill_id=skill_id, object_id=object_id, reasons=reasons, summary=summary)
            return self._blocked("; ".join(reasons) or "Trajectory safety validation failed.", summary=summary)

        if not self._review_trajectory(summary, trajectory_review_approved):
            self._audit("grasp_motion_validation_failed", skill_id=skill_id, object_id=object_id, reasons=["Trajectory review was not approved."], summary=summary)
            return self._blocked("Trajectory review was not approved.", summary=summary)

        self._audit("grasp_motion_confirmation_requested", skill_id=skill_id, object_id=object_id, plan_id=plan_result.get("plan_id"))
        if confirmation is None:
            confirmation = self.confirmation_provider("Type EXECUTE to execute this single real grasp motion:\n> ")
        if not require_execute_confirmation(confirmation, self.real_arm_safety.confirmation_phrase):
            return self._blocked("Exact EXECUTE confirmation is required; no motion was executed.", summary=summary)

        result = self.moveit_backend.execute_cached_plan_real(
            str(plan_result["plan_id"]), final_confirmed=True
        )
        if not isinstance(result, dict) or not result.get("success", False):
            message = result.get("message", "MoveIt real execution failed.") if isinstance(result, dict) else "MoveIt real execution failed."
            self._audit("grasp_motion_failed", skill_id=skill_id, object_id=object_id, reason=message, summary=summary)
            return self._blocked(str(message), summary=summary)
        self._audit("grasp_motion_executed", skill_id=skill_id, object_id=object_id, plan_id=plan_result["plan_id"], summary=summary)
        return {"success": True, "message": str(result.get("message", "Motion executed.")), "summary": summary, "output": result.get("output", {})}

    def _review_trajectory(self, summary: dict[str, Any], approved: bool | None) -> bool:
        if approved is not None:
            print(self.trajectory_reviewer.review(summary))
            return approved is True
        if self.review_approval_provider is not None:
            print(self.trajectory_reviewer.review(summary))
            return self.review_approval_provider(summary) is True
        return self.trajectory_reviewer.ask_for_review_approval(summary)

    def _motion_summary(
        self,
        raw: Any,
        skill_id: str,
        pose: dict[str, Any],
        velocity: float,
        acceleration: float,
        distance_m: float | None,
        grasp_score: float | None,
    ) -> dict[str, Any]:
        source = raw.model_dump() if hasattr(raw, "model_dump") else dict(raw or {})
        preview = source.get("trajectory_preview") or {}
        summary = {
            **source,
            "skill_id": skill_id,
            "target_pose": pose,
            "velocity_scale": velocity,
            "acceleration_scale": acceleration,
            "max_cartesian_step_m": source.get("max_cartesian_step_m", preview.get("max_cartesian_step_m")),
            "max_joint_delta_rad": source.get("max_joint_delta_rad", preview.get("max_joint_delta_rad")),
            "total_joint_delta_rad": source.get("total_joint_delta_rad", preview.get("total_joint_delta_rad")),
        }
        if distance_m is not None:
            summary["distance_m"] = distance_m
        if grasp_score is not None:
            summary["grasp_score"] = grasp_score
        return summary

    @staticmethod
    def _value(value: Any, name: str, default: Any = None) -> Any:
        return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)

    def _resolve_grasp_plan(self, object_id: str, grasp_plans: Any) -> Any | None:
        if grasp_plans is None:
            return None
        if self._value(grasp_plans, "pregrasp_pose") is not None:
            return grasp_plans
        if isinstance(grasp_plans, dict):
            return grasp_plans.get(object_id)
        return None

    @staticmethod
    def _normalise_pose(pose: Any) -> dict[str, Any] | None:
        if pose is None:
            return None
        if not isinstance(pose, dict):
            pose = {
                "frame": getattr(pose, "frame", None) or getattr(pose, "frame_id", None),
                "position": getattr(pose, "position", None),
                "orientation": getattr(pose, "orientation", None),
            }
        frame = pose.get("frame") or pose.get("frame_id")
        position = pose.get("position")
        orientation = pose.get("orientation")
        if not isinstance(frame, str) or not isinstance(position, dict) or not isinstance(orientation, dict):
            return None
        try:
            return {
                "frame": frame,
                "position": {axis: float(position[axis]) for axis in ("x", "y", "z")},
                "orientation": {axis: float(orientation[axis]) for axis in ("x", "y", "z", "w")},
            }
        except (KeyError, TypeError, ValueError):
            return None

    @staticmethod
    def _safe_scale(requested: float | None, maximum: float) -> float:
        if requested is None:
            return maximum

    @staticmethod
    def _pose_distance(first: dict[str, Any], second: dict[str, Any]) -> float:
        return sum(
            (first["position"][axis] - second["position"][axis]) ** 2
            for axis in ("x", "y", "z")
        ) ** 0.5
        try:
            return min(float(requested), maximum)
        except (TypeError, ValueError):
            return maximum

    def _audit(self, event: str, **fields: Any) -> None:
        if self.audit_logger is not None and hasattr(self.audit_logger, "log_hardware_safety_event"):
            self.audit_logger.log_hardware_safety_event(event, **fields)

    @staticmethod
    def _blocked(message: str, *, summary: dict[str, Any] | None = None) -> dict[str, Any]:
        result = {"success": False, "message": message, "output": {}}
        if summary is not None:
            result["summary"] = summary
        return result
