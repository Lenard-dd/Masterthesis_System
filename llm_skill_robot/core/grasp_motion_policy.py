"""Execution-neutral MoveIt policy for a reviewed grasp sequence.

This policy sits below Agent/Composite-Skill task selection and above a robot
backend.  It translates the reviewed ``plan_only_preview.moveit`` configuration
into per-stage request parameters without embedding them in a Tool schema.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GraspMotionPolicy:
    """Per-stage planner and scaling settings for one safe grasp workflow."""

    planning_kwargs_by_skill: dict[str, dict[str, Any]]
    velocity_scale: float
    acceleration_scale: float

    @classmethod
    def from_grasping_config(
        cls,
        grasping_config: dict[str, Any],
        *,
        max_velocity_scale: float,
        max_acceleration_scale: float,
    ) -> "GraspMotionPolicy":
        preview = grasping_config.get("plan_only_preview", {}) or {}
        return cls(
            planning_kwargs_by_skill={
                "move_to_pregrasp": cls._stage_kwargs(preview, "pregrasp"),
                "approach_grasp": cls._stage_kwargs(preview, "grasp"),
                "retreat_grasp": cls._stage_kwargs(preview, "retreat"),
                "lift_object": cls._stage_kwargs(preview, "retreat"),
            },
            velocity_scale=min(
                float(preview.get("velocity_scale", 0.05)), max_velocity_scale
            ),
            acceleration_scale=min(
                float(preview.get("acceleration_scale", 0.05)),
                max_acceleration_scale,
            ),
        )

    def kwargs_for(self, skill_id: str) -> dict[str, Any]:
        """Return a defensive copy suitable for one MoveIt request."""
        return dict(self.planning_kwargs_by_skill.get(skill_id, {}))

    def with_pregrasp_variant(self, variant_kwargs: dict[str, Any]) -> "GraspMotionPolicy":
        """Add reviewed pose-tolerance constraints to pregrasp only."""
        planning = {
            skill_id: dict(options)
            for skill_id, options in self.planning_kwargs_by_skill.items()
        }
        planning["move_to_pregrasp"].update(variant_kwargs)
        return GraspMotionPolicy(
            planning_kwargs_by_skill=planning,
            velocity_scale=self.velocity_scale,
            acceleration_scale=self.acceleration_scale,
        )

    @staticmethod
    def _stage_kwargs(preview: dict[str, Any], stage: str) -> dict[str, Any]:
        moveit = preview.get("moveit", {}) or {}
        result: dict[str, Any] = {}
        if moveit.get("pipeline_id"):
            result["pipeline_id"] = str(moveit["pipeline_id"])
        if moveit.get("planner_id"):
            result["planner_id"] = str(moveit["planner_id"])
        if "num_planning_attempts" in moveit:
            result["num_planning_attempts"] = int(moveit["num_planning_attempts"])
        if "allowed_planning_time_sec" in moveit:
            result["allowed_planning_time"] = float(moveit["allowed_planning_time_sec"])
        if "service_timeout_sec" in moveit:
            result["service_timeout_sec"] = float(moveit["service_timeout_sec"])

        stage_planners = moveit.get("stage_planner_ids", {}) or {}
        if stage in stage_planners:
            result["planner_id"] = str(stage_planners[stage])
        stage_pipelines = moveit.get("stage_pipeline_ids", {}) or {}
        if stage in stage_pipelines:
            result["pipeline_id"] = str(stage_pipelines[stage])

        override = (moveit.get("stage_overrides", {}) or {}).get(stage, {}) or {}
        if "planner_id" in override:
            result["planner_id"] = str(override["planner_id"])
        if "pipeline_id" in override:
            result["pipeline_id"] = str(override["pipeline_id"])
        if "num_planning_attempts" in override:
            result["num_planning_attempts"] = int(override["num_planning_attempts"])
        if "allowed_planning_time_sec" in override:
            result["allowed_planning_time"] = float(override["allowed_planning_time_sec"])
        if "service_timeout_sec" in override:
            result["service_timeout_sec"] = float(override["service_timeout_sec"])
        return result
