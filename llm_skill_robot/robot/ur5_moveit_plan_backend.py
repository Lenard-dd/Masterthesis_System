"""Dry-run UR5 MoveIt planning backend skeleton."""

from datetime import datetime, timezone
from typing import Any, Dict, Optional
from uuid import uuid4

from llm_skill_robot.core.schemas import MotionPlanCacheEntry, MotionPlanSummary


class UR5MoveItPlanBackend:
    """Plan-only UR5 backend skeleton.

    This class intentionally does not import MoveIt APIs, send trajectories,
    call controllers, or execute robot motion.
    """

    def __init__(
        self,
        node=None,
        robot_name: str = "ur5",
        planning_group: str = "ur_manipulator",
        base_frame: str = "base_link",
        ee_frame: str = "tool0",
        dry_run: bool = True,
    ):
        self.node = node
        self.robot_name = robot_name
        self.planning_group = planning_group
        self.base_frame = base_frame
        self.ee_frame = ee_frame
        self.dry_run = dry_run
        self.plan_cache: Dict[str, MotionPlanCacheEntry] = {}

    def plan_to_named_target(
        self,
        target: str,
        velocity_scale: float = 0.1,
        acceleration_scale: float = 0.1,
        **kwargs,
    ) -> Dict[str, Any]:
        """Create a dry-run plan summary for a named target."""
        summary = MotionPlanSummary(
            plan_id=self._new_plan_id(),
            target_type="named_target",
            target_name=target,
            planning_group=self.planning_group,
            success=True,
            message="Dry-run plan generated.",
            max_velocity_abs=velocity_scale,
            trajectory_preview={
                "dry_run": self.dry_run,
                "robot": self.robot_name,
                "base_frame": self.base_frame,
                "ee_frame": self.ee_frame,
                "velocity_scale": velocity_scale,
                "acceleration_scale": acceleration_scale,
            },
        )
        entry = self._cache_plan(
            summary=summary,
            skill_id=kwargs.get("skill_id", "move_to_named_target"),
            step_id=kwargs.get("step_id"),
        )
        return self._result_from_entry(entry)

    def plan_to_pose(
        self,
        frame: str,
        position: dict,
        orientation: dict,
        velocity_scale: float = 0.1,
        acceleration_scale: float = 0.1,
        **kwargs,
    ) -> Dict[str, Any]:
        """Create a dry-run plan summary for a target pose."""
        summary = MotionPlanSummary(
            plan_id=self._new_plan_id(),
            target_type="pose",
            target_pose={
                "frame": frame,
                "position": dict(position),
                "orientation": dict(orientation),
            },
            planning_group=self.planning_group,
            success=True,
            message="Dry-run plan generated.",
            max_velocity_abs=velocity_scale,
            trajectory_preview={
                "dry_run": self.dry_run,
                "robot": self.robot_name,
                "base_frame": self.base_frame,
                "ee_frame": self.ee_frame,
                "velocity_scale": velocity_scale,
                "acceleration_scale": acceleration_scale,
            },
        )
        entry = self._cache_plan(
            summary=summary,
            skill_id=kwargs.get("skill_id", "move_to_pose"),
            step_id=kwargs.get("step_id"),
        )
        return self._result_from_entry(entry)

    def get_plan(self, plan_id: str) -> Optional[MotionPlanCacheEntry]:
        """Return a cached plan entry by plan id."""
        return self.plan_cache.get(plan_id)

    def list_cached_plans(self):
        """Return all cached plan entries."""
        return list(self.plan_cache.values())

    def clear_plan_cache(self) -> None:
        """Clear all cached dry-run plans."""
        self.plan_cache.clear()

    def _cache_plan(
        self,
        summary: MotionPlanSummary,
        skill_id: str,
        step_id: Optional[int] = None,
    ) -> MotionPlanCacheEntry:
        entry = MotionPlanCacheEntry(
            plan_id=summary.plan_id,
            skill_id=skill_id,
            step_id=step_id,
            summary=summary,
            created_at=datetime.now(timezone.utc).isoformat(),
            raw_plan_available=False,
        )
        self.plan_cache[entry.plan_id] = entry
        return entry

    @staticmethod
    def _new_plan_id() -> str:
        return f"plan_{uuid4().hex}"

    @staticmethod
    def _result_from_entry(entry: MotionPlanCacheEntry) -> Dict[str, Any]:
        return {
            "success": entry.summary.success,
            "message": entry.summary.message,
            "plan_id": entry.plan_id,
            "summary": entry.summary.model_dump(),
        }
