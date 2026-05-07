"""UR5 MoveIt plan-only backend."""

from datetime import datetime, timezone
from typing import Any, Dict, Optional
from uuid import uuid4

from llm_skill_robot.core.schemas import MotionPlanCacheEntry, MotionPlanSummary

MOVEIT_UNAVAILABLE_MESSAGE = (
    "MoveIt Python planning interface is not available. Use dry_run=True or "
    "install required MoveIt Python bindings."
)


class UR5MoveItPlanBackend:
    """Plan-only UR5 backend.

    This class never sends trajectories, calls controller actions, or executes
    robot motion. When dry_run=False it only asks MoveIt to generate a plan.
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
        self._owns_node = False
        self._owns_rclpy_context = False
        self._node_error: Optional[str] = None
        self._moveit_py = None
        self._moveit_error: Optional[str] = None

        if not self.dry_run:
            self._ensure_node()

    def plan_to_named_target(
        self,
        target: str,
        velocity_scale: float = 0.1,
        acceleration_scale: float = 0.1,
        **kwargs,
    ) -> Dict[str, Any]:
        """Create a plan summary for a named target without executing it."""
        if not self.dry_run:
            return self._plan_to_named_target_moveit(
                target=target,
                velocity_scale=velocity_scale,
                acceleration_scale=acceleration_scale,
                **kwargs,
            )

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
        """Create a plan summary for a target pose without executing it."""
        if not self.dry_run:
            return self._plan_to_pose_moveit(
                frame=frame,
                position=position,
                orientation=orientation,
                velocity_scale=velocity_scale,
                acceleration_scale=acceleration_scale,
                **kwargs,
            )

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

    def shutdown(self) -> None:
        """Destroy only resources created by this backend."""
        if self._owns_node and self.node is not None:
            try:
                self.node.destroy_node()
            except Exception:
                pass
            self.node = None
        if self._owns_rclpy_context:
            try:
                import rclpy

                if rclpy.ok():
                    rclpy.shutdown()
            except Exception:
                pass

    def _plan_to_named_target_moveit(
        self,
        target: str,
        velocity_scale: float,
        acceleration_scale: float,
        **kwargs,
    ) -> Dict[str, Any]:
        if not self._moveit_interface_available():
            return self._failure_result(
                target_type="named_target",
                target_name=target,
                message=self._moveit_error or MOVEIT_UNAVAILABLE_MESSAGE,
                skill_id=kwargs.get("skill_id", "move_to_named_target"),
                step_id=kwargs.get("step_id"),
            )

        plan_id = self._new_plan_id()
        try:
            planning_component = self._moveit_py.get_planning_component(self.planning_group)
            planning_component.set_start_state_to_current_state()
            planning_component.set_goal_state(configuration_name=target)
            self._apply_scaling_if_supported(
                planning_component,
                velocity_scale=velocity_scale,
                acceleration_scale=acceleration_scale,
            )
            plan_result = planning_component.plan()
        except Exception as e:
            return self._failure_result(
                target_type="named_target",
                target_name=target,
                message=f"MoveIt plan-only request failed: {e}",
                skill_id=kwargs.get("skill_id", "move_to_named_target"),
                step_id=kwargs.get("step_id"),
                plan_id=plan_id,
            )

        summary = self._summary_from_plan_result(
            plan_id=plan_id,
            target_type="named_target",
            target_name=target,
            target_pose=None,
            plan_result=plan_result,
        )
        entry = self._cache_plan(
            summary=summary,
            skill_id=kwargs.get("skill_id", "move_to_named_target"),
            step_id=kwargs.get("step_id"),
            raw_plan_available=summary.success,
        )
        return self._result_from_entry(entry)

    def _plan_to_pose_moveit(
        self,
        frame: str,
        position: dict,
        orientation: dict,
        velocity_scale: float,
        acceleration_scale: float,
        **kwargs,
    ) -> Dict[str, Any]:
        target_pose = {
            "frame": frame,
            "position": dict(position),
            "orientation": dict(orientation),
        }
        if not self._moveit_interface_available():
            return self._failure_result(
                target_type="pose",
                target_pose=target_pose,
                message=self._moveit_error or MOVEIT_UNAVAILABLE_MESSAGE,
                skill_id=kwargs.get("skill_id", "move_to_pose"),
                step_id=kwargs.get("step_id"),
            )

        plan_id = self._new_plan_id()
        try:
            from geometry_msgs.msg import PoseStamped

            pose_msg = PoseStamped()
            pose_msg.header.frame_id = frame
            pose_msg.pose.position.x = float(position["x"])
            pose_msg.pose.position.y = float(position["y"])
            pose_msg.pose.position.z = float(position["z"])
            pose_msg.pose.orientation.x = float(orientation["x"])
            pose_msg.pose.orientation.y = float(orientation["y"])
            pose_msg.pose.orientation.z = float(orientation["z"])
            pose_msg.pose.orientation.w = float(orientation["w"])

            planning_component = self._moveit_py.get_planning_component(self.planning_group)
            planning_component.set_start_state_to_current_state()
            planning_component.set_goal_state(
                pose_stamped_msg=pose_msg,
                pose_link=self.ee_frame,
            )
            self._apply_scaling_if_supported(
                planning_component,
                velocity_scale=velocity_scale,
                acceleration_scale=acceleration_scale,
            )
            plan_result = planning_component.plan()
        except Exception as e:
            return self._failure_result(
                target_type="pose",
                target_pose=target_pose,
                message=f"MoveIt pose plan-only request failed: {e}",
                skill_id=kwargs.get("skill_id", "move_to_pose"),
                step_id=kwargs.get("step_id"),
                plan_id=plan_id,
            )

        summary = self._summary_from_plan_result(
            plan_id=plan_id,
            target_type="pose",
            target_name=None,
            target_pose=target_pose,
            plan_result=plan_result,
        )
        entry = self._cache_plan(
            summary=summary,
            skill_id=kwargs.get("skill_id", "move_to_pose"),
            step_id=kwargs.get("step_id"),
            raw_plan_available=summary.success,
        )
        return self._result_from_entry(entry)

    def _ensure_node(self) -> bool:
        if self.node is not None:
            return True

        try:
            import rclpy

            if not rclpy.ok():
                rclpy.init()
                self._owns_rclpy_context = True
            self.node = rclpy.create_node("ur5_moveit_plan_backend")
            self._owns_node = True
            return True
        except Exception as e:
            self._node_error = str(e)
            return False

    def _moveit_interface_available(self) -> bool:
        if self._moveit_py is not None:
            return True
        if self._node_error:
            self._moveit_error = f"{MOVEIT_UNAVAILABLE_MESSAGE} rclpy node error: {self._node_error}"
            return False

        try:
            from moveit.planning import MoveItPy
        except Exception:
            self._moveit_error = MOVEIT_UNAVAILABLE_MESSAGE
            return False

        try:
            self._moveit_py = MoveItPy(node_name="ur5_moveit_plan_only")
            return True
        except Exception as e:
            self._moveit_error = f"MoveIt Python planning interface failed to initialize: {e}"
            return False

    @staticmethod
    def _apply_scaling_if_supported(
        planning_component,
        velocity_scale: float,
        acceleration_scale: float,
    ) -> None:
        if hasattr(planning_component, "set_max_velocity_scaling_factor"):
            planning_component.set_max_velocity_scaling_factor(velocity_scale)
        if hasattr(planning_component, "set_max_acceleration_scaling_factor"):
            planning_component.set_max_acceleration_scaling_factor(acceleration_scale)

    def _summary_from_plan_result(
        self,
        plan_id: str,
        target_type: str,
        target_name: Optional[str],
        target_pose: Optional[Dict[str, Any]],
        plan_result,
    ) -> MotionPlanSummary:
        trajectory = getattr(plan_result, "trajectory", None)
        joint_trajectory = getattr(trajectory, "joint_trajectory", None)
        points = list(getattr(joint_trajectory, "points", []) or [])
        joint_names = list(getattr(joint_trajectory, "joint_names", []) or [])
        success = bool(plan_result) and trajectory is not None

        return MotionPlanSummary(
            plan_id=plan_id,
            target_type=target_type,
            target_name=target_name,
            target_pose=target_pose,
            planning_group=self.planning_group,
            success=success,
            message="MoveIt plan generated." if success else "MoveIt did not return a valid plan.",
            num_trajectory_points=len(points),
            duration_sec=self._duration_from_points(points),
            joint_names=joint_names,
            max_joint_delta_rad=self._max_joint_delta(points),
            max_velocity_abs=self._max_velocity_abs(points),
            trajectory_preview=self._trajectory_preview(points),
        )

    def _failure_result(
        self,
        target_type: str,
        message: str,
        skill_id: str,
        step_id: Optional[int],
        plan_id: Optional[str] = None,
        target_name: Optional[str] = None,
        target_pose: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        summary = MotionPlanSummary(
            plan_id=plan_id or self._new_plan_id(),
            target_type=target_type,
            target_name=target_name,
            target_pose=target_pose,
            planning_group=self.planning_group,
            success=False,
            message=message,
            trajectory_preview={
                "dry_run": self.dry_run,
                "robot": self.robot_name,
                "base_frame": self.base_frame,
                "ee_frame": self.ee_frame,
            },
        )
        entry = self._cache_plan(
            summary=summary,
            skill_id=skill_id,
            step_id=step_id,
            raw_plan_available=False,
        )
        return self._result_from_entry(entry)

    def _cache_plan(
        self,
        summary: MotionPlanSummary,
        skill_id: str,
        step_id: Optional[int] = None,
        raw_plan_available: bool = False,
    ) -> MotionPlanCacheEntry:
        entry = MotionPlanCacheEntry(
            plan_id=summary.plan_id,
            skill_id=skill_id,
            step_id=step_id,
            summary=summary,
            created_at=datetime.now(timezone.utc).isoformat(),
            raw_plan_available=raw_plan_available,
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

    @staticmethod
    def _duration_from_points(points) -> Optional[float]:
        if not points:
            return None
        duration = getattr(points[-1], "time_from_start", None)
        if duration is None:
            return None
        sec = getattr(duration, "sec", 0)
        nanosec = getattr(duration, "nanosec", 0)
        return float(sec) + float(nanosec) / 1_000_000_000.0

    @staticmethod
    def _max_joint_delta(points) -> Optional[float]:
        if len(points) < 2:
            return None
        first = list(getattr(points[0], "positions", []) or [])
        last = list(getattr(points[-1], "positions", []) or [])
        if not first or not last:
            return None
        return max(abs(end - start) for start, end in zip(first, last))

    @staticmethod
    def _max_velocity_abs(points) -> Optional[float]:
        velocities = []
        for point in points:
            velocities.extend(abs(value) for value in getattr(point, "velocities", []) or [])
        if not velocities:
            return None
        return max(velocities)

    @staticmethod
    def _trajectory_preview(points) -> Dict[str, Any]:
        if not points:
            return {}
        first = list(getattr(points[0], "positions", []) or [])
        last = list(getattr(points[-1], "positions", []) or [])
        return {
            "first_point_positions": first[:6],
            "last_point_positions": last[:6],
        }
