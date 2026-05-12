"""UR5 MoveIt plan-only backend."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

import yaml

from llm_skill_robot.core.execution_mode import is_execution_allowed, load_execution_config
from llm_skill_robot.core.schemas import MotionPlanCacheEntry, MotionPlanSummary
from llm_skill_robot.utils import get_config_dir

PYMOVEIT2_UNAVAILABLE_MESSAGE = (
    "pymoveit2 Python planning interface is not available. Use dry_run=True or "
    "install pymoveit2."
)
PYMOVEIT2_PLAN_ONLY_UNAVAILABLE_MESSAGE = (
    "Plan-only is not available through this pymoveit2 API. "
    "move_to_configuration() is not used in plan_only mode."
)
PYMOVEIT2_TARGET_COMMAND_WARNING = (
    "No explicit trajectory preview available; pymoveit2 will execute target command "
    "in RViz/fake hardware only after final confirmation."
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
        execution_mode: Optional[str] = None,
        execution_config: Optional[Dict[str, Any]] = None,
        mode: Optional[str] = None,
        ur5_config: Optional[Dict[str, Any]] = None,
        named_targets: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        ur5_config = ur5_config or self._load_ur5_config()
        self.node = node
        self.robot_name = robot_name
        self.joint_names = list(ur5_config.get("joint_names", self._default_joint_names()))
        self.planning_group = ur5_config.get("group_name", planning_group)
        self.base_frame = ur5_config.get("base_link_name", base_frame)
        self.ee_frame = ur5_config.get("end_effector_name", ee_frame)
        self.dry_run = dry_run
        self.execution_config = execution_config or load_execution_config().model_dump()
        self.execution_mode = mode or execution_mode or self.execution_config.get("mode", "plan_only")
        self.named_targets = named_targets or self._load_named_targets()
        self.plan_cache: Dict[str, MotionPlanCacheEntry] = {}
        self._raw_plan_cache: Dict[str, Any] = {}
        self._owns_node = False
        self._owns_rclpy_context = False
        self._node_error: Optional[str] = None
        self._moveit2 = None
        self._pymoveit2_error: Optional[str] = None

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
            return self._plan_to_named_target_pymoveit2(
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
            return self._plan_to_pose_pymoveit2(
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

    @staticmethod
    def _default_joint_names():
        return [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]

    @staticmethod
    def _load_yaml_config(filename: str) -> Dict[str, Any]:
        try:
            config_file = get_config_dir() / filename
        except FileNotFoundError:
            config_file = Path(filename)

        if not config_file.exists():
            return {}
        with open(config_file, "r") as f:
            return yaml.safe_load(f) or {}

    def _load_ur5_config(self) -> Dict[str, Any]:
        return self._load_yaml_config("ur5.yaml")

    def _load_named_targets(self) -> Dict[str, Dict[str, float]]:
        return self._load_yaml_config("named_targets.yaml")

    def _target_joint_positions(self, target: str) -> Optional[list]:
        target_config = self.named_targets.get(target)
        if target_config is None:
            return None
        try:
            return [float(target_config[joint_name]) for joint_name in self.joint_names]
        except KeyError:
            return None

    def get_plan(self, plan_id: str) -> Optional[MotionPlanCacheEntry]:
        """Return a cached plan entry by plan id."""
        return self.plan_cache.get(plan_id)

    def list_cached_plans(self):
        """Return all cached plan entries."""
        return list(self.plan_cache.values())

    def clear_plan_cache(self) -> None:
        """Clear all cached dry-run plans."""
        self.plan_cache.clear()
        self._raw_plan_cache.clear()

    def execute_cached_plan_simulated(
        self,
        plan_id: str,
        final_confirmed: bool = False,
    ) -> Dict[str, Any]:
        """Execute a cached plan only in RViz/fake-hardware simulation mode."""
        mode = self.execution_mode
        if mode != "rviz_sim" or not is_execution_allowed(mode, self.execution_config):
            return self._execution_result(
                success=False,
                message="Simulated execution is only allowed when execution mode is 'rviz_sim'.",
                plan_id=plan_id,
                output={},
            )

        if not final_confirmed:
            return self._execution_result(
                success=False,
                message="Final human confirmation is required before simulated execution.",
                plan_id=plan_id,
                output={},
            )

        cached_plan = self.get_plan(plan_id)
        if cached_plan is None:
            return self._execution_result(
                success=False,
                message=f"No cached plan found for plan_id '{plan_id}'.",
                plan_id=plan_id,
                output={},
            )

        if not cached_plan.summary.success:
            return self._execution_result(
                success=False,
                message="Cached plan summary indicates planning failed.",
                plan_id=plan_id,
                output={},
            )

        if self.execution_config.get("allow_real_robot_execution", False):
            return self._execution_result(
                success=False,
                message="Refusing simulated execution because real_robot execution is enabled in config.",
                plan_id=plan_id,
                output={},
            )

        print("RVIZ_SIM MODE: executing only in simulated/fake hardware environment.")
        if not self._moveit_execution_interface_available(plan_id):
            return self._execution_result(
                success=False,
                message="MoveIt simulated execution interface is not available for this cached plan.",
                plan_id=plan_id,
                output={},
            )

        try:
            output = self._execute_plan_in_rviz_sim(plan_id)
        except Exception as e:
            return self._execution_result(
                success=False,
                message=f"MoveIt simulated execution failed: {e}",
                plan_id=plan_id,
                output={},
            )

        return self._execution_result(
            success=True,
            message="Cached plan executed in RViz/fake-hardware simulation.",
            plan_id=plan_id,
            output=output,
        )

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

    def _plan_to_named_target_pymoveit2(
        self,
        target: str,
        velocity_scale: float,
        acceleration_scale: float,
        **kwargs,
    ) -> Dict[str, Any]:
        target_positions = self._target_joint_positions(target)
        if target_positions is None:
            return self._failure_result(
                target_type="named_target",
                target_name=target,
                message=f"Unknown named target '{target}'.",
                skill_id=kwargs.get("skill_id", "move_to_named_target"),
                step_id=kwargs.get("step_id"),
            )

        if not self._pymoveit2_interface_available():
            return self._failure_result(
                target_type="named_target",
                target_name=target,
                message=self._pymoveit2_error or PYMOVEIT2_UNAVAILABLE_MESSAGE,
                skill_id=kwargs.get("skill_id", "move_to_named_target"),
                step_id=kwargs.get("step_id"),
            )

        plan_id = self._new_plan_id()
        plan_result = self._call_pymoveit2_plan_api(target_positions)
        if plan_result is None:
            plan_result = self._request_moveit_plan_service(
                target_positions=target_positions,
                velocity_scale=velocity_scale,
                acceleration_scale=acceleration_scale,
            )

        if plan_result is None:
            if self.execution_mode != "rviz_sim":
                return self._failure_result(
                    target_type="named_target",
                    target_name=target,
                    message=PYMOVEIT2_PLAN_ONLY_UNAVAILABLE_MESSAGE,
                    skill_id=kwargs.get("skill_id", "move_to_named_target"),
                    step_id=kwargs.get("step_id"),
                    plan_id=plan_id,
                )
            return self._cache_pymoveit2_target_command(
                plan_id=plan_id,
                target=target,
                target_positions=target_positions,
                velocity_scale=velocity_scale,
                acceleration_scale=acceleration_scale,
                skill_id=kwargs.get("skill_id", "move_to_named_target"),
                step_id=kwargs.get("step_id"),
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
            raw_trajectory=self._extract_robot_trajectory(plan_result),
            trajectory_start=self._extract_trajectory_start(plan_result),
        )
        if summary.success:
            self._raw_plan_cache[summary.plan_id] = {
                "type": "pymoveit2_joint_trajectory",
                "joint_trajectory": self._extract_joint_trajectory(plan_result),
            }
        return self._result_from_entry(entry)

    def _request_moveit_plan_service(
        self,
        target_positions,
        velocity_scale: float,
        acceleration_scale: float,
    ):
        """Request a MoveIt plan via /plan_kinematic_path without executing it."""
        if self.node is None or not hasattr(self.node, "create_client"):
            return None

        try:
            import rclpy
            from moveit_msgs.msg import Constraints, JointConstraint
            from moveit_msgs.srv import GetMotionPlan
        except Exception as e:
            self._pymoveit2_error = f"MoveIt planning service types are not available: {e}"
            return None

        try:
            client = self.node.create_client(GetMotionPlan, "/plan_kinematic_path")
            if not client.wait_for_service(timeout_sec=2.0):
                self._pymoveit2_error = "MoveIt planning service /plan_kinematic_path is not available."
                return None

            request = GetMotionPlan.Request()
            motion_request = request.motion_plan_request
            motion_request.group_name = self.planning_group
            motion_request.num_planning_attempts = 1
            motion_request.allowed_planning_time = 5.0
            motion_request.max_velocity_scaling_factor = float(velocity_scale)
            motion_request.max_acceleration_scaling_factor = float(acceleration_scale)

            goal_constraints = Constraints()
            for joint_name, joint_position in zip(self.joint_names, target_positions):
                joint_constraint = JointConstraint()
                joint_constraint.joint_name = joint_name
                joint_constraint.position = float(joint_position)
                joint_constraint.tolerance_above = 0.001
                joint_constraint.tolerance_below = 0.001
                joint_constraint.weight = 1.0
                goal_constraints.joint_constraints.append(joint_constraint)
            motion_request.goal_constraints.append(goal_constraints)

            future = client.call_async(request)
            rclpy.spin_until_future_complete(self.node, future, timeout_sec=10.0)
            if not future.done():
                self._pymoveit2_error = "Timed out waiting for MoveIt planning service response."
                return None

            response = future.result()
            if response is None:
                self._pymoveit2_error = "MoveIt planning service returned no response."
                return None

            return response.motion_plan_response
        except Exception as e:
            self._pymoveit2_error = f"MoveIt planning service request failed: {e}"
            return None

    def _call_pymoveit2_plan_api(self, target_positions):
        """Use a pymoveit2 plan-only API if this environment exposes one."""
        for method_name in (
            "plan_to_configuration",
            "plan_to_joint_configuration",
            "plan_joint_positions",
        ):
            method = getattr(self._moveit2, method_name, None)
            if method is None:
                continue
            return method(target_positions)
        return None

    def _cache_pymoveit2_target_command(
        self,
        plan_id: str,
        target: str,
        target_positions,
        velocity_scale: float,
        acceleration_scale: float,
        skill_id: str,
        step_id: Optional[int],
    ) -> Dict[str, Any]:
        joint_positions_by_name = dict(zip(self.joint_names, target_positions))
        summary = MotionPlanSummary(
            plan_id=plan_id,
            target_type="named_target",
            target_name=target,
            planning_group=self.planning_group,
            success=True,
            message=PYMOVEIT2_TARGET_COMMAND_WARNING,
            num_trajectory_points=0,
            joint_names=self.joint_names,
            trajectory_preview={
                "mode": "rviz_sim",
                "pymoveit2_target_command": True,
                "no_explicit_trajectory_preview": True,
                "warning": PYMOVEIT2_TARGET_COMMAND_WARNING,
                "robot": self.robot_name,
                "base_frame": self.base_frame,
                "ee_frame": self.ee_frame,
                "velocity_scale": velocity_scale,
                "acceleration_scale": acceleration_scale,
                "joint_positions": joint_positions_by_name,
            },
        )
        entry = self._cache_plan(
            summary=summary,
            skill_id=skill_id,
            step_id=step_id,
            raw_plan_available=False,
            target_command={
                "type": "pymoveit2_target_command",
                "target_name": target,
                "joint_positions": list(target_positions),
                "joint_positions_by_name": joint_positions_by_name,
            },
            target_command={
                "type": "pymoveit2_target_command",
                "target_name": target,
                "joint_positions": list(target_positions),
                "joint_positions_by_name": joint_positions_by_name,
            },
        )
        self._raw_plan_cache[plan_id] = {
            "type": "pymoveit2_target_command",
            "target_name": target,
            "joint_positions": list(target_positions),
            "joint_positions_by_name": joint_positions_by_name,
        }
        return self._result_from_entry(entry)

    def _plan_to_pose_pymoveit2(
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
        if not self._pymoveit2_interface_available():
            return self._failure_result(
                target_type="pose",
                target_pose=target_pose,
                message=self._pymoveit2_error or PYMOVEIT2_UNAVAILABLE_MESSAGE,
                skill_id=kwargs.get("skill_id", "move_to_pose"),
                step_id=kwargs.get("step_id"),
            )

        return self._failure_result(
            target_type="pose",
            target_pose=target_pose,
            message="pymoveit2 pose plan-only support is not implemented in this backend yet.",
            skill_id=kwargs.get("skill_id", "move_to_pose"),
            step_id=kwargs.get("step_id"),
        )

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

    def _pymoveit2_interface_available(self) -> bool:
        if self._moveit2 is not None:
            return True
        if self._node_error:
            self._pymoveit2_error = f"{PYMOVEIT2_UNAVAILABLE_MESSAGE} rclpy node error: {self._node_error}"
            return False

        try:
            from pymoveit2 import MoveIt2
        except Exception:
            self._pymoveit2_error = PYMOVEIT2_UNAVAILABLE_MESSAGE
            return False

        try:
            self._moveit2 = MoveIt2(
                node=self.node,
                joint_names=self.joint_names,
                base_link_name=self.base_frame,
                end_effector_name=self.ee_frame,
                group_name=self.planning_group,
            )
            return True
        except Exception as e:
            self._pymoveit2_error = f"pymoveit2 interface failed to initialize: {e}"
            return False

    def _moveit_execution_interface_available(self, plan_id: str) -> bool:
        return self._moveit2 is not None and plan_id in self._raw_plan_cache

    def _execute_plan_in_rviz_sim(self, plan_id: str) -> Dict[str, Any]:
        """Execute only a backend-cached MoveIt plan in simulated execution mode."""
        cached_command = self._raw_plan_cache[plan_id]
        if (
            isinstance(cached_command, dict)
            and cached_command.get("type") == "pymoveit2_target_command"
        ):
        if (
            isinstance(cached_command, dict)
            and cached_command.get("type") == "pymoveit2_target_command"
        ):
            if not hasattr(self._moveit2, "move_to_configuration"):
                raise RuntimeError("pymoveit2 MoveIt2 object has no move_to_configuration method.")
            self._moveit2.move_to_configuration(cached_command["joint_positions"])
            if hasattr(self._moveit2, "wait_until_executed"):
                self._moveit2.wait_until_executed()
            return {
                "simulated_execution": True,
                "command_type": "pymoveit2_target_command",
                "target_name": cached_command["target_name"],
            }

        if (
            isinstance(cached_command, dict)
            and cached_command.get("type") == "pymoveit2_joint_trajectory"
        ):
            joint_trajectory = cached_command.get("joint_trajectory")
            if joint_trajectory is None:
                raise RuntimeError("Cached plan has no JointTrajectory for pymoveit2 execution.")
            if not hasattr(self._moveit2, "execute"):
                raise RuntimeError("pymoveit2 MoveIt2 object has no execute method.")
            self._moveit2.execute(joint_trajectory)
            if hasattr(self._moveit2, "wait_until_executed"):
                self._moveit2.wait_until_executed()
            return {
                "simulated_execution": True,
                "command_type": "pymoveit2_joint_trajectory",
            }

        if hasattr(self._moveit2, "execute"):
            self._moveit2.execute(cached_command)
            return {"simulated_execution": True, "command_type": "pymoveit2_plan"}

        raise RuntimeError("No safe pymoveit2 execution method is available.")

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

    @staticmethod
    def _extract_robot_trajectory(plan_result):
        trajectory = getattr(plan_result, "trajectory", None)
        if trajectory is not None:
            return trajectory
        return getattr(plan_result, "planned_trajectory", None)

    @classmethod
    def _extract_joint_trajectory(cls, plan_result):
        robot_trajectory = cls._extract_robot_trajectory(plan_result)
        if robot_trajectory is not None:
            return getattr(robot_trajectory, "joint_trajectory", None)
        if hasattr(plan_result, "joint_names") and hasattr(plan_result, "points"):
            return plan_result
        return None

    @staticmethod
    def _extract_trajectory_start(plan_result):
        return getattr(plan_result, "trajectory_start", None)

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
        raw_trajectory=None,
        trajectory_start=None,
        target_command: Optional[Dict[str, Any]] = None,
    ) -> MotionPlanCacheEntry:
        entry = MotionPlanCacheEntry(
            plan_id=summary.plan_id,
            skill_id=skill_id,
            step_id=step_id,
            summary=summary,
            created_at=datetime.now(timezone.utc).isoformat(),
            raw_plan_available=raw_plan_available,
            raw_trajectory=raw_trajectory,
            trajectory_start=trajectory_start,
            target_command=target_command,
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
    def _execution_result(
        success: bool,
        message: str,
        plan_id: str,
        output: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "success": success,
            "message": message,
            "plan_id": plan_id,
            "mode": "rviz_sim",
            "output": output,
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
