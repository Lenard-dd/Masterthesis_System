"""UR5 MoveIt plan-only backend."""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

import yaml

from llm_skill_robot.core.execution_mode import is_execution_allowed, load_execution_config
from llm_skill_robot.core.schemas import MotionPlanCacheEntry, MotionPlanSummary
from llm_skill_robot.safety.hardware_mode import (
    HardwareMode,
    assert_arm_execution_allowed,
    load_hardware_mode,
)
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
MOVEIT_ERROR_CODE_NAMES = {
    1: "SUCCESS",
    99999: "FAILURE",
    -1: "PLANNING_FAILED",
    -2: "INVALID_MOTION_PLAN",
    -3: "MOTION_PLAN_INVALIDATED_BY_ENVIRONMENT_CHANGE",
    -4: "CONTROL_FAILED",
    -5: "UNABLE_TO_AQUIRE_SENSOR_DATA",
    -6: "TIMED_OUT",
    -7: "PREEMPTED",
    -10: "START_STATE_IN_COLLISION",
    -11: "START_STATE_VIOLATES_PATH_CONSTRAINTS",
    -12: "GOAL_IN_COLLISION",
    -13: "GOAL_VIOLATES_PATH_CONSTRAINTS",
    -14: "GOAL_CONSTRAINTS_VIOLATED",
    -15: "INVALID_GROUP_NAME",
    -16: "INVALID_GOAL_CONSTRAINTS",
    -17: "INVALID_ROBOT_STATE",
    -18: "INVALID_LINK_NAME",
    -19: "INVALID_OBJECT_NAME",
    -21: "FRAME_TRANSFORM_FAILURE",
    -22: "COLLISION_CHECKING_UNAVAILABLE",
    -23: "ROBOT_STATE_STALE",
    -24: "SENSOR_INFO_STALE",
    -31: "NO_IK_SOLUTION",
}


@dataclass(frozen=True)
class JointTrajectoryExtraction:
    joint_trajectory: Optional[Any]
    reason: str
    source: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.joint_trajectory is not None


def extract_joint_trajectory(cached_plan) -> JointTrajectoryExtraction:
    """Extract a valid JointTrajectory message-like object from a cached plan."""
    if cached_plan is None:
        return JointTrajectoryExtraction(None, "No cached plan was provided.")

    candidates = (
        ("cached_plan.joint_trajectory", getattr(cached_plan, "joint_trajectory", None)),
        (
            "cached_plan.robot_trajectory.joint_trajectory",
            getattr(getattr(cached_plan, "robot_trajectory", None), "joint_trajectory", None),
        ),
        (
            "cached_plan.display_trajectory.trajectory[0].joint_trajectory",
            _first_display_joint_trajectory(getattr(cached_plan, "display_trajectory", None)),
        ),
    )

    rejection_reasons = []
    for source, candidate in candidates:
        if candidate is None:
            rejection_reasons.append(f"{source} is missing")
            continue

        valid_candidate = _valid_joint_trajectory(candidate)
        if valid_candidate is not None:
            return JointTrajectoryExtraction(valid_candidate, f"Using {source}.", source)

        rejection_reasons.append(f"{source} is not a valid JointTrajectory")

    return JointTrajectoryExtraction(
        None,
        "No valid JointTrajectory found: " + "; ".join(rejection_reasons) + ".",
    )


def _first_display_joint_trajectory(display_trajectory):
    trajectories = getattr(display_trajectory, "trajectory", None)
    if not trajectories:
        return None
    return getattr(trajectories[0], "joint_trajectory", None)


def _valid_joint_trajectory(candidate):
    if isinstance(candidate, dict):
        return _joint_trajectory_from_dict(candidate)
    if hasattr(candidate, "joint_names") and hasattr(candidate, "points"):
        return candidate
    return None


def _joint_trajectory_from_dict(data: Dict[str, Any]):
    try:
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    except Exception:
        return None

    joint_names = data.get("joint_names")
    points = data.get("points")
    if not isinstance(joint_names, list) or not isinstance(points, list):
        return None

    trajectory = JointTrajectory()
    trajectory.joint_names = list(joint_names)
    for point_data in points:
        if isinstance(point_data, JointTrajectoryPoint):
            trajectory.points.append(point_data)
            continue
        if not isinstance(point_data, dict):
            return None

        point = JointTrajectoryPoint()
        for field in ("positions", "velocities", "accelerations", "effort"):
            values = point_data.get(field)
            if values is not None:
                setattr(point, field, list(values))
        if "time_from_start" in point_data:
            _assign_duration(point, point_data["time_from_start"])
        trajectory.points.append(point)
    return trajectory


def _assign_duration(point, duration_data) -> None:
    if hasattr(duration_data, "sec") and hasattr(duration_data, "nanosec"):
        point.time_from_start = duration_data
        return
    if not isinstance(duration_data, dict):
        return

    try:
        from builtin_interfaces.msg import Duration
    except Exception:
        return

    point.time_from_start = Duration(
        sec=int(duration_data.get("sec", 0)),
        nanosec=int(duration_data.get("nanosec", 0)),
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
        hardware_mode: Optional[HardwareMode] = None,
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
        self.hardware_mode = hardware_mode or load_hardware_mode()
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
        return self.get_cached_plan(plan_id)

    def get_cached_plan(self, plan_id: str) -> Optional[MotionPlanCacheEntry]:
        """Return a cached motion plan by plan id."""
        return self.plan_cache.get(plan_id)

    def has_executable_trajectory(self, plan_id: str) -> bool:
        """Return True when a cached plan has data that may be simulated."""
        cached_plan = self.get_cached_plan(plan_id)
        if cached_plan is None:
            return False
        if cached_plan.target_command is not None or extract_joint_trajectory(cached_plan).success:
            return True
        raw_command = self._raw_plan_cache.get(plan_id)
        return isinstance(raw_command, dict) and raw_command.get("type") == "pymoveit2_target_command"

    def has_visual_preview(self, plan_id: str) -> bool:
        """Return True when a cached plan has RViz preview data."""
        cached_plan = self.get_cached_plan(plan_id)
        return bool(cached_plan and cached_plan.has_visual_preview)

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
        trajectory_validation_allowed: bool = False,
        trajectory_review_approved: bool = False,
    ) -> Dict[str, Any]:
        """Execute a cached plan only in RViz/fake-hardware simulation mode."""
        if self.execution_mode == "real_robot":
            try:
                assert_arm_execution_allowed(self.hardware_mode)
            except PermissionError as exc:
                return self._execution_result(
                    success=False,
                    message=f"{exc} Only rviz_sim execution may be used here.",
                    plan_id=plan_id,
                    output={},
                )
        if not self.hardware_mode.arm.allow_rviz_fake_execution:
            return self._execution_result(
                success=False,
                message="RViz/fake hardware arm execution is disabled by hardware_mode.yaml.",
                plan_id=plan_id,
                output={},
            )
        mode = self.execution_mode
        if mode != "rviz_sim" or not is_execution_allowed(mode, self.execution_config):
            return self._execution_result(
                success=False,
                message="Simulated execution is only allowed when execution mode is 'rviz_sim'.",
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

        if not trajectory_validation_allowed:
            return self._execution_result(
                success=False,
                message="Trajectory validation must allow the plan before RViz/fake hardware execution.",
                plan_id=plan_id,
                output={},
            )

        if not trajectory_review_approved:
            return self._execution_result(
                success=False,
                message="Trajectory review must be approved before RViz/fake hardware execution.",
                plan_id=plan_id,
                output={},
            )

        if not final_confirmed:
            message = "Final confirmation required before RViz/fake hardware execution."
            print(message)
            return self._execution_result(
                success=False,
                message=message,
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
        if (
            self.execution_mode == "rviz_sim"
            and summary.num_trajectory_points <= 0
        ):
            return self._cache_pymoveit2_target_command(
                plan_id=plan_id,
                target=target,
                target_positions=target_positions,
                velocity_scale=velocity_scale,
                acceleration_scale=acceleration_scale,
                skill_id=kwargs.get("skill_id", "move_to_named_target"),
                step_id=kwargs.get("step_id"),
            )

        entry = self._cache_plan(
            summary=summary,
            skill_id=kwargs.get("skill_id", "move_to_named_target"),
            step_id=kwargs.get("step_id"),
            raw_plan_available=summary.success,
            robot_trajectory=self._extract_robot_trajectory(plan_result),
            joint_trajectory=self._extract_joint_trajectory(plan_result),
            trajectory_start=self._extract_trajectory_start(plan_result),
        )
        if summary.success:
            self._raw_plan_cache[summary.plan_id] = {
                "type": "pymoveit2_joint_trajectory",
                "joint_trajectory": entry.joint_trajectory,
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
        plan_id = self._new_plan_id()
        plan_result = self._request_moveit_pose_plan_service(
            frame=frame,
            position=position,
            orientation=orientation,
            velocity_scale=velocity_scale,
            acceleration_scale=acceleration_scale,
            start_state=kwargs.get("start_state"),
            position_tolerance=float(kwargs.get("position_tolerance", 0.005)),
            orientation_tolerance=float(kwargs.get("orientation_tolerance", 0.05)),
            use_orientation_constraint=bool(
                kwargs.get("use_orientation_constraint", True)
            ),
            planner_id=kwargs.get("planner_id", ""),
            pipeline_id=kwargs.get("pipeline_id", ""),
            num_planning_attempts=int(kwargs.get("num_planning_attempts", 1)),
            allowed_planning_time=float(kwargs.get("allowed_planning_time", 5.0)),
            service_timeout_sec=float(kwargs.get("service_timeout_sec", 10.0)),
        )
        if plan_result is None:
            return self._failure_result(
                target_type="pose",
                target_pose=target_pose,
                message=self._pymoveit2_error or "MoveIt pose planning service is unavailable.",
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
            robot_trajectory=self._extract_robot_trajectory(plan_result),
            joint_trajectory=self._extract_joint_trajectory(plan_result),
            trajectory_start=self._extract_trajectory_start(plan_result),
        )
        return self._result_from_entry(entry)

    def _request_moveit_pose_plan_service(
        self,
        frame: str,
        position: dict,
        orientation: dict,
        velocity_scale: float,
        acceleration_scale: float,
        start_state=None,
        position_tolerance: float = 0.005,
        orientation_tolerance: float = 0.05,
        use_orientation_constraint: bool = True,
        planner_id: str = "",
        pipeline_id: str = "",
        num_planning_attempts: int = 1,
        allowed_planning_time: float = 5.0,
        service_timeout_sec: float = 10.0,
    ):
        """Request a pose-constrained MoveIt plan without executing it."""
        if self.node is None or not hasattr(self.node, "create_client"):
            return None
        try:
            import rclpy
            from geometry_msgs.msg import Pose
            from moveit_msgs.msg import (
                BoundingVolume,
                Constraints,
                OrientationConstraint,
                PositionConstraint,
            )
            from moveit_msgs.srv import GetMotionPlan
            from shape_msgs.msg import SolidPrimitive
        except Exception as exc:
            self._pymoveit2_error = f"MoveIt pose planning service types are unavailable: {exc}"
            return None

        try:
            client = self.node.create_client(GetMotionPlan, "/plan_kinematic_path")
            if not client.wait_for_service(timeout_sec=2.0):
                self._pymoveit2_error = "MoveIt planning service /plan_kinematic_path is not available."
                return None

            request = GetMotionPlan.Request()
            motion_request = request.motion_plan_request
            motion_request.group_name = self.planning_group
            if pipeline_id and hasattr(motion_request, "pipeline_id"):
                motion_request.pipeline_id = str(pipeline_id)
            if planner_id:
                motion_request.planner_id = str(planner_id)
            motion_request.num_planning_attempts = max(1, int(num_planning_attempts))
            motion_request.allowed_planning_time = max(0.1, float(allowed_planning_time))
            motion_request.max_velocity_scaling_factor = float(velocity_scale)
            motion_request.max_acceleration_scaling_factor = float(acceleration_scale)
            if start_state is not None:
                motion_request.start_state = start_state

            goal = Constraints()
            position_constraint = PositionConstraint()
            position_constraint.header.frame_id = frame
            position_constraint.link_name = self.ee_frame
            region = BoundingVolume()
            primitive = SolidPrimitive()
            primitive.type = SolidPrimitive.BOX
            primitive.dimensions = [position_tolerance * 2.0] * 3
            region.primitives.append(primitive)
            region_pose = Pose()
            region_pose.position.x = float(position["x"])
            region_pose.position.y = float(position["y"])
            region_pose.position.z = float(position["z"])
            region_pose.orientation.w = 1.0
            region.primitive_poses.append(region_pose)
            position_constraint.constraint_region = region
            position_constraint.weight = 1.0
            goal.position_constraints.append(position_constraint)

            if use_orientation_constraint:
                orientation_constraint = OrientationConstraint()
                orientation_constraint.header.frame_id = frame
                orientation_constraint.link_name = self.ee_frame
                orientation_constraint.orientation.x = float(orientation.get("x", 0.0))
                orientation_constraint.orientation.y = float(orientation.get("y", 0.0))
                orientation_constraint.orientation.z = float(orientation.get("z", 0.0))
                orientation_constraint.orientation.w = float(orientation.get("w", 1.0))
                orientation_constraint.absolute_x_axis_tolerance = orientation_tolerance
                orientation_constraint.absolute_y_axis_tolerance = orientation_tolerance
                orientation_constraint.absolute_z_axis_tolerance = orientation_tolerance
                orientation_constraint.weight = 1.0
                goal.orientation_constraints.append(orientation_constraint)
            motion_request.goal_constraints.append(goal)

            future = client.call_async(request)
            rclpy.spin_until_future_complete(
                self.node,
                future,
                timeout_sec=max(0.1, float(service_timeout_sec)),
            )
            if not future.done() or future.result() is None:
                self._pymoveit2_error = "MoveIt pose planning service returned no response."
                return None
            return future.result().motion_plan_response
        except Exception as exc:
            self._pymoveit2_error = f"MoveIt pose planning service request failed: {exc}"
            return None

    def get_plan_end_state(self, plan_id: str):
        """Build a RobotState from a cached trajectory's final point."""
        cached = self.get_cached_plan(plan_id)
        extraction = extract_joint_trajectory(cached)
        if not extraction.success:
            return None
        trajectory = extraction.joint_trajectory
        points = list(getattr(trajectory, "points", []) or [])
        joint_names = list(getattr(trajectory, "joint_names", []) or [])
        positions = list(getattr(points[-1], "positions", []) or []) if points else []
        if not joint_names or len(positions) != len(joint_names):
            return None
        try:
            from moveit_msgs.msg import RobotState
        except Exception:
            return None
        state = RobotState()
        state.joint_state.name = joint_names
        state.joint_state.position = positions
        state.is_diff = True
        return state

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
                use_move_group_action=True,
            )
            return True
        except Exception as e:
            self._pymoveit2_error = f"pymoveit2 interface failed to initialize: {e}"
            return False

    def _moveit_execution_interface_available(self, plan_id: str) -> bool:
        return self._moveit2 is not None and self.has_executable_trajectory(plan_id)

    def _execute_plan_in_rviz_sim(self, plan_id: str) -> Dict[str, Any]:
        """Execute only a backend-cached MoveIt plan in simulated execution mode."""
        cached_plan = self.get_cached_plan(plan_id)
        cached_command = self._raw_plan_cache.get(plan_id)

        target_command = cached_plan.target_command if cached_plan is not None else None
        if target_command is None and isinstance(cached_command, dict):
            if cached_command.get("type") == "pymoveit2_target_command":
                target_command = cached_command

        if target_command is not None:
            self._debug_execution_input(
                plan_id=plan_id,
                trajectory_type="pymoveit2_target_command",
                joint_names=list(target_command.get("joint_positions_by_name", {}).keys()),
                num_points=0,
            )
            if not hasattr(self._moveit2, "move_to_configuration"):
                raise RuntimeError("pymoveit2 MoveIt2 object has no move_to_configuration method.")
            self._moveit2.move_to_configuration(target_command["joint_positions"])
            executed = True
            if hasattr(self._moveit2, "wait_until_executed"):
                executed = bool(self._moveit2.wait_until_executed())
            if not executed:
                raise RuntimeError("MoveIt target command was not executed successfully.")
            return {
                "simulated_execution": True,
                "command_type": "pymoveit2_target_command",
                "target_name": target_command["target_name"],
                "moveit_execution_succeeded": executed,
            }

        extraction = extract_joint_trajectory(cached_plan)
        if not extraction.success:
            raise RuntimeError(extraction.reason)

        joint_trajectory = extraction.joint_trajectory
        self._debug_execution_input(
            plan_id=plan_id,
            trajectory_type=type(joint_trajectory).__name__,
            joint_names=list(getattr(joint_trajectory, "joint_names", []) or []),
            num_points=len(getattr(joint_trajectory, "points", []) or []),
        )
        if not hasattr(self._moveit2, "execute"):
            raise RuntimeError("pymoveit2 MoveIt2 object has no execute method.")
        self._moveit2.execute(joint_trajectory)
        executed = True
        if hasattr(self._moveit2, "wait_until_executed"):
            executed = bool(self._moveit2.wait_until_executed())
        if not executed:
            raise RuntimeError("MoveIt joint trajectory was not executed successfully.")
        return {
            "simulated_execution": True,
            "command_type": "pymoveit2_joint_trajectory",
            "trajectory_source": extraction.source,
            "moveit_execution_succeeded": executed,
        }

    @staticmethod
    def _debug_execution_input(
        plan_id: str,
        trajectory_type: str,
        joint_names,
        num_points: int,
    ) -> None:
        print(
            "RVIZ_SIM execution input: "
            f"plan_id={plan_id}, "
            f"trajectory_type={trajectory_type}, "
            f"joint_names={joint_names}, "
            f"num_points={num_points}"
        )

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
        diagnostics = self._moveit_plan_diagnostics(plan_result, points)
        success = (
            bool(plan_result)
            and trajectory is not None
            and len(points) > 0
            and diagnostics.get("error_code_name") in (None, "SUCCESS")
        )
        message = self._moveit_plan_message(success, trajectory, points, diagnostics)
        preview = self._trajectory_preview(points)
        preview["moveit_diagnostics"] = diagnostics

        return MotionPlanSummary(
            plan_id=plan_id,
            target_type=target_type,
            target_name=target_name,
            target_pose=target_pose,
            planning_group=self.planning_group,
            success=success,
            message=message,
            num_trajectory_points=len(points),
            duration_sec=self._duration_from_points(points),
            joint_names=joint_names,
            max_joint_delta_rad=self._max_joint_delta(points),
            max_velocity_abs=self._max_velocity_abs(points),
            trajectory_preview=preview,
        )

    @classmethod
    def _moveit_plan_diagnostics(cls, plan_result, points) -> Dict[str, Any]:
        error_code = getattr(plan_result, "error_code", None)
        value = getattr(error_code, "val", None)
        if value is None:
            value = getattr(error_code, "value", None)
        try:
            value = int(value) if value is not None else None
        except (TypeError, ValueError):
            value = None
        return {
            "error_code_value": value,
            "error_code_name": cls._moveit_error_code_name(value),
            "trajectory_present": getattr(plan_result, "trajectory", None) is not None,
            "joint_trajectory_points": len(points),
        }

    @staticmethod
    def _moveit_error_code_name(value: int | None) -> str | None:
        if value is None:
            return None
        return MOVEIT_ERROR_CODE_NAMES.get(value, f"UNKNOWN_{value}")

    @staticmethod
    def _moveit_plan_message(success: bool, trajectory, points, diagnostics: Dict[str, Any]) -> str:
        if success:
            return "MoveIt plan generated with a non-empty joint trajectory."
        if trajectory is None:
            return "MoveIt did not return a trajectory."
        if diagnostics.get("error_code_name") not in (None, "SUCCESS"):
            return (
                "MoveIt pose planning failed: "
                f"{diagnostics.get('error_code_name')} "
                f"({diagnostics.get('error_code_value')})."
            )
        if len(points) <= 0:
            return "MoveIt returned an empty joint trajectory."
        return "MoveIt did not return a valid plan."

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
        display_trajectory=None,
        robot_trajectory=None,
        joint_trajectory=None,
        trajectory_start=None,
        target_command: Optional[Dict[str, Any]] = None,
    ) -> MotionPlanCacheEntry:
        entry = MotionPlanCacheEntry(
            plan_id=summary.plan_id,
            skill_id=skill_id,
            step_id=step_id,
            summary=summary,
            created_at=datetime.now(timezone.utc).isoformat(),
            mode=self.execution_mode,
            display_trajectory=display_trajectory,
            robot_trajectory=robot_trajectory,
            joint_trajectory=joint_trajectory,
            raw_plan_available=raw_plan_available,
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
