"""Safety-gated real Robotiq 2F-140 backend with explicit driver adapters."""

from __future__ import annotations

import math
import time
from typing import Any

import yaml

from llm_skill_robot.safety.hardware_mode import (
    HardwareMode,
    assert_gripper_execution_allowed,
    clamp_or_reject_gripper_command,
    load_hardware_mode,
)
from llm_skill_robot.safety.real_arm_safety import (
    RealArmSafetyConfig,
    load_real_arm_safety,
    validate_contact_close_allowed,
)
from llm_skill_robot.utils import get_config_dir


DRIVER_NOT_CONFIGURED_MESSAGE = "Real gripper driver interface is not configured."
CONFIRMATION_REQUIRED_MESSAGE = (
    "Human confirmation is required before real gripper execution."
)


class Robotiq2F140RealBackend:
    """Adapt safe width commands to an explicitly configured ROS driver interface.

    The backend deliberately has no guessed Robotiq topic or action defaults. A
    driver adapter must be selected in ``config/gripper.yaml`` before any ROS
    command can be published.
    """

    def __init__(
        self,
        node=None,
        config: dict | None = None,
        hardware_mode: HardwareMode | None = None,
        real_arm_safety: RealArmSafetyConfig | None = None,
        confirmation_provider=None,
        audit_logger=None,
    ):
        raw_config = self._load_config() if config is None else config
        self.config = raw_config.get("gripper", raw_config)
        self.driver_config = dict(self.config.get("driver", {}))
        self.node = node
        self.hardware_mode = hardware_mode or load_hardware_mode()
        self.real_arm_safety = real_arm_safety or load_real_arm_safety()
        self.confirmation_provider = confirmation_provider or input
        self.audit_logger = audit_logger

        self.default_open_width_m = float(
            self.config.get("default_open_width_m", self.hardware_mode.gripper.max_width_m)
        )
        self.safe_close_width_m = float(
            self.config.get("safe_close_width_m", self.hardware_mode.gripper.safe_close_width_m)
        )
        self.default_speed_mps = float(
            self.config.get("default_speed_mps", self.hardware_mode.gripper.max_speed_mps)
        )
        self.default_force_N = float(
            self.config.get("default_force_N", self.hardware_mode.gripper.max_force_N)
        )
        self.real_gripper_enabled = self.config.get(
            "real_gripper_enabled",
            self.hardware_mode.gripper.real_gripper_enabled,
        ) is True
        self.current_width_m: float | None = None
        self.last_command: dict[str, float] | None = None
        self._width_publisher = None
        self._action_client = None
        self._contact_close_client = None
        self._state_subscription = None
        self._latest_feedback_width_m: float | None = None
        self._latest_feedback_time: float | None = None
        self._state_adapter_error: str | None = None
        self._configure_state_feedback()

    def open_gripper(
        self,
        width_m: float | None = None,
        speed_mps: float | None = None,
        force_N: float | None = None,
        **kwargs: Any,
    ) -> dict:
        return self.set_gripper_width(
            self.default_open_width_m if width_m is None else width_m,
            speed_mps=speed_mps,
            force_N=force_N,
            command="open_gripper",
            **kwargs,
        )

    def close_gripper(
        self,
        width_m: float | None = None,
        speed_mps: float | None = None,
        force_N: float | None = None,
        **kwargs: Any,
    ) -> dict:
        if kwargs.pop("during_contact", False) is True:
            return self.close_for_grasp(
                width_m=width_m,
                speed_mps=speed_mps,
                force_N=force_N,
                **kwargs,
            )
        return self.set_gripper_width(
            self.safe_close_width_m if width_m is None else width_m,
            speed_mps=speed_mps,
            force_N=force_N,
            command="close_gripper",
            **kwargs,
        )

    def close_for_grasp(
        self,
        width_m: float | None = None,
        speed_mps: float | None = None,
        force_N: float | None = None,
        **kwargs: Any,
    ) -> dict:
        """Close on an object only after the full-supervised YES confirmation.

        This method is intentionally separate from ordinary non-contact closing:
        a boolean ``confirmed=True`` is never sufficient for contact close.
        """
        if not self.hardware_mode.gripper.allow_contact_close:
            return self._result(
                False,
                "Contact close is disabled by hardware_mode.yaml.",
                status="REJECTED",
            )
        contact_allowed = validate_contact_close_allowed(self.real_arm_safety)
        if not contact_allowed.get("allowed", False):
            return self._result(False, str(contact_allowed["reason"]), status="REJECTED")

        configured_close_width = self.real_arm_safety.gripper.get(
            "close_width_m", self.config.get("close_width_m", self.safe_close_width_m)
        )
        resolved_width = configured_close_width if width_m is None else width_m
        resolved_speed = self.default_speed_mps if speed_mps is None else speed_mps
        resolved_force = self.default_force_N if force_N is None else force_N
        preview = {
            "width_m": resolved_width,
            "speed_mps": resolved_speed,
            "force_N": resolved_force,
            "mode": "real_grasp_contact_close",
        }
        self._audit("contact_close_requested", command_preview=preview)
        print("Real grasp contact-close command preview:")
        print(f"  width_m: {resolved_width}")
        print(f"  force_N: {resolved_force}")
        print(f"  speed_mps: {resolved_speed}")
        print("  mode: real_grasp_contact_close")

        try:
            safe_command = clamp_or_reject_gripper_command(
                width_m=resolved_width,
                speed_mps=resolved_speed,
                force_N=resolved_force,
                mode=self.hardware_mode,
            )
        except (ValueError, TypeError, PermissionError) as exc:
            return self._result(False, str(exc), status="REJECTED")

        confirmation = kwargs.get("confirmation")
        if confirmation is None:
            confirmation = self.confirmation_provider(
                "Type YES to close the REAL gripper on the object:\n> "
            )
        if confirmation != self.real_arm_safety.gripper.get("confirmation_phrase", "YES"):
            return self._result(
                False,
                "Exact YES confirmation is required before real grasp contact close.",
                status="REJECTED",
                extra_output={"command_preview": preview},
            )
        self._audit("contact_close_confirmed", command_preview=preview)

        # Some real Robotiq drivers expose a dedicated close service.  Keep the
        # normal action adapter for open/width commands, but prefer that native
        # service for an object-contact close when explicitly configured.
        if self._contact_close_interface() == "service_close_gripper":
            result = self._send_contact_close_service(safe_command)
        else:
            result = self.set_gripper_width(
                safe_command["width_m"],
                speed_mps=safe_command["speed_mps"],
                force_N=safe_command["force_N"],
                command="close_for_grasp",
                confirmed=True,
            )
        output = result.setdefault("output", {})
        output["command_preview"] = preview
        output["mode"] = "real_grasp_contact_close"
        if output.get("command_sent", False):
            self._audit("contact_close_command_sent", command_preview=preview)

        self._wait_for_state_feedback()
        feedback_width = self._latest_feedback_width_m
        if feedback_width is None and output.get("feedback_available", False):
            feedback_width = output.get("current_width_m")
        feedback_available = feedback_width is not None
        if feedback_available:
            feedback_width = float(feedback_width)
        object_contact_detected = bool(output.get("object_contact_detected", False))
        output.update(
            {
                "feedback_available": feedback_available,
                "current_width_m": feedback_width,
                "object_may_be_held": (
                    True
                    if object_contact_detected
                    else (
                        bool(feedback_width > safe_command["width_m"] + 0.002)
                        if feedback_available
                        else None
                    )
                ),
            }
        )
        self._audit(
            "contact_close_feedback",
            feedback_available=feedback_available,
            current_width_m=feedback_width,
            object_may_be_held=output["object_may_be_held"],
        )
        return result

    def set_gripper_width(
        self,
        width_m: float,
        speed_mps: float | None = None,
        force_N: float | None = None,
        **kwargs: Any,
    ) -> dict:
        command_name = str(kwargs.pop("command", "set_gripper_width"))
        confirmed = self._confirmation_from_kwargs(kwargs)
        resolved_speed = self.default_speed_mps if speed_mps is None else speed_mps
        resolved_force = self.default_force_N if force_N is None else force_N

        try:
            assert_gripper_execution_allowed(self.hardware_mode)
            if not self.real_gripper_enabled:
                raise PermissionError("Real gripper is disabled in config/gripper.yaml.")
            if self.hardware_mode.gripper.require_human_confirmation and not confirmed:
                raise PermissionError(CONFIRMATION_REQUIRED_MESSAGE)
            safe_command = clamp_or_reject_gripper_command(
                width_m=width_m,
                speed_mps=resolved_speed,
                force_N=resolved_force,
                mode=self.hardware_mode,
            )
        except (PermissionError, ValueError, TypeError) as exc:
            return self._result(False, str(exc), status="REJECTED")

        driver_result = self._send_driver_command(**safe_command)
        reported_width = None
        if driver_result.get("success", False):
            reported_width = driver_result.get("output", {}).get("current_width_m")
            self.current_width_m = (
                float(reported_width)
                if reported_width is not None
                else safe_command["width_m"]
            )
            self.last_command = {"command": command_name, **safe_command}

        output = dict(driver_result.get("output", {}))
        output.setdefault("requested_command", command_name)
        output.setdefault("validated_command", safe_command)
        output.setdefault(
            "feedback_available",
            reported_width is not None or self._latest_feedback_width_m is not None,
        )
        return self._result(
            bool(driver_result.get("success", False)),
            str(driver_result.get("message", DRIVER_NOT_CONFIGURED_MESSAGE)),
            status=str(driver_result.get("status", "NOT_AVAILABLE")),
            extra_output=output,
        )

    def test_gripper_width(
        self,
        width_m: float | None = None,
        speed_mps: float | None = None,
        force_N: float | None = None,
        **kwargs: Any,
    ) -> dict:
        """Execute the reviewed candidate width through the same safety gate."""
        if width_m is None:
            return self._result(
                False,
                "Reviewed candidate gripper_width_m has not been resolved.",
                status="REJECTED",
            )
        kwargs.pop("object_id", None)
        kwargs.pop("candidate_id", None)
        return self.set_gripper_width(
            width_m=width_m,
            speed_mps=speed_mps,
            force_N=force_N,
            command="test_gripper_width",
            **kwargs,
        )

    def get_gripper_state(self, **kwargs: Any) -> dict:
        state_topic = str(self.driver_config.get("state_topic", "")).strip()
        self._wait_for_state_feedback()
        feedback_available = self._latest_feedback_width_m is not None
        if not state_topic:
            message = "Real gripper feedback topic is not configured."
        elif self._state_adapter_error:
            message = self._state_adapter_error
        elif not feedback_available:
            message = f"No real gripper feedback received from {state_topic}."
        else:
            message = f"Real gripper state received from {state_topic}."

        return self._result(
            feedback_available,
            message,
            status="OK" if feedback_available else "NOT_AVAILABLE",
            extra_output={
                "feedback_available": feedback_available,
                "current_width_m": self._latest_feedback_width_m,
                "state_topic": state_topic or None,
                "feedback_age_sec": self._feedback_age_sec(),
            },
        )

    def _send_driver_command(
        self,
        width_m: float,
        speed_mps: float,
        force_N: float,
    ) -> dict:
        """Send one already-validated command through an explicit adapter."""
        interface_type = str(self.driver_config.get("interface_type", "auto")).strip().lower()
        command = {
            "width_m": width_m,
            "speed_mps": speed_mps,
            "force_N": force_N,
        }

        if interface_type in {"", "auto"}:
            return {
                "success": False,
                "status": "NOT_AVAILABLE",
                "message": DRIVER_NOT_CONFIGURED_MESSAGE,
                "output": {"command_preview": command, "command_sent": False},
            }
        if interface_type == "dry_run":
            return {
                "success": False,
                "status": "DRY_RUN",
                "message": "Dry-run only; no real gripper command was sent.",
                "output": {"command_preview": command, "command_sent": False},
            }
        if self.driver_config.get("width_unit", "meters") != "meters":
            return {
                "success": False,
                "status": "NOT_AVAILABLE",
                "message": "Only driver width_unit 'meters' is currently supported.",
                "output": {"command_sent": False},
            }
        if interface_type == "topic_float":
            return self._publish_float_width(command)
        if interface_type == "topic_joint_trajectory":
            return self._publish_joint_trajectory(command)
        if interface_type == "action_gripper_command":
            return self._send_gripper_command_action(command)
        return {
            "success": False,
            "status": "NOT_AVAILABLE",
            "message": f"Unsupported real gripper driver interface_type: {interface_type!r}.",
            "output": {"command_sent": False},
        }

    def _contact_close_interface(self) -> str:
        return str(
            self.driver_config.get("contact_close_interface_type", "")
        ).strip().lower()

    def _send_contact_close_service(self, command: dict[str, float]) -> dict:
        """Call a driver's explicit close service after normal safety gating."""
        service_name = str(
            self.driver_config.get(
                "contact_close_service_name",
                self.driver_config.get("service_name", ""),
            )
        ).strip()
        if not service_name or self.node is None:
            return self._driver_unavailable(
                "service_close_gripper requires a node and contact_close_service_name."
            )
        try:
            assert_gripper_execution_allowed(self.hardware_mode)
            if not self.real_gripper_enabled:
                raise PermissionError("Real gripper is disabled in config/gripper.yaml.")
            import rclpy
            from robotiq_gripper_communication.srv import CloseGripper
        except (ImportError, PermissionError) as exc:
            return self._driver_unavailable(f"Close-gripper service unavailable: {exc}")

        try:
            if self._contact_close_client is None:
                self._contact_close_client = self.node.create_client(
                    CloseGripper, service_name
                )
            timeout = float(self.driver_config.get("server_wait_timeout_sec", 2.0))
            if not self._contact_close_client.wait_for_service(timeout_sec=timeout):
                return self._driver_unavailable(
                    f"Close-gripper service is not available: {service_name}"
                )
            request = CloseGripper.Request()
            future = self._contact_close_client.call_async(request)
            timeout = float(self.driver_config.get("command_timeout_sec", 10.0))
            rclpy.spin_until_future_complete(self.node, future, timeout_sec=timeout)
            response = future.result() if future.done() else None
            if response is None:
                return {
                    "success": False,
                    "status": "TIMEOUT",
                    "message": f"Timed out waiting for close-gripper service {service_name}.",
                    "output": {"command_sent": True, "service_name": service_name},
                }
            return self._interpret_contact_close_service_response(
                response, service_name=service_name, command=command
            )
        except Exception as exc:
            return self._driver_unavailable(f"Close-gripper service failed safely: {exc}")

    @staticmethod
    def _interpret_contact_close_service_response(
        response, *, service_name: str, command: dict[str, float]
    ) -> dict:
        """Map native Robotiq object status to supervised-grasp semantics."""
        error_code = getattr(response, "error_code", None)
        try:
            error_code = int(error_code) if error_code is not None else None
        except (TypeError, ValueError):
            error_code = None
        driver_success = bool(getattr(response, "success", False))
        # ur_rtde::RobotiqGripper::STOPPED_INNER_OBJECT == 2. For an
        # explicitly approved contact-close this is the desired outcome.
        object_contact_detected = error_code == 2
        succeeded = driver_success or object_contact_detected
        native_message = str(getattr(response, "message", "")).strip()
        message = (
            "Grasp contact detected while closing; the gripper stopped on the object."
            if object_contact_detected
            else native_message or (
                f"Close-gripper service completed through {service_name}."
                if succeeded
                else f"Close-gripper service failed through {service_name}."
            )
        )
        return {
            "success": succeeded,
            "status": "SUCCEEDED" if succeeded else "FAILED",
            "message": message,
            "output": {
                "command_sent": True,
                "service_name": service_name,
                "validated_command": command,
                "error_code": error_code,
                "native_contact_close": True,
                "native_service_success": driver_success,
                "object_contact_detected": object_contact_detected,
                "native_message": native_message,
            },
        }

    def _publish_float_width(self, command: dict[str, float]) -> dict:
        topic = str(self.driver_config.get("command_topic", "")).strip()
        if not topic or self.node is None:
            return self._driver_unavailable("topic_float requires a node and command_topic.")
        try:
            from std_msgs.msg import Float64

            if self._width_publisher is None:
                self._width_publisher = self.node.create_publisher(Float64, topic, 10)
            message = Float64()
            message.data = command["width_m"]
            self._width_publisher.publish(message)
        except Exception as exc:  # ROS availability and driver errors must fail closed.
            return self._driver_unavailable(f"topic_float command failed: {exc}")
        return {
            "success": True,
            "status": "SENT",
            "message": f"Real gripper width command published to {topic}.",
            "output": {"command_sent": True, "command_topic": topic},
        }

    def _publish_joint_trajectory(self, command: dict[str, float]) -> dict:
        topic = str(self.driver_config.get("command_topic", "")).strip()
        joint_names = self.driver_config.get("joint_names", [])
        scales = self.driver_config.get("width_to_joint_scale")
        offsets = self.driver_config.get("joint_offsets")
        if not topic or self.node is None or not joint_names or scales is None or offsets is None:
            return self._driver_unavailable(
                "topic_joint_trajectory requires a node, command_topic, driver.joint_names, "
                "driver.width_to_joint_scale, and driver.joint_offsets."
            )
        try:
            scales = self._mapping_values(scales, len(joint_names), "width_to_joint_scale")
            offsets = self._mapping_values(offsets, len(joint_names), "joint_offsets")
            from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

            publisher = self.node.create_publisher(JointTrajectory, topic, 10)
            message = JointTrajectory()
            message.joint_names = list(joint_names)
            point = JointTrajectoryPoint()
            point.positions = [
                offset + scale * command["width_m"]
                for scale, offset in zip(scales, offsets)
            ]
            message.points = [point]
            publisher.publish(message)
        except Exception as exc:
            return self._driver_unavailable(f"topic_joint_trajectory command failed: {exc}")
        return {
            "success": True,
            "status": "SENT",
            "message": f"Real gripper joint trajectory published to {topic}.",
            "output": {"command_sent": True, "command_topic": topic},
        }

    def _send_gripper_command_action(self, command: dict[str, float]) -> dict:
        """Send one validated width through control_msgs/GripperCommand."""
        action_name = str(self.driver_config.get("action_name", "")).strip()
        if not action_name or self.node is None:
            return self._driver_unavailable(
                "action_gripper_command requires a node and driver.action_name."
            )
        configured_speed = self.driver_config.get("controller_speed_mps")
        if configured_speed is None:
            return self._driver_unavailable(
                "action_gripper_command requires verified driver.controller_speed_mps "
                "because GripperCommand has no speed field."
            )
        try:
            configured_speed = float(configured_speed)
        except (TypeError, ValueError):
            return self._driver_unavailable("driver.controller_speed_mps must be numeric.")
        if (
            not math.isfinite(configured_speed)
            or configured_speed < 0.0
            or configured_speed > self.hardware_mode.gripper.max_speed_mps
        ):
            return self._driver_unavailable(
                "driver.controller_speed_mps exceeds hardware_mode safety limits."
            )
        if configured_speed > command["speed_mps"]:
            return {
                "success": False,
                "status": "REJECTED",
                "message": (
                    f"Controller speed {configured_speed} m/s exceeds requested "
                    f"speed {command['speed_mps']} m/s."
                ),
                "output": {"command_sent": False, "action_name": action_name},
            }
        try:
            import rclpy
            from action_msgs.msg import GoalStatus
            from control_msgs.action import GripperCommand
            from rclpy.action import ActionClient
        except Exception as exc:
            return self._driver_unavailable(
                f"GripperCommand action types are unavailable: {exc}"
            )

        try:
            if self._action_client is None:
                self._action_client = ActionClient(
                    self.node, GripperCommand, action_name
                )
            server_timeout = float(
                self.driver_config.get("server_wait_timeout_sec", 2.0)
            )
            if not self._action_client.wait_for_server(timeout_sec=server_timeout):
                return self._driver_unavailable(
                    f"GripperCommand action server is not available: {action_name}"
                )

            driver_position = self._driver_position_from_width(command["width_m"])
            goal = GripperCommand.Goal()
            goal.command.position = driver_position
            goal.command.max_effort = command["force_N"]
            command_timeout = float(
                self.driver_config.get("command_timeout_sec", 10.0)
            )
            goal_future = self._action_client.send_goal_async(goal)
            rclpy.spin_until_future_complete(
                self.node, goal_future, timeout_sec=command_timeout
            )
            if not goal_future.done() or goal_future.result() is None:
                return self._driver_unavailable(
                    f"Timed out sending GripperCommand goal to {action_name}."
                )

            goal_handle = goal_future.result()
            if not getattr(goal_handle, "accepted", False):
                return {
                    "success": False,
                    "status": "REJECTED",
                    "message": (
                        f"GripperCommand goal was rejected by {action_name} "
                        f"for driver position {driver_position:.6f}."
                    ),
                    "output": {
                        "command_sent": False,
                        "action_name": action_name,
                        "requested_width_m": command["width_m"],
                        "commanded_driver_position": driver_position,
                    },
                }

            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(
                self.node, result_future, timeout_sec=command_timeout
            )
            wrapped_result = result_future.result() if result_future.done() else None
            if wrapped_result is None:
                return {
                    "success": False,
                    "status": "TIMEOUT",
                    "message": f"Timed out waiting for GripperCommand result from {action_name}.",
                    "output": {"command_sent": True, "action_name": action_name},
                }

            action_result = getattr(wrapped_result, "result", None)
            action_status = getattr(wrapped_result, "status", None)
            succeeded = action_status == GoalStatus.STATUS_SUCCEEDED
            result_position = getattr(action_result, "position", None)
            if result_position is not None and math.isfinite(float(result_position)):
                self._latest_feedback_width_m = self._width_from_driver_position(
                    float(result_position)
                )
                self._latest_feedback_time = time.monotonic()
            output = {
                "command_sent": True,
                "action_name": action_name,
                "action_status": action_status,
                "driver_position": result_position,
                "commanded_driver_position": driver_position,
                "current_width_m": self._latest_feedback_width_m,
                "effort_N": getattr(action_result, "effort", None),
                "stalled": bool(getattr(action_result, "stalled", False)),
                "reached_goal": bool(getattr(action_result, "reached_goal", False)),
                "speed_note": (
                    "GripperCommand has no speed field; validated controller speed is "
                    f"{configured_speed} m/s."
                ),
                "controller_speed_mps": configured_speed,
            }
            return {
                "success": succeeded,
                "status": "SUCCEEDED" if succeeded else "FAILED",
                "message": (
                    f"GripperCommand completed through {action_name}."
                    if succeeded
                    else f"GripperCommand failed through {action_name} with status {action_status}."
                ),
                "output": output,
            }
        except Exception as exc:
            return self._driver_unavailable(
                f"GripperCommand action failed safely: {exc}"
            )

    def _configure_state_feedback(self) -> None:
        state_topic = str(self.driver_config.get("state_topic", "")).strip()
        if not state_topic or self.node is None:
            return
        state_type = str(
            self.driver_config.get("state_message_type", "std_msgs/msg/Float32")
        ).strip().lower()
        try:
            if state_type == "std_msgs/msg/float32":
                from std_msgs.msg import Float32

                self._state_subscription = self.node.create_subscription(
                    Float32, state_topic, self._on_float_state, 10
                )
            elif state_type == "std_msgs/msg/float64":
                from std_msgs.msg import Float64

                self._state_subscription = self.node.create_subscription(
                    Float64, state_topic, self._on_float_state, 10
                )
            elif state_type == "sensor_msgs/msg/jointstate":
                from sensor_msgs.msg import JointState

                self._state_subscription = self.node.create_subscription(
                    JointState, state_topic, self._on_joint_state, 10
                )
            else:
                self._state_adapter_error = (
                    f"Unsupported gripper state_message_type: {state_type!r}."
                )
        except Exception as exc:
            self._state_adapter_error = (
                f"Could not subscribe to real gripper state topic {state_topic}: {exc}"
            )

    def _on_float_state(self, message) -> None:
        try:
            state_value = float(message.data)
        except (AttributeError, TypeError, ValueError):
            return
        if not math.isfinite(state_value):
            return
        try:
            width = self._width_from_driver_position(state_value)
        except ValueError:
            return
        self._latest_feedback_width_m = width
        self._latest_feedback_time = time.monotonic()
        self.current_width_m = width

    def _on_joint_state(self, message) -> None:
        position = self._extract_joint_state_position(message)
        if position is None or not math.isfinite(position):
            return
        try:
            width = self._width_from_driver_position(position)
        except ValueError:
            return
        self._latest_feedback_width_m = width
        self._latest_feedback_time = time.monotonic()
        self.current_width_m = width

    def _extract_joint_state_position(self, message) -> float | None:
        joint_names = self.driver_config.get("state_joint_names")
        if joint_names is None:
            joint_names = self.driver_config.get("joint_names")
        if isinstance(joint_names, str):
            joint_names = [joint_names]
        if joint_names:
            names = [str(name) for name in joint_names]
            positions = [
                float(pos)
                for name, pos in zip(message.name, message.position)
                if name in names
            ]
            if positions:
                return sum(positions) / len(positions)
        try:
            return float(message.position[0])
        except (AttributeError, IndexError, TypeError, ValueError):
            return None

    def _wait_for_state_feedback(self) -> None:
        if self._latest_feedback_width_m is not None:
            return
        if self._state_subscription is None or self.node is None:
            return
        try:
            import rclpy
        except Exception:
            return
        timeout = max(
            0.0, float(self.driver_config.get("state_wait_timeout_sec", 1.0))
        )
        deadline = time.monotonic() + timeout
        while self._latest_feedback_width_m is None and time.monotonic() < deadline:
            remaining = max(0.0, deadline - time.monotonic())
            rclpy.spin_once(self.node, timeout_sec=min(0.1, remaining))

    def _feedback_age_sec(self) -> float | None:
        if self._latest_feedback_time is None:
            return None
        return max(0.0, time.monotonic() - self._latest_feedback_time)

    def _driver_position_from_width(self, width_m: float) -> float:
        unit = str(
            self.driver_config.get("action_position_unit", "meters")
        ).strip()
        if unit == "meters":
            return float(width_m)
        if unit != "joint_radians":
            raise ValueError(f"Unsupported action_position_unit: {unit!r}.")
        open_position, closed_position = self._joint_position_limits()
        width_span = self.hardware_mode.gripper.max_width_m - self.hardware_mode.gripper.min_width_m
        if width_span <= 0.0:
            raise ValueError("Hardware-mode gripper width span must be positive.")
        closed_fraction = (
            self.hardware_mode.gripper.max_width_m - float(width_m)
        ) / width_span
        return open_position + closed_fraction * (closed_position - open_position)

    def _width_from_driver_position(self, position: float) -> float:
        unit = str(self.driver_config.get("state_unit", "meters")).strip()
        if unit == "meters":
            return float(position)
        if unit != "joint_radians":
            raise ValueError(f"Unsupported state_unit: {unit!r}.")
        open_position, closed_position = self._joint_position_limits()
        joint_span = closed_position - open_position
        if abs(joint_span) <= 1e-12:
            raise ValueError("Configured gripper joint-position span must be non-zero.")
        closed_fraction = (float(position) - open_position) / joint_span
        tolerance = 1e-3
        if closed_fraction < -tolerance or closed_fraction > 1.0 + tolerance:
            raise ValueError("Gripper state is outside configured joint-position limits.")
        closed_fraction = max(0.0, min(1.0, closed_fraction))
        width_span = self.hardware_mode.gripper.max_width_m - self.hardware_mode.gripper.min_width_m
        return self.hardware_mode.gripper.max_width_m - closed_fraction * width_span

    def _joint_position_limits(self) -> tuple[float, float]:
        try:
            open_position = float(self.driver_config["open_joint_position"])
            closed_position = float(self.driver_config["closed_joint_position"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "joint_radians mapping requires open_joint_position and "
                "closed_joint_position."
            ) from exc
        if not math.isfinite(open_position) or not math.isfinite(closed_position):
            raise ValueError("Gripper joint-position limits must be finite.")
        return open_position, closed_position

    @staticmethod
    def _confirmation_from_kwargs(kwargs: dict[str, Any]) -> bool:
        values = (
            kwargs.get("confirmed", False),
            kwargs.get("human_confirmed", False),
            kwargs.get("human_confirmation", False),
            kwargs.get("confirmation", False),
            kwargs.get("confirmation_approved", False),
            kwargs.get("final_confirmed", False),
        )
        return any(value is True for value in values)

    def _result(
        self,
        success: bool,
        message: str,
        *,
        status: str,
        extra_output: dict[str, Any] | None = None,
    ) -> dict:
        output = {
            "mode": "real_hardware",
            "feedback_available": False,
            "current_width_m": self.current_width_m,
            "real_gripper_enabled": self.real_gripper_enabled,
            "command_sent": False,
        }
        output.update(extra_output or {})
        return {"success": success, "status": status, "message": message, "output": output}

    def _audit(self, event: str, **fields: Any) -> None:
        if self.audit_logger is not None and hasattr(
            self.audit_logger, "log_hardware_safety_event"
        ):
            self.audit_logger.log_hardware_safety_event(event, **fields)

    @staticmethod
    def _driver_unavailable(detail: str) -> dict:
        return {
            "success": False,
            "status": "NOT_AVAILABLE",
            "message": detail,
            "output": {"command_sent": False},
        }

    @staticmethod
    def _mapping_values(value: Any, count: int, name: str) -> list[float]:
        values = value if isinstance(value, list) else [value] * count
        if len(values) != count:
            raise ValueError(f"driver.{name} must contain one value per joint.")
        return [float(item) for item in values]

    @staticmethod
    def _load_config() -> dict:
        config_path = get_config_dir() / "gripper.yaml"
        if not config_path.is_file():
            return {}
        with config_path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
