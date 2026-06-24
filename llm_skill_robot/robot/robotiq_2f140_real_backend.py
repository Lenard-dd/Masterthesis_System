"""Safety-gated real Robotiq 2F-140 backend with explicit driver adapters."""

from __future__ import annotations

from typing import Any

import yaml

from llm_skill_robot.safety.hardware_mode import (
    HardwareMode,
    assert_gripper_execution_allowed,
    clamp_or_reject_gripper_command,
    load_hardware_mode,
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
    ):
        raw_config = self._load_config() if config is None else config
        self.config = raw_config.get("gripper", raw_config)
        self.driver_config = dict(self.config.get("driver", {}))
        self.node = node
        self.hardware_mode = hardware_mode or load_hardware_mode()

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
        return self.set_gripper_width(
            self.safe_close_width_m if width_m is None else width_m,
            speed_mps=speed_mps,
            force_N=force_N,
            command="close_gripper",
            **kwargs,
        )

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
        if driver_result.get("success", False):
            self.current_width_m = safe_command["width_m"]
            self.last_command = {"command": command_name, **safe_command}

        output = dict(driver_result.get("output", {}))
        output.setdefault("requested_command", command_name)
        output.setdefault("validated_command", safe_command)
        return self._result(
            bool(driver_result.get("success", False)),
            str(driver_result.get("message", DRIVER_NOT_CONFIGURED_MESSAGE)),
            status=str(driver_result.get("status", "NOT_AVAILABLE")),
            extra_output=output,
        )

    def get_gripper_state(self, **kwargs: Any) -> dict:
        state_topic = str(self.driver_config.get("state_topic", "")).strip()
        # A command target is not feedback. A typed state adapter will populate
        # feedback in a later driver-specific integration step.
        feedback_available = False
        if not state_topic:
            message = "Real gripper feedback topic is not configured."
        elif not feedback_available:
            message = "Real gripper feedback is not available."
        else:
            message = "Real gripper state is available."

        return self._result(
            feedback_available,
            message,
            status="OK" if feedback_available else "NOT_AVAILABLE",
            extra_output={"feedback_available": feedback_available},
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
        return {
            "success": False,
            "status": "NOT_AVAILABLE",
            "message": f"Unsupported real gripper driver interface_type: {interface_type!r}.",
            "output": {"command_sent": False},
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
