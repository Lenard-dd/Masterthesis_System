"""Optional RViz-only joint state publisher for Robotiq 2F-140 gripper joints."""

from __future__ import annotations

from typing import Any


class Robotiq2F140JointStateSim:
    """Publish simulated gripper-only JointState messages for RViz.

    The class is disabled by default through config. It does not connect to any
    gripper driver and should only be used when no other publisher owns the same
    gripper joints.
    """

    def __init__(self, node=None, config: dict | None = None):
        mapping = (config or {}).get("gripper_joint_mapping", config or {})
        self.enabled = bool(mapping.get("enabled", False))
        self.joint_names = list(mapping.get("joint_names", []) or [])
        self.open_positions = dict(mapping.get("open_positions", {}) or {})
        self.closed_positions = dict(mapping.get("closed_positions", {}) or {})
        self.publish_topic = mapping.get("publish_topic", "/joint_states")
        self.publish_rate_hz = float(mapping.get("publish_rate_hz", 10.0))
        self.mode = mapping.get("mode", "rviz_sim")
        self.real_gripper_enabled = False
        self.max_width_m = float(mapping.get("max_width_m", 0.14))
        self.min_width_m = float(mapping.get("min_width_m", 0.0))
        self.current_width_m = self.max_width_m
        self.joint_positions = {}
        self._node = node
        self._joint_state_cls = None
        self._publisher = None
        self._timer = None

        if self.enabled and self.joint_names and node is not None:
            from sensor_msgs.msg import JointState

            self._joint_state_cls = JointState
            self._publisher = node.create_publisher(JointState, self.publish_topic, 10)

        if self.joint_names and self._mapping_complete():
            self.joint_positions = self.width_to_positions(self.current_width_m)

    def width_to_positions(self, width_m: float, max_width_m: float | None = None) -> dict[str, float]:
        max_width = self.max_width_m if max_width_m is None else float(max_width_m)
        if max_width <= self.min_width_m:
            raise ValueError("max_width_m must be greater than min_width_m")
        width = float(width_m)
        ratio = 1.0 - ((width - self.min_width_m) / (max_width - self.min_width_m))
        ratio = min(1.0, max(0.0, ratio))
        positions = {}
        for joint_name in self.joint_names:
            open_position = float(self.open_positions[joint_name])
            closed_position = float(self.closed_positions[joint_name])
            positions[joint_name] = open_position + ratio * (closed_position - open_position)
        return positions

    def set_width(self, width_m: float) -> dict:
        validation_error = self._validate_mapping()
        if validation_error is not None:
            return self._result(False, validation_error)

        try:
            width = float(width_m)
        except (TypeError, ValueError):
            return self._result(False, f"Invalid width_m value: {width_m!r}")

        if width < self.min_width_m:
            return self._result(
                False,
                f"width_m {width:.3f} below minimum {self.min_width_m:.3f}",
            )
        if width > self.max_width_m:
            return self._result(
                False,
                f"width_m {width:.3f} above maximum {self.max_width_m:.3f}",
            )

        self.current_width_m = width
        self.joint_positions = self.width_to_positions(width)
        self.publish_once()
        return self._result(True, f"Set simulated Robotiq 2F-140 width to {width:.3f} m.")

    def open(self) -> dict:
        return self.set_width(self.max_width_m)

    def close(self) -> dict:
        return self.set_width(self.min_width_m)

    def publish_once(self) -> None:
        if self._publisher is None or self._joint_state_cls is None or self._node is None:
            return
        if not self.joint_positions:
            return

        msg = self._joint_state_cls()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.name = list(self.joint_names)
        msg.position = [self.joint_positions[name] for name in self.joint_names]
        self._publisher.publish(msg)

    def start_periodic_publishing(self) -> None:
        if not self.enabled or self._node is None or self._timer is not None:
            return
        if self._publisher is None or not self.joint_names:
            return
        period_sec = 1.0 / self.publish_rate_hz if self.publish_rate_hz > 0 else 0.1
        self._timer = self._node.create_timer(period_sec, self.publish_once)

    def stop(self) -> None:
        if self._timer is None:
            return
        try:
            self._timer.cancel()
            if self._node is not None and hasattr(self._node, "destroy_timer"):
                self._node.destroy_timer(self._timer)
        finally:
            self._timer = None

    def get_state(self) -> dict:
        return self._result(True, "Simulated Robotiq 2F-140 joint state.")

    def publish_width(self, width_m: float, max_width_m: float | None = None) -> dict[str, Any]:
        del max_width_m
        return self.set_width(width_m)

    def _validate_mapping(self) -> str | None:
        if not self.joint_names:
            return "No gripper joint names configured."
        missing_open = [name for name in self.joint_names if name not in self.open_positions]
        if missing_open:
            return f"Missing open_positions for joints: {missing_open}"
        missing_closed = [name for name in self.joint_names if name not in self.closed_positions]
        if missing_closed:
            return f"Missing closed_positions for joints: {missing_closed}"
        if self.max_width_m <= self.min_width_m:
            return "max_width_m must be greater than min_width_m"
        return None

    def _mapping_complete(self) -> bool:
        return self._validate_mapping() is None

    def _result(self, success: bool, message: str) -> dict:
        return {
            "success": success,
            "message": message,
            "output": {
                "current_width_m": self.current_width_m,
                "joint_positions": dict(self.joint_positions),
                "mode": self.mode,
                "real_gripper_enabled": self.real_gripper_enabled,
            },
        }
