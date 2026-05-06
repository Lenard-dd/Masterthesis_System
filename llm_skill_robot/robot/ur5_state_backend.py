"""Read-only UR5 backend that subscribes to /joint_states."""

import time
from typing import Any, Dict, Optional


class UR5StateBackend:
    """Read-only backend for UR5 joint state inspection."""

    def __init__(
        self,
        node=None,
        joint_states_topic: str = "/joint_states",
        robot_name: str = "ur5",
    ):
        """
        Initialize UR5StateBackend.

        Args:
            node: Optional rclpy.node.Node. If None, creates internal node.
            joint_states_topic: Topic to subscribe to joint states. Default: "/joint_states"
            robot_name: Robot identifier. Default: "ur5"
        """
        self.robot_name = robot_name
        self.joint_states_topic = joint_states_topic
        self.node = node
        self.subscription = None
        self.latest_joint_state: Optional[Any] = None
        self.joint_state_received_time: Optional[float] = None
        self._owns_node = node is None
        self._owns_rclpy_context = False
        self._is_shutdown = False

        if self.node is None:
            import rclpy

            if not rclpy.ok():
                rclpy.init()
                self._owns_rclpy_context = True

            try:
                self.node = rclpy.create_node("ur5_state_backend")
            except Exception as e:
                if self._owns_rclpy_context and rclpy.ok():
                    rclpy.shutdown()
                raise RuntimeError(f"Failed to create ROS 2 node: {e}") from e

        self._setup_subscription()

    def _setup_subscription(self):
        """Set up subscription to /joint_states."""
        try:
            from sensor_msgs.msg import JointState

            self.subscription = self.node.create_subscription(
                JointState,
                self.joint_states_topic,
                self._joint_state_callback,
                qos_profile=10
            )
        except Exception as e:
            raise RuntimeError(f"Failed to set up joint state subscription: {e}") from e

    def _joint_state_callback(self, msg) -> None:
        """Store incoming joint state messages."""
        self.latest_joint_state = msg
        self.joint_state_received_time = time.time()

    def get_robot_state(self, timeout_sec: float = 3.0) -> Dict[str, Any]:
        """
        Get current robot state from latest /joint_states message.

        Args:
            timeout_sec: Time to wait for a joint state message (seconds).

        Returns:
            Dictionary with robot state information.
        """
        self._wait_for_joint_state(timeout_sec)

        if self.latest_joint_state is None:
            return {
                "robot": self.robot_name,
                "joint_positions": {},
                "joint_velocities": {},
                "joint_efforts": {},
                "ee_pose": None,
                "status": {
                    "joint_state_received": False,
                    "message": "No /joint_states message received."
                }
            }

        joint_state = self.latest_joint_state

        return {
            "robot": self.robot_name,
            "joint_positions": self._sequence_by_joint_name(
                joint_state.name,
                joint_state.position,
            ),
            "joint_velocities": self._sequence_by_joint_name(
                joint_state.name,
                joint_state.velocity,
            ),
            "joint_efforts": self._sequence_by_joint_name(
                joint_state.name,
                joint_state.effort,
            ),
            "ee_pose": None,
            "status": {
                "joint_state_received": True,
                "message": "Joint state received."
            }
        }

    def _wait_for_joint_state(self, timeout_sec: float) -> None:
        """Spin the node until a joint state arrives or the timeout expires."""
        if self.latest_joint_state is not None or timeout_sec <= 0:
            return

        import rclpy

        deadline = time.monotonic() + timeout_sec
        while self.latest_joint_state is None and time.monotonic() < deadline:
            remaining = max(0.0, deadline - time.monotonic())
            try:
                rclpy.spin_once(self.node, timeout_sec=min(0.1, remaining))
            except Exception as e:
                self.node.get_logger().warn(f"Error while waiting for joint state: {e}")
                break

    @staticmethod
    def _sequence_by_joint_name(joint_names, values) -> Dict[str, float]:
        """Return a name-to-value mapping when the JointState field is populated."""
        return {
            name: float(values[index])
            for index, name in enumerate(joint_names)
            if index < len(values)
        }

    def shutdown(self):
        """Shut down the backend cleanly."""
        if self._is_shutdown:
            return

        try:
            if self.subscription is not None:
                self.node.destroy_subscription(self.subscription)
                self.subscription = None
        except Exception as e:
            if hasattr(self.node, "get_logger"):
                self.node.get_logger().warn(f"Error destroying subscription: {e}")

        if self._owns_node and self.node is not None:
            try:
                self.node.destroy_node()
            except Exception as e:
                print(f"Error destroying internal node: {e}")

        if self._owns_rclpy_context:
            try:
                import rclpy

                if rclpy.ok():
                    rclpy.shutdown()
            except Exception as e:
                print(f"Error shutting down rclpy: {e}")

        self._is_shutdown = True

    def __enter__(self):
        """Support use as a context manager."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Clean up ROS resources when leaving a context manager."""
        self.shutdown()
