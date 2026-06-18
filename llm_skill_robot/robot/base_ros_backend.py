from typing import Dict, Any, List, Optional
from .base_robot_backend import BaseRobotBackend


class BaseROSBackend(BaseRobotBackend):
    """Base class for ROS 2 robot backends.
    
    This class provides the foundation for ROS 2-based robot implementations.
    It does not subscribe to topics or perform actual motion control.
    Subclasses should implement the abstract methods.
    """

    def __init__(self, robot_name: str = "ur5_robot", node=None):
        """Initialize the ROS robot backend.
        
        Args:
            robot_name: Name of the robot (used for logging and naming)
            node: rclpy Node instance. If None, will be created or obtained from context.
        """
        self.robot_name = robot_name
        self.node = node
        self._is_shutdown = False

    def get_node(self):
        """Get or create the ROS 2 node.
        
        Lazy initialization to avoid import issues when rclpy is not available.
        """
        if self.node is None:
            try:
                import rclpy
                self.node = rclpy.create_node(f"{self.robot_name}_backend")
            except Exception as e:
                raise RuntimeError(f"Failed to create ROS 2 node: {e}")
        return self.node

    def shutdown(self):
        """Gracefully shutdown the ROS backend."""
        if self.node and not self._is_shutdown:
            try:
                self.node.destroy_node()
                self._is_shutdown = True
            except Exception as e:
                self.node.get_logger().error(f"Error during shutdown: {e}")

    def get_robot_state(self) -> Dict[str, Any]:
        """Get the current robot state. Placeholder for ROS implementation."""
        return {
            "robot_name": self.robot_name,
            "is_shutdown": self._is_shutdown,
            "status": "idle"
        }

    def list_named_targets(self) -> List[str]:
        """List available named targets. Placeholder for ROS implementation."""
        return ["home", "observe"]

    def move_to_named_target(self, target_name: str, velocity_scale: float = 0.1, **kwargs) -> Dict[str, Any]:
        """Move to a named target. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement move_to_named_target")

    def open_gripper(self, **kwargs) -> Dict[str, Any]:
        """Open the gripper. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement open_gripper")

    def close_gripper(self, **kwargs) -> Dict[str, Any]:
        """Close the gripper. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement close_gripper")

    def pick_object(self, object_id: str, **kwargs) -> Dict[str, Any]:
        """Pick up an object. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement pick_object")

    def place_object(self, object_id: str, target_location: str = None, target_id: str = None, **kwargs) -> Dict[str, Any]:
        """Place the held object. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement place_object")

    def stop_motion(self, **kwargs) -> Dict[str, Any]:
        """Stop current motion. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement stop_motion")

    def __del__(self):
        """Cleanup on deletion."""
        self.shutdown()
