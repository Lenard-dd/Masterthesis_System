from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional


class BaseRobotBackend(ABC):
    """Abstract base class for robot backends."""

    @abstractmethod
    def get_robot_state(self) -> Dict[str, Any]:
        """Get the current robot state."""
        pass

    @abstractmethod
    def list_named_targets(self) -> List[str]:
        """List available named targets."""
        pass

    @abstractmethod
    def move_to_named_target(self, target_name: str, velocity_scale: float = 0.1, **kwargs) -> Dict[str, Any]:
        """Move to a named target position."""
        pass

    @abstractmethod
    def open_gripper(self, **kwargs) -> Dict[str, Any]:
        """Open the gripper."""
        pass

    @abstractmethod
    def close_gripper(self, **kwargs) -> Dict[str, Any]:
        """Close the gripper."""
        pass

    @abstractmethod
    def pick_object(self, object_id: str, **kwargs) -> Dict[str, Any]:
        """Pick up an object."""
        pass

    @abstractmethod
    def place_object(self, object_id: str, target_location: str = None, target_id: str = None, **kwargs) -> Dict[str, Any]:
        """Place the held object."""
        pass

    @abstractmethod
    def stop_motion(self, **kwargs) -> Dict[str, Any]:
        """Stop current motion."""
        pass