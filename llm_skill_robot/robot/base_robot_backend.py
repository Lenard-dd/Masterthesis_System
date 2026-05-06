from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseRobotBackend(ABC):
    """Abstract base class for robot backends."""

    @abstractmethod
    def move_to_named_target(self, target_name: str) -> bool:
        """Move to a named target position."""
        pass

    @abstractmethod
    def open_gripper(self) -> bool:
        """Open the gripper."""
        pass

    @abstractmethod
    def close_gripper(self) -> bool:
        """Close the gripper."""
        pass

    @abstractmethod
    def pick_object(self, object_id: str) -> bool:
        """Pick up an object."""
        pass

    @abstractmethod
    def place_object(self) -> bool:
        """Place the held object."""
        pass