from .base_robot_backend import BaseRobotBackend


class MockRobotBackend(BaseRobotBackend):
    """Mock implementation of robot backend for testing."""

    def __init__(self, known_named_targets=None, known_objects=None):
        self.current_named_target = None
        self.gripper_open = False
        self.held_object = None
        self.known_named_targets = set(known_named_targets or ["home", "observe"])
        self.known_objects = set(known_objects or ["cube_1", "sphere_1"])

    def move_to_named_target(self, target: str, velocity_scale: float = 0.1, **kwargs):
        if target not in self.known_named_targets:
            return {
                "success": False,
                "message": f"Unknown named target: {target}",
                "output": {"target": target, "velocity_scale": velocity_scale}
            }

        self.current_named_target = target
        return {
            "success": True,
            "message": f"Moved to named target {target} at velocity scale {velocity_scale}.",
            "output": {"target": target, "velocity_scale": velocity_scale}
        }

    def open_gripper(self, **kwargs):
        self.gripper_open = True
        return {
            "success": True,
            "message": "Gripper opened.",
            "output": {"gripper_open": self.gripper_open}
        }

    def close_gripper(self, **kwargs):
        self.gripper_open = False
        return {
            "success": True,
            "message": "Gripper closed.",
            "output": {"gripper_open": self.gripper_open}
        }

    def pick_object(self, object_id: str, **kwargs):
        if not self.gripper_open:
            return {
                "success": False,
                "message": "Cannot pick object because gripper is closed.",
                "output": {"object_id": object_id}
            }

        if object_id not in self.known_objects:
            return {
                "success": False,
                "message": f"Object '{object_id}' is unknown.",
                "output": {"object_id": object_id}
            }

        self.held_object = object_id
        self.gripper_open = False
        return {
            "success": True,
            "message": f"Picked object {object_id} successfully.",
            "output": {"held_object": self.held_object}
        }

    def place_object(self, object_id: str, target_location: str = None, target_id: str = None, **kwargs):
        if self.held_object != object_id:
            return {
                "success": False,
                "message": "Cannot place object because the held object does not match.",
                "output": {"held_object": self.held_object, "object_id": object_id}
            }

        self.held_object = None
        self.gripper_open = True
        return {
            "success": True,
            "message": f"Placed object {object_id} successfully.",
            "output": {
                "target_location": target_location,
                "target_id": target_id,
                "gripper_open": self.gripper_open
            }
        }

    def get_robot_state(self):
        """Get the current mock robot state."""
        return {
            "current_named_target": self.current_named_target,
            "gripper_open": self.gripper_open,
            "held_object": self.held_object,
        }

    def list_named_targets(self):
        """List available named targets."""
        return list(self.known_named_targets)

    def stop_motion(self, **kwargs):
        """Stop current motion."""
        return {
            "success": True,
            "message": "Motion stopped.",
            "output": {}
        }
