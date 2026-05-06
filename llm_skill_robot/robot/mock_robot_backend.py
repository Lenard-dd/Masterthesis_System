from .base_robot_backend import BaseRobotBackend


class MockRobotBackend(BaseRobotBackend):
    """Mock implementation of robot backend for testing."""

    def move_to_named_target(self, target_name: str) -> bool:
        print(f"[MOCK] Moving to named target: {target_name}")
        return True

    def open_gripper(self) -> bool:
        print("[MOCK] Opening gripper")
        return True

    def close_gripper(self) -> bool:
        print("[MOCK] Closing gripper")
        return True

    def pick_object(self, object_id: str) -> bool:
        print(f"[MOCK] Picking object: {object_id}")
        return True

    def place_object(self) -> bool:
        print("[MOCK] Placing object")
        return True