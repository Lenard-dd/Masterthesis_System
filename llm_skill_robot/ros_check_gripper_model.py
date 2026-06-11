"""Inspect /joint_states for Robotiq gripper-related joints.

This diagnostic entry point is read-only. It subscribes to /joint_states,
prints gripper-like joint names and positions, and never moves hardware.
"""

import sys
import time
from typing import Iterable


GRIPPER_JOINT_KEYWORDS = (
    "robotiq",
    "finger",
    "gripper",
    "knuckle",
    "inner",
    "outer",
)
EXPECTED_ROBOTIQ_2F140_JOINTS = ("robotiq_finger_joint",)


def _is_gripper_joint(name: str, keywords: Iterable[str] = GRIPPER_JOINT_KEYWORDS) -> bool:
    lowered = name.lower()
    return any(keyword in lowered for keyword in keywords)


def _matching_joint_positions(msg) -> list[tuple[str, float | None]]:
    positions = list(getattr(msg, "position", []) or [])
    matches = []
    for index, name in enumerate(getattr(msg, "name", []) or []):
        if _is_gripper_joint(name):
            position = positions[index] if index < len(positions) else None
            matches.append((name, position))
    return matches


def main() -> None:
    try:
        import rclpy
        from sensor_msgs.msg import JointState
    except ImportError:
        print("Error: ROS 2 Python packages not available. Please source ROS 2 setup.", file=sys.stderr)
        sys.exit(1)

    rclpy.init()
    node = rclpy.create_node("ros_check_gripper_model")
    matches_by_name: dict[str, float | None] = {}

    def callback(msg):
        for name, position in _matching_joint_positions(msg):
            matches_by_name[name] = position

    subscription = node.create_subscription(JointState, "/joint_states", callback, 10)
    del subscription

    print("Checking /joint_states for gripper-related joints...")
    print(f"Keywords: {', '.join(GRIPPER_JOINT_KEYWORDS)}")
    deadline = time.monotonic() + 3.0
    try:
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    print("\nGripper Joint Diagnostics:")
    if not matches_by_name:
        print("No gripper-like joints were found on /joint_states.")
        print("If you are using the real Robotiq xacro, check that RViz/model publishers are running.")
        print("Run the UR5 + Robotiq visualization launch and inspect joint_state_publisher_gui.")
        return

    for name in sorted(matches_by_name):
        position = matches_by_name[name]
        position_text = "N/A" if position is None else f"{position:.6f}"
        print(f"- {name}: position={position_text}")

    print("\nExpected active Robotiq 2F-140 joint:")
    for name in EXPECTED_ROBOTIQ_2F140_JOINTS:
        status = "FOUND" if name in matches_by_name else "MISSING"
        print(f"- {name}: {status}")

    print("\nIf joint_state_publisher_gui is publishing these joints, do not also enable")
    print("the optional simulated gripper joint state publisher for the same joints.")


if __name__ == "__main__":
    main()
