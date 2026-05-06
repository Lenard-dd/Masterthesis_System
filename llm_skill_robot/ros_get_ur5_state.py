"""
ROS 2 CLI entry point to read and display UR5 joint state.

This module provides a simple entry point for reading the current UR5 robot state
via the /joint_states topic.

Usage:
    ros2 run llm_skill_robot ros-get-ur5-state
    
Or after installation:
    ros-get-ur5-state
"""

import json
import sys


def main():
    """
    Read UR5 state and print as formatted JSON.

    Initializes rclpy, creates UR5StateBackend, reads state, and shuts down cleanly.
    """
    try:
        import rclpy
    except ImportError:
        print("Error: rclpy not available. Please source ROS 2 setup.", file=sys.stderr)
        sys.exit(1)

    try:
        from llm_skill_robot.robot.ur5_state_backend import UR5StateBackend
    except ImportError as e:
        print(f"Error: Failed to import UR5StateBackend: {e}", file=sys.stderr)
        sys.exit(1)

    backend = None
    try:
        rclpy.init()
        backend = UR5StateBackend(joint_states_topic="/joint_states", robot_name="ur5")
        state = backend.get_robot_state(timeout_sec=3.0)
        print(json.dumps(state, indent=2))

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    finally:
        if backend is not None:
            try:
                backend.shutdown()
            except Exception as e:
                print(f"Warning: Error during shutdown: {e}", file=sys.stderr)
        if "rclpy" in locals() and rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
