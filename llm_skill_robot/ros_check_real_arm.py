"""Read-only readiness diagnostics for restricted real UR5 arm testing."""

from __future__ import annotations

from dataclasses import asdict
import pprint
import sys
from typing import Any

from llm_skill_robot.safety.hardware_mode import load_hardware_mode, print_hardware_mode
from llm_skill_robot.safety.real_arm_safety import (
    load_real_arm_safety,
    print_real_arm_safety,
)


UR5_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)


def main() -> None:
    """Run a read-only real UR5 readiness check."""
    try:
        import rclpy
    except ImportError:
        print("Error: rclpy not available. Please source ROS 2 setup.", file=sys.stderr)
        sys.exit(1)

    try:
        hardware_mode = load_hardware_mode()
        real_arm_safety = load_real_arm_safety()
    except Exception as exc:
        print("REAL UR5 STATUS CHECK")
        print(f"BLOCKED: failed to load safety configuration: {exc}")
        print("No motion was executed.")
        return

    print("REAL UR5 STATUS CHECK")
    print_hardware_mode(hardware_mode)
    print_real_arm_safety(real_arm_safety)
    print("\nHardware-mode arm settings:")
    pprint.pprint(asdict(hardware_mode.arm), sort_dicts=False)
    print("\nReal-arm safety limits:")
    pprint.pprint(asdict(real_arm_safety.limits), sort_dicts=False)
    print("\nREAD-ONLY DIAGNOSTIC: no arm motion, trajectory, or gripper command will be sent.")

    node = None
    try:
        rclpy.init()
        node = rclpy.create_node("ros_check_real_arm")
        status = run_status_checks(node, hardware_mode, real_arm_safety)
        print_status_report(status)
    except Exception as exc:
        print(f"WARNING: real UR5 status check failed before completion: {exc}")
        print("No motion was executed.")
    finally:
        if node is not None:
            try:
                node.destroy_node()
            except Exception as exc:
                print(f"WARNING: failed to destroy ROS node cleanly: {exc}")
        if rclpy.ok():
            rclpy.shutdown()


def run_status_checks(node, hardware_mode, real_arm_safety) -> dict[str, Any]:
    joint_state = wait_for_joint_state(node)
    joint_positions = positions_by_joint_name(joint_state)
    found_joints = sorted(set(joint_positions).intersection(UR5_JOINT_NAMES))
    missing_joints = [joint for joint in UR5_JOINT_NAMES if joint not in joint_positions]

    return {
        "real_arm_enabled": bool(hardware_mode.arm.allow_arm_execution),
        "hardware_execution_mode": getattr(hardware_mode.arm, "execution_mode", ""),
        "safety_execution_mode": real_arm_safety.execution_mode,
        "joint_states": {
            "status": "OK" if joint_state is not None else "BLOCKED",
            "message": "Received /joint_states." if joint_state is not None else "No /joint_states received.",
        },
        "ur5_joints": {
            "status": "OK" if not missing_joints else "BLOCKED",
            "found": found_joints,
            "missing": missing_joints,
        },
        "controllers": check_controllers(node),
        "moveit_planning": check_moveit_planning(node),
        "joint_positions": joint_positions,
        "restricted_real_motion": {
            "status": (
                "OK"
                if real_arm_safety.execution_mode == "restricted_real_motion"
                else "BLOCKED"
            ),
            "message": real_arm_safety.execution_mode,
        },
    }


def wait_for_joint_state(node, timeout_sec: float = 3.0):
    """Wait for one /joint_states message without publishing anything."""
    import time

    import rclpy
    from sensor_msgs.msg import JointState

    state = {"message": None}

    def callback(msg):
        state["message"] = msg

    subscription = node.create_subscription(JointState, "/joint_states", callback, 10)
    try:
        deadline = time.monotonic() + timeout_sec
        while state["message"] is None and time.monotonic() < deadline:
            remaining = max(0.0, deadline - time.monotonic())
            rclpy.spin_once(node, timeout_sec=min(0.1, remaining))
    finally:
        node.destroy_subscription(subscription)
    return state["message"]


def positions_by_joint_name(joint_state) -> dict[str, float]:
    if joint_state is None:
        return {}
    return {
        name: float(joint_state.position[index])
        for index, name in enumerate(joint_state.name)
        if index < len(joint_state.position)
    }


def check_controllers(node, timeout_sec: float = 1.0) -> dict[str, Any]:
    """Check controller_manager/list_controllers when ros2_control is present."""
    try:
        import rclpy
        from controller_manager_msgs.srv import ListControllers
    except Exception as exc:
        return {"status": "UNKNOWN", "message": f"controller service type unavailable: {exc}"}

    service_name = "/controller_manager/list_controllers"
    try:
        client = node.create_client(ListControllers, service_name)
        if not client.wait_for_service(timeout_sec=timeout_sec):
            return {"status": "UNKNOWN", "message": f"{service_name} is not available."}
        future = client.call_async(ListControllers.Request())
        deadline = node.get_clock().now().nanoseconds + int(timeout_sec * 1e9)
        while not future.done() and node.get_clock().now().nanoseconds < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
        if not future.done():
            return {"status": "UNKNOWN", "message": "Timed out waiting for controllers."}
        response = future.result()
        controllers = [
            {"name": item.name, "state": item.state, "type": item.type}
            for item in getattr(response, "controller", [])
        ]
        active = [item for item in controllers if item["state"] == "active"]
        return {
            "status": "OK" if active else "WARNING",
            "message": f"{len(active)} active controller(s) found.",
            "controllers": controllers,
        }
    except Exception as exc:
        return {"status": "UNKNOWN", "message": f"controller check failed: {exc}"}


def check_moveit_planning(node, timeout_sec: float = 1.0) -> dict[str, str]:
    """Check whether MoveIt's read-only planning service is available."""
    try:
        from moveit_msgs.srv import GetMotionPlan
    except Exception as exc:
        return {"status": "UNKNOWN", "message": f"MoveIt service type unavailable: {exc}"}

    service_name = "/plan_kinematic_path"
    try:
        client = node.create_client(GetMotionPlan, service_name)
        if client.wait_for_service(timeout_sec=timeout_sec):
            return {"status": "OK", "message": f"{service_name} is available."}
        return {"status": "UNKNOWN", "message": f"{service_name} is not available."}
    except Exception as exc:
        return {"status": "UNKNOWN", "message": f"MoveIt planning check failed: {exc}"}


def print_status_report(status: dict[str, Any]) -> None:
    print("\nREAL UR5 STATUS CHECK SUMMARY")
    print(f"Real arm enabled: {str(status['real_arm_enabled']).lower()}")
    print(f"Execution mode: {status['hardware_execution_mode']}")
    print(f"Restricted safety mode: {status['safety_execution_mode']}")
    print(f"Joint states: {status['joint_states']['status']}")
    print(f"UR5 joints found: {status['ur5_joints']['status']}")
    print(f"Controllers: {status['controllers']['status']}")
    print(f"MoveIt planning: {status['moveit_planning']['status']}")

    if status["joint_positions"]:
        print("\nCurrent joint positions:")
        for joint_name in UR5_JOINT_NAMES:
            if joint_name in status["joint_positions"]:
                print(f"- {joint_name}: {status['joint_positions'][joint_name]:.6f} rad")

    if status["ur5_joints"]["missing"]:
        print("BLOCKED: missing UR5 joints: " + ", ".join(status["ur5_joints"]["missing"]))
    if status["restricted_real_motion"]["status"] != "OK":
        print("BLOCKED: real_arm_safety.execution_mode is not restricted_real_motion.")
    if not status["real_arm_enabled"]:
        print("WARNING: hardware_mode.arm.allow_arm_execution is false.")

    print(f"Controller detail: {status['controllers']['message']}")
    print(f"MoveIt detail: {status['moveit_planning']['message']}")
    print("No motion was executed.")


if __name__ == "__main__":
    main()
