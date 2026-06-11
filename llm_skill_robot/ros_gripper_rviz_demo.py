"""Interactive RViz-only Robotiq 2F-140 gripper visual control demo."""

import pprint
import sys

import yaml

from llm_skill_robot.robot.robotiq_2f140_joint_state_sim import Robotiq2F140JointStateSim
from llm_skill_robot.robot.robotiq_2f140_sim_backend import Robotiq2F140SimBackend
from llm_skill_robot.utils import get_config_dir


WARNING = "RVIZ GRIPPER SIM MODE. No real gripper hardware is controlled."


def _load_yaml_config(filename: str) -> dict:
    config_file = get_config_dir() / filename
    if not config_file.exists():
        return {}
    with open(config_file, "r") as f:
        return yaml.safe_load(f) or {}


def _print_result(result: dict) -> None:
    print("\nResult:")
    pprint.pprint(result, sort_dicts=False)


def main() -> None:
    try:
        import rclpy
    except ImportError:
        print("Error: rclpy not available. Please source ROS 2 setup.", file=sys.stderr)
        sys.exit(1)

    print(WARNING)
    gripper_config = _load_yaml_config("gripper.yaml")
    mapping_config = _load_yaml_config("gripper_joint_mapping.yaml")
    mapping = mapping_config.get("gripper_joint_mapping", {})

    print("\nConfigured gripper joint names:")
    for joint_name in mapping.get("joint_names", []) or []:
        print(f"- {joint_name}")
    print(
        "\nIf joint_state_publisher_gui is also publishing these joints, RViz may "
        "flicker or ignore updates. Close joint_state_publisher_gui or disable "
        "duplicate publishing."
    )

    rclpy.init()
    node = rclpy.create_node("ros_gripper_rviz_demo")
    joint_state_sim = None
    try:
        joint_state_sim = Robotiq2F140JointStateSim(node=node, config=mapping_config)
        backend = Robotiq2F140SimBackend(
            config=gripper_config,
            joint_state_sim=joint_state_sim,
        )
        joint_state_sim.start_periodic_publishing()
        backend.open_gripper()

        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            choice = input(
                "\n[o] open gripper\n"
                "[c] close gripper\n"
                "[w] set width\n"
                "[s] state\n"
                "[q] quit\n> "
            ).strip().lower()

            if choice == "q":
                break
            if choice == "o":
                result = backend.open_gripper()
            elif choice == "c":
                result = backend.close_gripper()
            elif choice == "w":
                raw_width = input("Width in meters: ").strip()
                try:
                    width_m = float(raw_width)
                except ValueError:
                    result = {
                        "success": False,
                        "message": f"Invalid width value: {raw_width!r}",
                        "output": backend.get_gripper_state()["output"],
                    }
                else:
                    result = backend.set_gripper_width(width_m)
            elif choice == "s":
                result = backend.get_gripper_state()
            else:
                result = {
                    "success": False,
                    "message": f"Unknown command: {choice}",
                    "output": backend.get_gripper_state()["output"],
                }
            _print_result(result)
            rclpy.spin_once(node, timeout_sec=0.05)

    except KeyboardInterrupt:
        pass
    finally:
        if joint_state_sim is not None:
            joint_state_sim.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
