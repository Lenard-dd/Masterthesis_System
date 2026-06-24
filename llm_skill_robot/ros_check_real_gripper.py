"""Read-only ROS graph diagnostics for configuring a real Robotiq driver."""

from __future__ import annotations

from dataclasses import asdict
import pprint
import sys

import yaml

from llm_skill_robot.safety.hardware_mode import load_hardware_mode, print_hardware_mode
from llm_skill_robot.utils import get_config_dir


DRIVER_KEYWORDS = ("robotiq", "gripper", "finger")


def _load_gripper_config() -> dict:
    config_path = get_config_dir() / "gripper.yaml"
    if not config_path.is_file():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _matches_driver_name(name: str) -> bool:
    lowered = name.lower()
    return any(keyword in lowered for keyword in DRIVER_KEYWORDS)


def _matching_graph_entries(entries) -> list[tuple[str, list[str]]]:
    return sorted(
        (name, list(types))
        for name, types in entries
        if _matches_driver_name(name)
    )


def _print_entries(label: str, entries) -> None:
    matches = _matching_graph_entries(entries)
    print(f"\n{label}:")
    if not matches:
        print("- none found")
        return
    for name, types in matches:
        print(f"- {name}: {', '.join(types) if types else 'type unavailable'}")


def main() -> None:
    try:
        import rclpy
        from rclpy.action import get_action_names_and_types
    except ImportError:
        print("Error: rclpy not available. Please source ROS 2 setup.", file=sys.stderr)
        sys.exit(1)

    hardware_mode = load_hardware_mode()
    gripper_config = _load_gripper_config().get("gripper", {})
    print_hardware_mode(hardware_mode)
    print("\nHardware-mode gripper settings:")
    pprint.pprint(asdict(hardware_mode.gripper), sort_dicts=False)
    print("\ngripper.yaml driver config:")
    pprint.pprint(gripper_config.get("driver", {}), sort_dicts=False)
    print("\nREAD-ONLY DIAGNOSTIC: no gripper command will be sent.")

    node = None
    try:
        rclpy.init()
        node = rclpy.create_node("ros_check_real_gripper")
        _print_entries("Matching topics", node.get_topic_names_and_types())
        _print_entries("Matching actions", get_action_names_and_types(node))
        _print_entries("Matching services", node.get_service_names_and_types())
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
