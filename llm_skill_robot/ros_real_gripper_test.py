"""Interactive, safety-gated test CLI for a real Robotiq 2F-140 gripper."""

from __future__ import annotations

import pprint
import sys
from typing import Any

import yaml

from llm_skill_robot.robot.robotiq_2f140_real_backend import Robotiq2F140RealBackend
from llm_skill_robot.safety.hardware_mode import (
    HardwareMode,
    assert_gripper_execution_allowed,
    clamp_or_reject_gripper_command,
    load_hardware_mode,
)
from llm_skill_robot.utils import get_config_dir


CONFIRMATION_PROMPT = "Send this command to the REAL gripper? Type YES to continue:"


def confirmation_is_yes(response: str) -> bool:
    """Return true only for the exact, case-sensitive confirmation token."""
    return response == "YES"


def build_command_preview(width_m: float, speed_mps: float, force_N: float) -> dict:
    """Build the three-field preview shown before every real command."""
    return {
        "width_m": float(width_m),
        "speed_mps": float(speed_mps),
        "force_N": float(force_N),
    }


def close_command_preview(
    gripper_config: dict,
    hardware_mode: HardwareMode,
    speed_mps: float | None = None,
    force_N: float | None = None,
) -> dict:
    """Build a safe-close preview without ever falling back to fully closed."""
    width = float(
        gripper_config.get(
            "safe_close_width_m", hardware_mode.gripper.safe_close_width_m
        )
    )
    speed = float(
        gripper_config.get("default_speed_mps", hardware_mode.gripper.max_speed_mps)
        if speed_mps is None
        else speed_mps
    )
    force = float(
        gripper_config.get("default_force_N", hardware_mode.gripper.max_force_N)
        if force_N is None
        else force_N
    )
    return build_command_preview(width, speed, force)


def _load_gripper_config() -> dict:
    config_path = get_config_dir() / "gripper.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Gripper config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle) or {}
    config = document.get("gripper")
    if not isinstance(config, dict):
        raise ValueError("gripper.yaml must contain a 'gripper' mapping.")
    return config


def _print_startup_configuration(
    hardware_mode: HardwareMode, gripper_config: dict
) -> None:
    limits = hardware_mode.gripper
    print("REAL GRIPPER TEST MODE")
    print("This tool never sends arm plans or arm controller commands.")
    print(f"Configured arm execution mode: {hardware_mode.arm.execution_mode}")
    print(f"max_width_m: {limits.max_width_m}")
    print(f"min_width_m: {limits.min_width_m}")
    print(f"safe_close_width_m: {limits.safe_close_width_m}")
    print(f"max_speed_mps: {limits.max_speed_mps}")
    print(f"max_force_N: {limits.max_force_N}")
    print("driver interface config:")
    pprint.pprint(gripper_config.get("driver", {}), sort_dicts=False)


def _prompt_optional_float(label: str, default: float) -> float:
    raw_value = input(f"{label} [{default}]: ")
    if raw_value == "":
        return float(default)
    return float(raw_value)


def _prompt_command_preview(
    choice: str, gripper_config: dict, hardware_mode: HardwareMode
) -> dict:
    default_speed = float(gripper_config.get("default_speed_mps", 0.02))
    default_force = float(gripper_config.get("default_force_N", 10.0))

    if choice == "w":
        width = float(input("Width in meters: "))
    elif choice == "o":
        width = float(
            gripper_config.get(
                "default_open_width_m", hardware_mode.gripper.max_width_m
            )
        )
    elif choice == "c":
        width = float(
            gripper_config.get(
                "safe_close_width_m", hardware_mode.gripper.safe_close_width_m
            )
        )
    else:
        raise ValueError(f"Unsupported real gripper command: {choice!r}")

    speed = _prompt_optional_float("Speed in m/s", default_speed)
    force = _prompt_optional_float("Force in N", default_force)
    preview = build_command_preview(width, speed, force)
    return clamp_or_reject_gripper_command(mode=hardware_mode, **preview)


def _print_command_preview(preview: dict) -> None:
    print("\nCommand preview:")
    print(f"width_m: {preview['width_m']}")
    print(f"speed_mps: {preview['speed_mps']}")
    print(f"force_N: {preview['force_N']}")


def _send_confirmed_command(
    backend: Robotiq2F140RealBackend, choice: str, preview: dict
) -> dict:
    common: dict[str, Any] = {
        "speed_mps": preview["speed_mps"],
        "force_N": preview["force_N"],
        "confirmed": True,
    }
    if choice == "o":
        return backend.open_gripper(**common)
    if choice == "c":
        return backend.close_gripper(**common)
    return backend.set_gripper_width(preview["width_m"], **common)


def assert_real_gripper_test_allowed(mode: HardwareMode) -> None:
    """Permit a gripper-only test even when supervised arm execution is enabled.

    The CLI has no arm backend and cannot publish arm trajectories.  Requiring
    the arm to be globally disabled prevented safe, independently confirmed
    gripper tests in full_grasp_supervised mode.
    """
    assert_gripper_execution_allowed(mode)


def main() -> None:
    try:
        import rclpy
    except ImportError:
        print("Error: rclpy not available. Please source ROS 2 setup.", file=sys.stderr)
        sys.exit(1)

    hardware_mode = load_hardware_mode()
    gripper_config = _load_gripper_config()
    try:
        assert_real_gripper_test_allowed(hardware_mode)
    except PermissionError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return

    _print_startup_configuration(hardware_mode, gripper_config)
    node = None
    try:
        rclpy.init()
        node = rclpy.create_node("ros_real_gripper_test")
        backend = Robotiq2F140RealBackend(
            node=node,
            config={"gripper": gripper_config},
            hardware_mode=hardware_mode,
        )

        while rclpy.ok():
            choice = input(
                "\n[o] open gripper\n"
                "[c] safe close gripper\n"
                "[w] set width\n"
                "[s] state\n"
                "[q] quit\n> "
            ).strip().lower()

            if choice == "q":
                break
            if choice == "s":
                pprint.pprint(backend.get_gripper_state(), sort_dicts=False)
                continue
            if choice not in {"o", "c", "w"}:
                print(f"Unknown command: {choice!r}")
                continue

            try:
                preview = _prompt_command_preview(
                    choice, gripper_config, hardware_mode
                )
            except (TypeError, ValueError, PermissionError) as exc:
                print(f"Command rejected: {exc}")
                continue

            _print_command_preview(preview)
            confirmation = input(f"{CONFIRMATION_PROMPT} ")
            if not confirmation_is_yes(confirmation):
                print("Command cancelled. No real gripper command was sent.")
                continue

            result = _send_confirmed_command(backend, choice, preview)
            print("\nResult:")
            pprint.pprint(result, sort_dicts=False)
    except KeyboardInterrupt:
        print("\nInterrupted. No further commands will be sent.")
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
