"""Central, non-LLM-overridable hardware safety mode configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from llm_skill_robot.utils import get_config_dir


ARM_DISABLED_MESSAGE = "Real arm execution is disabled by hardware_mode.yaml."


@dataclass(frozen=True)
class CameraMode:
    enabled: bool
    type: str


@dataclass(frozen=True)
class SubprocessMode:
    enabled: bool
    execution: str
    conda_env: str


@dataclass(frozen=True)
class ArmMode:
    real_arm_enabled: bool
    allow_arm_execution: bool
    allow_moveit_execution: bool
    allow_rviz_fake_execution: bool
    allow_plan_only: bool
    require_human_confirmation: bool
    execution_mode: str = "disabled"


@dataclass(frozen=True)
class GripperMode:
    real_gripper_enabled: bool
    allow_gripper_execution: bool
    require_human_confirmation: bool
    max_width_m: float
    min_width_m: float
    safe_close_width_m: float
    max_force_N: float
    max_speed_mps: float


@dataclass(frozen=True)
class HardwareMode:
    mode: str
    camera: CameraMode
    sam3: SubprocessMode
    graspgenx: SubprocessMode
    arm: ArmMode
    gripper: GripperMode


def load_hardware_mode(path: str | None = None) -> HardwareMode:
    """Load and validate the central hardware mode YAML file."""
    config_path = Path(path).expanduser() if path else get_config_dir() / "hardware_mode.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Hardware mode config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle) or {}

    data = document.get("hardware_mode")
    if not isinstance(data, dict):
        raise ValueError("hardware_mode.yaml must contain a 'hardware_mode' mapping.")

    try:
        mode = HardwareMode(
            mode=str(data["mode"]),
            camera=CameraMode(**data["camera"]),
            sam3=SubprocessMode(**data["sam3"]),
            graspgenx=SubprocessMode(**data["graspgenx"]),
            arm=ArmMode(**data["arm"]),
            gripper=GripperMode(**data["gripper"]),
        )
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Invalid hardware mode config: {exc}") from exc

    _validate_mode(mode)
    return mode


def _validate_mode(mode: HardwareMode) -> None:
    boolean_fields = {
        "camera.enabled": mode.camera.enabled,
        "sam3.enabled": mode.sam3.enabled,
        "graspgenx.enabled": mode.graspgenx.enabled,
        "arm.real_arm_enabled": mode.arm.real_arm_enabled,
        "arm.allow_arm_execution": mode.arm.allow_arm_execution,
        "arm.allow_moveit_execution": mode.arm.allow_moveit_execution,
        "arm.allow_rviz_fake_execution": mode.arm.allow_rviz_fake_execution,
        "arm.allow_plan_only": mode.arm.allow_plan_only,
        "arm.require_human_confirmation": mode.arm.require_human_confirmation,
        "gripper.real_gripper_enabled": mode.gripper.real_gripper_enabled,
        "gripper.allow_gripper_execution": mode.gripper.allow_gripper_execution,
        "gripper.require_human_confirmation": mode.gripper.require_human_confirmation,
    }
    for name, value in boolean_fields.items():
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be a boolean.")

    gripper = mode.gripper
    if gripper.min_width_m > gripper.max_width_m:
        raise ValueError("Gripper min_width_m must not exceed max_width_m.")
    if not gripper.min_width_m <= gripper.safe_close_width_m <= gripper.max_width_m:
        raise ValueError("Gripper safe_close_width_m must be within the width limits.")
    if gripper.max_speed_mps < 0 or gripper.max_force_N < 0:
        raise ValueError("Gripper speed and force limits must be non-negative.")
    if mode.arm.allow_arm_execution and not mode.arm.real_arm_enabled:
        raise ValueError("Arm execution cannot be allowed while real_arm_enabled is false.")
    if mode.arm.allow_arm_execution and mode.arm.execution_mode != "restricted_real_motion":
        raise ValueError(
            "Real arm execution may only be enabled with restricted_real_motion."
        )


def print_hardware_mode(mode: HardwareMode) -> None:
    """Print the safety-relevant startup mode in a consistent format."""
    camera = "REAL D435i" if mode.camera.enabled and mode.camera.type.lower() == "d435i" else "DISABLED"
    arm = "ENABLED" if mode.arm.real_arm_enabled and mode.arm.allow_arm_execution else "DISABLED"
    gripper = "ENABLED with limits" if (
        mode.gripper.real_gripper_enabled and mode.gripper.allow_gripper_execution
    ) else "DISABLED"

    print(f"Hardware mode: {mode.mode}")
    print(f"Camera: {camera}")
    print(f"SAM3: {mode.sam3.execution.replace('_', ' ')}")
    print(f"GraspGenX: {mode.graspgenx.execution.replace('_', ' ')}")
    print(f"Arm execution: {arm}")
    if arm == "ENABLED":
        print(f"Arm execution mode: {mode.arm.execution_mode}")
    print(f"Gripper execution: {gripper}")
    if gripper != "DISABLED":
        print(
            "Gripper limits: "
            f"width=[{mode.gripper.min_width_m:.3f}, {mode.gripper.max_width_m:.3f}] m, "
            f"speed<={mode.gripper.max_speed_mps:.3f} m/s, "
            f"force<={mode.gripper.max_force_N:.1f} N"
        )


def assert_arm_execution_allowed(mode: HardwareMode) -> None:
    """Fail closed unless real-arm execution is explicitly enabled."""
    if not mode.arm.real_arm_enabled or not mode.arm.allow_arm_execution:
        raise PermissionError(ARM_DISABLED_MESSAGE)


def assert_gripper_execution_allowed(mode: HardwareMode) -> None:
    """Fail closed unless real-gripper execution is explicitly enabled."""
    if not mode.gripper.real_gripper_enabled:
        raise PermissionError("Real gripper execution is disabled by hardware_mode.yaml.")
    if not mode.gripper.allow_gripper_execution:
        raise PermissionError("Gripper execution is disabled by hardware_mode.yaml.")


def clamp_or_reject_gripper_command(
    width_m: float,
    speed_mps: float,
    force_N: float,
    mode: HardwareMode,
) -> dict[str, float]:
    """Validate a real-gripper command and return it unchanged when safe.

    Despite the historical function name, unsafe values are rejected rather
    than silently clamped so callers and operators always see the violation.
    """
    assert_gripper_execution_allowed(mode)
    values = _finite_numeric_values(width_m, speed_mps, force_N)
    width, speed, force = values
    limits = mode.gripper

    if width < limits.min_width_m:
        raise ValueError(f"width_m {width} is below minimum {limits.min_width_m}.")
    if width > limits.max_width_m:
        raise ValueError(f"width_m {width} exceeds maximum {limits.max_width_m}.")
    if speed < 0:
        raise ValueError("speed_mps must be non-negative.")
    if speed > limits.max_speed_mps:
        raise ValueError(f"speed_mps {speed} exceeds maximum {limits.max_speed_mps}.")
    if force < 0:
        raise ValueError("force_N must be non-negative.")
    if force > limits.max_force_N:
        raise ValueError(f"force_N {force} exceeds maximum {limits.max_force_N}.")

    return {"width_m": width, "speed_mps": speed, "force_N": force}


def _finite_numeric_values(*values: Any) -> tuple[float, ...]:
    import math

    try:
        converted = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError("Gripper width, speed, and force must be numeric.") from exc
    if not all(math.isfinite(value) for value in converted):
        raise ValueError("Gripper width, speed, and force must be finite.")
    return converted
