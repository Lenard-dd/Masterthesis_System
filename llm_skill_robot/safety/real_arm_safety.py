"""Restricted real-arm motion safety configuration and validators."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from llm_skill_robot.utils import get_config_dir


@dataclass(frozen=True)
class RealArmLimits:
    max_cartesian_step_m: float
    max_joint_delta_rad: float
    max_total_joint_delta_rad: float
    max_velocity_scale: float
    max_acceleration_scale: float
    min_planning_time_sec: float


@dataclass(frozen=True)
class RealArmSafetyConfig:
    enabled: bool
    execution_mode: str
    require_human_confirmation_each_motion: bool
    confirmation_phrase: str
    limits: RealArmLimits
    allowed_real_motion_skills: tuple[str, ...]
    blocked_real_motion_skills: tuple[str, ...]
    allowed_named_targets: tuple[str, ...]
    gripper: dict[str, Any]


def load_real_arm_safety(path: str | None = None) -> RealArmSafetyConfig:
    """Load and validate the restricted real-arm safety config."""
    config_path = Path(path).expanduser() if path else get_config_dir() / "real_arm_safety.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Real arm safety config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle) or {}

    data = document.get("real_arm_safety")
    if not isinstance(data, dict):
        raise ValueError("real_arm_safety.yaml must contain a 'real_arm_safety' mapping.")

    try:
        limits = RealArmLimits(**data["limits"])
        config = RealArmSafetyConfig(
            enabled=bool(data["enabled"]),
            execution_mode=str(data["execution_mode"]),
            require_human_confirmation_each_motion=bool(
                data["require_human_confirmation_each_motion"]
            ),
            confirmation_phrase=str(data["confirmation_phrase"]),
            limits=limits,
            allowed_real_motion_skills=tuple(data["allowed_real_motion_skills"]),
            blocked_real_motion_skills=tuple(data["blocked_real_motion_skills"]),
            allowed_named_targets=tuple(data["allowed_named_targets"]),
            gripper=dict(data["gripper"]),
        )
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Invalid real arm safety config: {exc}") from exc

    _validate_config(config)
    return config


def print_real_arm_safety(config: RealArmSafetyConfig) -> None:
    """Print a concise operator-facing summary of restricted motion limits."""
    status = "ENABLED" if config.enabled else "DISABLED"
    print(f"Real arm safety: {status}")
    print(f"Execution mode: {config.execution_mode}")
    print(
        "Motion limits: "
        f"cartesian_step<={config.limits.max_cartesian_step_m:.3f} m, "
        f"joint_delta<={config.limits.max_joint_delta_rad:.3f} rad, "
        f"total_joint_delta<={config.limits.max_total_joint_delta_rad:.3f} rad, "
        f"velocity_scale<={config.limits.max_velocity_scale:.3f}, "
        f"acceleration_scale<={config.limits.max_acceleration_scale:.3f}"
    )
    if config.require_human_confirmation_each_motion:
        print(f"Each motion requires exact confirmation: {config.confirmation_phrase}")


def is_real_motion_skill_allowed(
    skill_id: str,
    config: RealArmSafetyConfig,
) -> dict[str, Any]:
    """Return whether a skill may request restricted real-arm motion."""
    if not config.enabled:
        return _blocked("Real arm safety is disabled.")
    if config.execution_mode != "restricted_real_motion":
        return _blocked("Real arm safety execution_mode must be restricted_real_motion.")
    if skill_id in config.blocked_real_motion_skills:
        return _blocked(f"{skill_id} is blocked for restricted real-arm motion.")
    if skill_id not in config.allowed_real_motion_skills:
        return _blocked(f"{skill_id} is not allowed for restricted real-arm motion.")
    return _allowed(f"{skill_id} is allowed for restricted real-arm motion.")


def validate_named_target(
    target_name: str,
    config: RealArmSafetyConfig,
) -> dict[str, Any]:
    """Return whether a named target is allowed for restricted real motion."""
    if target_name not in config.allowed_named_targets:
        return _blocked(f"Named target '{target_name}' is not allowed.")
    return _allowed(f"Named target '{target_name}' is allowed.")


def validate_motion_summary(
    summary: dict,
    config: RealArmSafetyConfig,
) -> dict[str, Any]:
    """Validate a planned motion summary against restricted real-arm limits."""
    checks = (
        ("max_cartesian_step_m", config.limits.max_cartesian_step_m),
        ("max_joint_delta_rad", config.limits.max_joint_delta_rad),
        ("total_joint_delta_rad", config.limits.max_total_joint_delta_rad),
        ("velocity_scale", config.limits.max_velocity_scale),
        ("acceleration_scale", config.limits.max_acceleration_scale),
    )
    for key, limit in checks:
        value = summary.get(key)
        if value is None:
            continue
        numeric_value = _finite_float(value, key)
        if numeric_value > limit:
            return _blocked(f"{key} {numeric_value} exceeds limit {limit}.")

    planning_time = summary.get("planning_time_sec")
    if planning_time is not None:
        numeric_planning_time = _finite_float(planning_time, "planning_time_sec")
        if numeric_planning_time < config.limits.min_planning_time_sec:
            return _blocked(
                "planning_time_sec "
                f"{numeric_planning_time} is below minimum {config.limits.min_planning_time_sec}."
            )

    return _allowed("Motion summary satisfies restricted real-arm limits.")


def require_execute_confirmation(
    prompt_text: str,
    confirmation_phrase: str = "EXECUTE",
) -> bool:
    """Accept only the exact configured real-arm execution confirmation phrase."""
    return prompt_text == confirmation_phrase


def _validate_config(config: RealArmSafetyConfig) -> None:
    if config.execution_mode != "restricted_real_motion":
        raise ValueError("real_arm_safety.execution_mode must be restricted_real_motion.")
    if not config.confirmation_phrase:
        raise ValueError("confirmation_phrase must not be empty.")
    if set(config.allowed_real_motion_skills).intersection(config.blocked_real_motion_skills):
        raise ValueError("Allowed and blocked real-motion skills must not overlap.")

    limits = config.limits
    positive_limits = {
        "max_cartesian_step_m": limits.max_cartesian_step_m,
        "max_joint_delta_rad": limits.max_joint_delta_rad,
        "max_total_joint_delta_rad": limits.max_total_joint_delta_rad,
        "max_velocity_scale": limits.max_velocity_scale,
        "max_acceleration_scale": limits.max_acceleration_scale,
        "min_planning_time_sec": limits.min_planning_time_sec,
    }
    for name, value in positive_limits.items():
        numeric_value = _finite_float(value, name)
        if numeric_value <= 0:
            raise ValueError(f"{name} must be positive.")

    if not isinstance(config.gripper.get("allow_close_gripper"), bool):
        raise ValueError("gripper.allow_close_gripper must be a boolean.")


def _finite_float(value: Any, name: str) -> float:
    import math

    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric.") from exc
    if not math.isfinite(numeric_value):
        raise ValueError(f"{name} must be finite.")
    return numeric_value


def _allowed(message: str) -> dict[str, Any]:
    return {"allowed": True, "reason": message}


def _blocked(message: str) -> dict[str, Any]:
    return {"allowed": False, "reason": message}
