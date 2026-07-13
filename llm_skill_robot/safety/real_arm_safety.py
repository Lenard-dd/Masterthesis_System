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
    max_approach_distance_m: float
    max_retreat_distance_m: float
    max_lift_distance_m: float
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
    grasp: dict[str, Any]
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
            grasp=dict(data["grasp"]),
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
    if config.execution_mode not in _SUPPORTED_EXECUTION_MODES:
        return _blocked("Real arm safety execution_mode is not supported.")
    if skill_id in config.blocked_real_motion_skills:
        return _blocked(f"{skill_id} is blocked for real-arm motion.")
    if skill_id not in config.allowed_real_motion_skills:
        return _blocked(f"{skill_id} is not allowed for real-arm motion.")
    return _allowed(f"{skill_id} is allowed for real-arm motion.")


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


def validate_approach_distance(
    distance_m: Any,
    config: RealArmSafetyConfig,
) -> dict[str, Any]:
    """Validate the final Cartesian approach distance for a real grasp."""
    return _validate_distance(
        distance_m, config.limits.max_approach_distance_m, "approach_distance_m"
    )


def validate_retreat_distance(
    distance_m: Any,
    config: RealArmSafetyConfig,
) -> dict[str, Any]:
    """Validate the retreat distance after a real grasp."""
    return _validate_distance(
        distance_m, config.limits.max_retreat_distance_m, "retreat_distance_m"
    )


def validate_lift_distance(
    distance_m: Any,
    config: RealArmSafetyConfig,
) -> dict[str, Any]:
    """Validate the supervised post-contact lift distance."""
    return _validate_distance(
        distance_m, config.limits.max_lift_distance_m, "lift_distance_m"
    )


def validate_grasp_score(score: Any, config: RealArmSafetyConfig) -> dict[str, Any]:
    """Validate that a selected grasp meets the configured quality threshold."""
    if not config.grasp.get("require_min_grasp_score", False):
        return _allowed("Minimum grasp-score validation is disabled.")
    numeric_score = _finite_float(score, "grasp_score")
    minimum = _finite_float(config.grasp.get("min_grasp_score"), "min_grasp_score")
    if numeric_score < minimum:
        return _blocked(f"grasp_score {numeric_score} is below minimum {minimum}.")
    return _allowed(f"grasp_score {numeric_score} meets minimum {minimum}.")


def validate_contact_close_allowed(config: RealArmSafetyConfig) -> dict[str, Any]:
    """Allow object-contact closing only when both grasp and gripper gates allow it."""
    if config.execution_mode != "full_grasp_supervised":
        return _blocked("Contact close requires full_grasp_supervised mode.")
    if not config.grasp.get("allow_contact_close", False):
        return _blocked("Contact close is disabled by grasp safety configuration.")
    if not config.gripper.get("allow_real_gripper_during_grasp", False):
        return _blocked("Real gripper use during grasp is disabled.")
    if not config.gripper.get("allow_close_gripper", False):
        return _blocked("Gripper close is disabled by safety configuration.")
    if not config.gripper.get("contact_close_requires_confirmation", False):
        return _blocked("Contact close must require explicit confirmation.")
    return _allowed("Contact close is allowed with its required per-step confirmation.")


def validate_full_grasp_step(
    skill_id: str,
    summary: dict[str, Any],
    config: RealArmSafetyConfig,
) -> dict[str, Any]:
    """Validate one supervised grasp step; never authorize a whole pick sequence.

    This is deliberately step-scoped.  The executor must still collect the
    configured confirmation immediately before executing the individual step.
    """
    if config.execution_mode != "full_grasp_supervised":
        return _blocked("Full grasp steps require full_grasp_supervised mode.")
    if not isinstance(summary, dict):
        raise ValueError("Full grasp step summary must be a mapping.")

    if skill_id in config.blocked_real_motion_skills:
        return _blocked(f"{skill_id} is blocked for full supervised grasp.")
    if skill_id in {"pick_object", "execute_pick_sequence"}:
        return _blocked(f"{skill_id} is not permitted; execute only individual grasp steps.")

    if skill_id == "close_gripper":
        result = validate_contact_close_allowed(config)
    else:
        result = is_real_motion_skill_allowed(skill_id, config)
    if not result["allowed"]:
        return result

    distance = summary.get("distance_m")
    if skill_id == "approach_grasp":
        if distance is None:
            return _blocked("approach_grasp requires distance_m.")
        result = validate_approach_distance(distance, config)
    elif skill_id == "retreat_grasp":
        if distance is None:
            return _blocked("retreat_grasp requires distance_m.")
        result = validate_retreat_distance(distance, config)
    elif skill_id == "lift_object":
        if not config.grasp.get("allow_lift_after_close", False):
            return _blocked("Lift after close is disabled by grasp safety configuration.")
        if distance is None:
            return _blocked("lift_object requires distance_m.")
        result = validate_lift_distance(distance, config)
    if not result["allowed"]:
        return result

    if "grasp_score" in summary:
        result = validate_grasp_score(summary["grasp_score"], config)
        if not result["allowed"]:
            return result

    return {
        "allowed": True,
        "reason": f"{skill_id} satisfies full supervised grasp limits.",
        "requires_confirmation": config.require_human_confirmation_each_motion
        or config.grasp.get("require_final_confirmation_each_step", False),
        "confirmation_phrase": (
            config.gripper["confirmation_phrase"]
            if skill_id == "close_gripper"
            else config.confirmation_phrase
        ),
    }


def require_execute_confirmation(
    prompt_text: str,
    confirmation_phrase: str = "EXECUTE",
) -> bool:
    """Accept only the exact configured real-arm execution confirmation phrase."""
    return prompt_text == confirmation_phrase


_SUPPORTED_EXECUTION_MODES = frozenset({"restricted_real_motion", "full_grasp_supervised"})


def _validate_config(config: RealArmSafetyConfig) -> None:
    if config.execution_mode not in _SUPPORTED_EXECUTION_MODES:
        raise ValueError(
            "real_arm_safety.execution_mode must be restricted_real_motion or "
            "full_grasp_supervised."
        )
    if not config.confirmation_phrase:
        raise ValueError("confirmation_phrase must not be empty.")
    if set(config.allowed_real_motion_skills).intersection(config.blocked_real_motion_skills):
        raise ValueError("Allowed and blocked real-motion skills must not overlap.")

    limits = config.limits
    positive_limits = {
        "max_cartesian_step_m": limits.max_cartesian_step_m,
        "max_approach_distance_m": limits.max_approach_distance_m,
        "max_retreat_distance_m": limits.max_retreat_distance_m,
        "max_lift_distance_m": limits.max_lift_distance_m,
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
    if not isinstance(config.grasp.get("allow_contact_close"), bool):
        raise ValueError("grasp.allow_contact_close must be a boolean.")
    if config.execution_mode == "full_grasp_supervised":
        required_grasp_flags = (
            "require_detected_object",
            "require_base_link_pose",
            "require_grasp_plan",
            "require_min_grasp_score",
            "require_trajectory_review",
            "require_final_confirmation_each_step",
            "allow_contact_close",
            "allow_lift_after_close",
        )
        for name in required_grasp_flags:
            if not isinstance(config.grasp.get(name), bool):
                raise ValueError(f"grasp.{name} must be a boolean.")
        _finite_float(config.grasp.get("min_grasp_score"), "min_grasp_score")
        if not config.require_human_confirmation_each_motion:
            raise ValueError(
                "full_grasp_supervised requires confirmation for each real motion."
            )
        if not config.grasp["require_final_confirmation_each_step"]:
            raise ValueError(
                "full_grasp_supervised requires final confirmation for every step."
            )
        if not isinstance(config.gripper.get("contact_close_requires_confirmation"), bool):
            raise ValueError("gripper.contact_close_requires_confirmation must be a boolean.")
        if not config.gripper.get("confirmation_phrase"):
            raise ValueError("gripper.confirmation_phrase must not be empty.")


def _validate_distance(distance_m: Any, limit: float, name: str) -> dict[str, Any]:
    distance = _finite_float(distance_m, name)
    if distance < 0:
        return _blocked(f"{name} must be non-negative.")
    if distance > limit:
        return _blocked(f"{name} {distance} exceeds limit {limit}.")
    return _allowed(f"{name} {distance} is within limit {limit}.")


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
