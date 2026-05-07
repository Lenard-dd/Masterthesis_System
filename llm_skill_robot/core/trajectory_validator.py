"""Validation for plan-only motion trajectory summaries."""

from typing import Any, Dict


class TrajectoryValidator:
    """Validate a MotionPlanSummary before any execution attempt."""

    DEFAULT_LIMITS = {
        "max_joint_delta_rad": 0.5,
        "max_velocity_abs": 1.0,
        "min_duration_sec": 0.1,
    }

    def __init__(self, safety_config: dict):
        self.safety_config = safety_config or {}
        configured_limits = self.safety_config.get("trajectory_limits", {})
        self.trajectory_limits = {
            **self.DEFAULT_LIMITS,
            **(configured_limits or {}),
        }

    def validate_motion_plan_summary(self, summary) -> Dict[str, Any]:
        """Validate a MotionPlanSummary-compatible object or dict."""
        summary_data = self._as_dict(summary)
        reasons = []
        warnings = []
        is_dry_run = bool(summary_data.get("trajectory_preview", {}).get("dry_run", False))

        if not summary_data.get("success", False):
            reasons.append("Motion plan summary indicates planning failed.")

        num_points = summary_data.get("num_trajectory_points", 0)
        if not is_dry_run and num_points <= 0:
            reasons.append("Motion plan has no trajectory points.")

        max_joint_delta = summary_data.get("max_joint_delta_rad")
        configured_max_joint_delta = self.trajectory_limits.get("max_joint_delta_rad")
        if (
            max_joint_delta is not None
            and configured_max_joint_delta is not None
            and max_joint_delta > configured_max_joint_delta
        ):
            reasons.append(
                f"Max joint delta {max_joint_delta} rad exceeds limit "
                f"{configured_max_joint_delta} rad."
            )

        max_velocity = summary_data.get("max_velocity_abs")
        configured_max_velocity = self.trajectory_limits.get("max_velocity_abs")
        if (
            max_velocity is not None
            and configured_max_velocity is not None
            and max_velocity > configured_max_velocity
        ):
            reasons.append(
                f"Max velocity {max_velocity} exceeds limit {configured_max_velocity}."
            )

        duration_sec = summary_data.get("duration_sec")
        min_duration_sec = self.trajectory_limits.get("min_duration_sec")
        if duration_sec is not None:
            if duration_sec <= 0:
                reasons.append("Motion plan duration must be positive.")
            elif min_duration_sec is not None and duration_sec < min_duration_sec:
                warnings.append(
                    f"Motion plan duration {duration_sec} sec is below configured "
                    f"minimum {min_duration_sec} sec."
                )

        return {
            "decision": "BLOCK" if reasons else "ALLOW",
            "reasons": reasons,
            "warnings": warnings,
        }

    @staticmethod
    def _as_dict(summary) -> Dict[str, Any]:
        if hasattr(summary, "model_dump"):
            return summary.model_dump()
        return dict(summary)
