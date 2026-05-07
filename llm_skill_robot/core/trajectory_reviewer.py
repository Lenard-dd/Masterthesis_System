"""Human-readable trajectory review helpers."""

from typing import Any, Dict, List


class TrajectoryReviewer:
    """Prepare and ask for human review of a motion plan summary."""

    def review(self, summary) -> str:
        """Return a human-readable review for a MotionPlanSummary-compatible object."""
        summary_data = self._as_dict(summary)
        warnings = self._missing_value_warnings(summary_data)
        target = summary_data.get("target_name") or summary_data.get("target_pose")

        lines = [
            "Trajectory Preview Review",
            "This is only a review and not execution.",
            f"Plan ID: {summary_data.get('plan_id')}",
            f"Target type: {summary_data.get('target_type')}",
            f"Target: {target}",
            f"Planning group: {summary_data.get('planning_group')}",
            f"Success: {summary_data.get('success')}",
            f"Trajectory points: {summary_data.get('num_trajectory_points')}",
            f"Estimated duration sec: {summary_data.get('duration_sec')}",
            f"Max joint delta rad: {summary_data.get('max_joint_delta_rad')}",
            f"Max velocity abs: {summary_data.get('max_velocity_abs')}",
        ]

        if warnings:
            lines.append("Warnings:")
            lines.extend(f"- {warning}" for warning in warnings)

        return "\n".join(lines)

    def ask_for_review_approval(self, summary) -> bool:
        """Print the review and ask for explicit approval."""
        print(self.review(summary))
        answer = input(
            "\nApprove this trajectory preview for RViz/fake hardware execution? [y/N] "
        ).strip().lower()
        return answer in {"y", "yes"}

    @staticmethod
    def _as_dict(summary) -> Dict[str, Any]:
        if hasattr(summary, "model_dump"):
            return summary.model_dump()
        return dict(summary)

    @staticmethod
    def _missing_value_warnings(summary_data: Dict[str, Any]) -> List[str]:
        warnings = []
        optional_fields = [
            "duration_sec",
            "max_joint_delta_rad",
            "max_velocity_abs",
            "trajectory_preview",
        ]
        for field in optional_fields:
            value = summary_data.get(field)
            if value is None or value == {}:
                warnings.append(f"{field} is missing or unavailable.")
        return warnings
