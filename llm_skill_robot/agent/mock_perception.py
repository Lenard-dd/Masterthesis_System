"""Deterministic perception substitute for RViz Agent integration tests.

It intentionally replaces only sensor-derived observations.  Motion, gripper,
verification, Agent policy, PlanVerifier, and HITL remain the production paths.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


DEFAULT_MOCK_GRASP_POSES = {
    "pregrasp": {
        "frame": "base_link",
        "position": {"x": 0.35, "y": -0.25, "z": 0.30},
        "orientation": {"x": 0.0, "y": 1.0, "z": 0.0, "w": 0.0},
    },
    "grasp": {
        "frame": "base_link",
        "position": {"x": 0.35, "y": -0.25, "z": 0.24},
        "orientation": {"x": 0.0, "y": 1.0, "z": 0.0, "w": 0.0},
    },
    "retreat": {
        "frame": "base_link",
        "position": {"x": 0.35, "y": -0.25, "z": 0.34},
        "orientation": {"x": 0.0, "y": 1.0, "z": 0.0, "w": 0.0},
    },
}


class MockPerceptionAdapter:
    """Provide trusted, deterministic inputs for sensor-dependent Tools only."""

    def __init__(self, config: dict[str, Any] | None = None):
        config = config or {}
        self.object_id = str(config.get("object_id", "mock_red_cube_1"))
        self.confidence = float(config.get("confidence", 0.99))
        self.grasp_poses = deepcopy(config.get("grasp_poses", DEFAULT_MOCK_GRASP_POSES))

    def execute(self, step, context: dict[str, Any]) -> dict[str, Any] | None:
        """Return a result for sensor-related Tools, otherwise return ``None``."""
        if step.skill_id == "detect_object":
            query = str(step.parameters.get("query", "object"))
            selected = {
                "object_id": self.object_id,
                "label": query,
                "confidence": self.confidence,
                "source": "mock_perception",
            }
            context["selected_object"] = selected
            context["resolved_object_id"] = self.object_id
            return self._result(
                step,
                f"Mock perception detected {query!r}.",
                {"object_id": self.object_id, "objects": [selected], "source": "mock_perception"},
            )

        if step.skill_id == "build_object_point_cloud":
            point_cloud = {
                "success": True,
                "object_id": self.object_id,
                "point_cloud_path": f"mock://point_clouds/{self.object_id}.ply",
                "frame": "base_link",
                "source": "mock_perception",
            }
            context["point_cloud"] = point_cloud
            context["resolved_object_id"] = self.object_id
            return self._result(step, "Mock object point cloud is available.", point_cloud)

        if step.skill_id == "generate_grasp_pose":
            candidate = {
                "candidate_id": f"mock_grasp_{self.object_id}",
                "object_id": self.object_id,
                "score": 0.99,
                "valid": True,
                "source": "mock_perception",
            }
            context["grasp_candidates"] = [candidate]
            context["selected_grasp_candidate"] = candidate
            context["grasp_motion_poses"] = deepcopy(self.grasp_poses)
            return self._result(
                step,
                "Mock grasp candidate and RViz target poses are available.",
                {
                    "candidate_id": candidate["candidate_id"],
                    "candidate": candidate,
                    "candidates": [candidate],
                    "source": "mock_perception",
                },
            )
        return None

    @staticmethod
    def _result(step, message: str, output: dict[str, Any]) -> dict[str, Any]:
        return {
            "step_id": step.step_id,
            "skill_id": step.skill_id,
            "mode": "rviz_sim",
            "success": True,
            "status": "SUCCESS",
            "message": message,
            "output": output,
        }
