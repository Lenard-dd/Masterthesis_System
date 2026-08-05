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
        self.objects = deepcopy(config.get("objects"))
        self.grasp_candidates = deepcopy(config.get("grasp_candidates"))
        self.fail_stage = config.get("fail_stage")
        self.verification_success = bool(config.get("verification_success", True))

    def execute(self, step, context: dict[str, Any]) -> dict[str, Any] | None:
        """Return a result for sensor-related Tools, otherwise return ``None``."""
        if step.skill_id == self.fail_stage:
            return {
                "step_id": step.step_id, "skill_id": step.skill_id,
                "mode": "rviz_sim", "success": False, "status": "FAILED",
                "message": f"Configured mock failure at {step.skill_id}.", "output": {},
            }
        if step.skill_id == "detect_object":
            query = str(step.parameters.get("query", "object"))
            objects = deepcopy(self.objects) if isinstance(self.objects, list) else [{
                "object_id": self.object_id, "label": query,
                "confidence": self.confidence, "source": "mock_perception",
                "position_summary": "base_link (0.35, -0.25, 0.24) m",
            }]
            objects = [item for item in objects if isinstance(item, dict)]
            if not objects:
                return self._result(step, "No mock target found.", {
                    "detected_count": 0, "objects": [], "source": "mock_perception",
                })
            for index, item in enumerate(objects):
                item.setdefault("object_id", f"mock_object_{index + 1}")
                item.setdefault("label", query)
                item.setdefault("confidence", self.confidence)
                item.setdefault("source", "mock_perception")
            selected = max(objects, key=lambda item: float(item.get("confidence", 0.0)))
            context["selected_object"] = selected
            context["resolved_object_id"] = selected["object_id"]
            return self._result(
                step,
                f"Mock perception detected {query!r}.",
                {"object_id": selected["object_id"], "detected_count": len(objects),
                 "objects": objects, "source": "mock_perception"},
            )

        if step.skill_id == "build_object_point_cloud":
            object_id = str(context.get("resolved_object_id", self.object_id))
            point_cloud = {
                "success": True,
                "object_id": object_id,
                "point_cloud_path": f"mock://point_clouds/{object_id}.ply",
                "frame": "base_link",
                "source": "mock_perception",
            }
            context["point_cloud"] = point_cloud
            context["resolved_object_id"] = object_id
            return self._result(step, "Mock object point cloud is available.", point_cloud)

        if step.skill_id == "generate_grasp_pose":
            object_id = str(context.get("resolved_object_id", self.object_id))
            candidates = deepcopy(self.grasp_candidates) if isinstance(self.grasp_candidates, list) else [{
                "candidate_id": f"mock_grasp_{object_id}", "score": 0.99,
                "valid": True, "ik_result": "PASSED", "collision_result": "ALLOW",
                "approach_distance": 0.06, "joint_margin": None,
                "orientation_change": None, "source": "mock_perception",
            }]
            candidates = [item for item in candidates if isinstance(item, dict)]
            for index, item in enumerate(candidates):
                item.setdefault("candidate_id", f"mock_grasp_{object_id}_{index + 1}")
                item.setdefault("object_id", object_id)
                item.setdefault("rank", index + 1)
                item.setdefault("valid", True)
                item.setdefault("ik_result", "PASSED" if item["valid"] else "FAILED")
                item.setdefault("collision_result", "ALLOW" if item["valid"] else "REJECT")
                item.setdefault("source", "mock_perception")
            candidates.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
            for index, item in enumerate(candidates):
                item["rank"] = index + 1
            candidate = candidates[0] if candidates else None
            context["grasp_candidates"] = candidates
            context["selected_grasp_candidate"] = candidate
            context["grasp_motion_poses"] = deepcopy(self.grasp_poses)
            return self._result(
                step,
                "Mock grasp candidate and RViz target poses are available.",
                {
                    "candidate_id": candidate["candidate_id"] if candidate else None,
                    "candidate": candidate,
                    "candidate_count": len(candidates),
                    "best_score": candidate.get("score") if candidate else None,
                    "candidates": candidates,
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
