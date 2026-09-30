"""GraspGenX candidate generation and deterministic best-candidate selection."""

from __future__ import annotations

import math
import inspect
from pathlib import Path
from typing import Any

from llm_skill_robot.perception.object_point_cloud_builder import ObjectPointCloudBuilder


class GraspGenXGraspPlanner:
    def __init__(self, adapter):
        self.adapter = adapter

    def generate_candidates(
        self, point_cloud_path: str, *, point_cloud: dict | None = None,
        transformer: Any | None = None,
    ) -> dict:
        """Use base-frame cloud for top-down sparse generation when TF is available.

        The original camera-frame file and diffusion path remain usable. The
        point_cloud metadata is updated only after a successful conversion so
        downstream candidate localization uses the same frame as inference.
        """
        strategy = str(getattr(self.adapter, "config", {}).get("strategy", "diffusion"))
        scene_path = (
            str(point_cloud.get("environment_point_cloud_path"))
            if point_cloud and point_cloud.get("environment_point_cloud_path")
            else None
        )
        if strategy != "sparse":
            return self._generate(
                point_cloud_path, scene_point_cloud_path=scene_path
            )
        if point_cloud is None or transformer is None:
            result = self._generate(
                point_cloud_path, strategy="diffusion",
                scene_point_cloud_path=scene_path,
            )
            result["message"] += " Sparse requested but point-cloud metadata or TF is unavailable; used diffusion."
            return result
        source_frame = str(point_cloud.get("frame") or "")
        target_frame = str(getattr(transformer, "target_frame", ""))
        if not source_frame or target_frame != "base_link":
            result = self._generate(
                point_cloud_path, strategy="diffusion",
                scene_point_cloud_path=scene_path,
            )
            result["message"] += " Sparse requested but base_link TF is unavailable; used diffusion."
            return result
        try:
            import numpy as np

            points = self._read_ascii_ply_xyz(Path(point_cloud_path))
            transformed = (
                points if source_frame == target_frame
                else transformer.transform_points(points, source_frame)
            )
            if transformed is None:
                raise ValueError(f"TF {source_frame} to {target_frame} is unavailable")
            transformed = np.asarray(transformed, dtype=float)
            if transformed.shape != points.shape or not np.isfinite(transformed).all():
                raise ValueError("TF returned an invalid point cloud")
            output_dir = Path(str(getattr(self.adapter, "config", {}).get("output_dir", "data/grasping/graspgenx_outputs")))
            output_dir.mkdir(parents=True, exist_ok=True)
            base_path = output_dir / f"{Path(point_cloud_path).stem}_base_link.ply"
            ObjectPointCloudBuilder._write_ascii_ply(base_path, transformed)
            base_scene_path = None
            if scene_path:
                scene_points = self._read_ascii_ply_xyz(Path(scene_path))
                transformed_scene = (
                    scene_points if source_frame == target_frame
                    else transformer.transform_points(scene_points, source_frame)
                )
                transformed_scene = np.asarray(transformed_scene, dtype=float)
                if transformed_scene.shape != scene_points.shape or not np.isfinite(transformed_scene).all():
                    raise ValueError("TF returned an invalid environment point cloud")
                base_scene_path = output_dir / f"{Path(point_cloud_path).stem}_environment_base_link.ply"
                ObjectPointCloudBuilder._write_ascii_ply(base_scene_path, transformed_scene)
        except (ImportError, OSError, ValueError, TypeError) as exc:
            result = self._generate(
                point_cloud_path, strategy="diffusion",
                scene_point_cloud_path=scene_path,
            )
            result["message"] += f" Sparse point-cloud conversion failed ({exc}); used diffusion."
            return result
        centroid = transformed.mean(axis=0)
        updated_metadata = {
            "point_cloud_path": str(base_path),
            "frame": target_frame,
            "centroid_m": {axis: float(centroid[index]) for index, axis in enumerate("xyz")},
            "original_point_cloud_path": point_cloud_path,
            "original_point_cloud_frame": source_frame,
            "environment_point_cloud_path": str(base_scene_path) if base_scene_path else None,
            "planar_principal_axes": ObjectPointCloudBuilder._planar_principal_axes(transformed),
        }
        result = self._generate(
            str(base_path),
            scene_point_cloud_path=str(base_scene_path) if base_scene_path else None,
        )
        if result.get("success"):
            point_cloud.update(updated_metadata)
        return result

    def _generate(
        self, point_cloud_path: str, *, strategy: str | None = None,
        scene_point_cloud_path: str | None = None,
    ) -> dict:
        """Call new adapters with scene context and preserve legacy adapters."""
        parameters = inspect.signature(self.adapter.generate_candidates).parameters
        kwargs: dict[str, Any] = {}
        if strategy is not None and "strategy" in parameters:
            kwargs["strategy"] = strategy
        if scene_point_cloud_path and "scene_point_cloud_path" in parameters:
            kwargs["scene_point_cloud_path"] = scene_point_cloud_path
        return self.adapter.generate_candidates(point_cloud_path, **kwargs)

    @staticmethod
    def _read_ascii_ply_xyz(path: Path):
        import numpy as np

        points = []
        with path.open("r", encoding="utf-8") as handle:
            if handle.readline().strip() != "ply":
                raise ValueError("Sparse strategy requires an ASCII PLY point cloud")
            in_body = False
            for line in handle:
                if not in_body:
                    if line.strip() == "end_header":
                        in_body = True
                    continue
                fields = line.split()
                if len(fields) >= 3:
                    points.append([float(value) for value in fields[:3]])
        result = np.asarray(points, dtype=float)
        if result.ndim != 2 or result.shape[1:] != (3,) or len(result) < 10:
            raise ValueError("Sparse strategy requires at least ten XYZ points")
        return result

    def select_best_candidate(self, candidates: list[Any]) -> Any | None:
        valid = [candidate for candidate in candidates if self._is_valid(candidate)]
        if not valid:
            return None
        return max(valid, key=lambda candidate: self._score(candidate))

    @classmethod
    def _is_valid(cls, candidate: Any) -> bool:
        if cls._value(candidate, "valid", True) is not True:
            return False
        return math.isfinite(cls._score(candidate))

    @classmethod
    def _score(cls, candidate: Any) -> float:
        try:
            return float(cls._value(candidate, "score", float("-inf")))
        except (TypeError, ValueError):
            return float("-inf")

    @staticmethod
    def _value(candidate: Any, name: str, default: Any = None) -> Any:
        return candidate.get(name, default) if isinstance(candidate, dict) else getattr(candidate, name, default)


def generate_candidates_with_context(
    planner: Any, point_cloud: dict, transformer: Any | None = None,
) -> dict:
    """Pass frame context to the project planner while retaining legacy planners."""
    path = point_cloud["point_cloud_path"]
    if isinstance(planner, GraspGenXGraspPlanner):
        return planner.generate_candidates(path, point_cloud=point_cloud, transformer=transformer)
    return planner.generate_candidates(path)


def screen_candidate_order(candidates: list[Any], max_candidates: int, sparse_quota: int = 12) -> list[Any]:
    """Choose the bounded candidate set passed to MoveIt screening.

    Environment-constrained candidates already arrive in the order selected
    after collision/path checks and yaw-diversity sampling. Preserve that
    order and screen the complete bounded set; sorting it again by the original
    object-only score would undo the environment-aware selection.
    """
    maximum = max(1, int(max_candidates))
    environment_selected = [
        candidate for candidate in candidates
        if bool(GraspGenXGraspPlanner._value(
            GraspGenXGraspPlanner._value(candidate, "metadata", {}) or {},
            "environment_constrained_trajectory", False,
        ))
    ]
    if environment_selected:
        selected_ids = {id(candidate) for candidate in environment_selected}
        remaining = sorted(
            (candidate for candidate in candidates if id(candidate) not in selected_ids),
            key=lambda candidate: float(
                GraspGenXGraspPlanner._value(candidate, "score", 0.0) or 0.0
            ),
            reverse=True,
        )
        return (environment_selected + remaining)[:maximum]
    ordered = sorted(
        candidates,
        key=lambda candidate: float(GraspGenXGraspPlanner._value(candidate, "score", 0.0) or 0.0),
        reverse=True,
    )
    sparse = [
        candidate for candidate in ordered
        if GraspGenXGraspPlanner._value(
            GraspGenXGraspPlanner._value(candidate, "metadata", {}) or {},
            "generation_branch",
        ) == "obb_sparse"
    ]
    if not sparse:
        return ordered[:maximum]
    reserved = sparse[:min(maximum, max(1, sparse_quota))]
    reserved_ids = {id(candidate) for candidate in reserved}
    remaining = [candidate for candidate in ordered if id(candidate) not in reserved_ids]
    return (reserved + remaining[:maximum - len(reserved)])[:maximum]
