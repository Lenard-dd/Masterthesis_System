"""Deterministic relative-placement pose planning from localized objects.

An LLM may choose a symbolic relation (for example ``between``), but this
module owns all metric computation.  It deliberately produces a plan-only
pose and never sends a robot command.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from llm_skill_robot.perception.scene_model import ObjectPose, SceneModel, SceneObject


SpatialRelation = Literal["between", "right_of", "left_of", "on_top_of"]
SUPPORTED_RELATIONS = {"between", "right_of", "left_of", "on_top_of"}


class SpatialReasoningError(ValueError):
    """Raised when a relative placement cannot be computed safely."""


@dataclass(frozen=True)
class ObjectDimensions:
    x: float
    y: float
    z: float
    estimated: bool


class RelativePlacePosePlanner:
    """Compute conservative placement centres in a verified common frame."""

    def __init__(
        self,
        *,
        default_object_size_m: float = 0.04,
        horizontal_clearance_m: float = 0.03,
        vertical_clearance_m: float = 0.003,
        right_axis: str = "x",
    ) -> None:
        if default_object_size_m <= 0.0:
            raise ValueError("default_object_size_m must be positive.")
        if horizontal_clearance_m < 0.0 or vertical_clearance_m < 0.0:
            raise ValueError("Clearances must not be negative.")
        if right_axis not in {"x", "y"}:
            raise ValueError("right_axis must be 'x' or 'y'.")
        self.default_object_size_m = float(default_object_size_m)
        self.horizontal_clearance_m = float(horizontal_clearance_m)
        self.vertical_clearance_m = float(vertical_clearance_m)
        self.right_axis = right_axis

    def compute(
        self,
        scene: SceneModel | dict[str, Any],
        *,
        source_id: str,
        relation: SpatialRelation | str,
        reference_ids: list[str],
    ) -> dict[str, Any]:
        """Return a plan-only target pose for a source object.

        ``right_of`` uses positive ``right_axis`` in ``base_link``.  Deployments
        must calibrate that convention once; the returned assumptions preserve
        it for review.
        """
        scene_model = self._scene(scene)
        normalized_relation = str(relation).strip().lower()
        if normalized_relation not in SUPPORTED_RELATIONS:
            raise SpatialReasoningError(
                f"Unsupported spatial relation {relation!r}; expected one of "
                f"{sorted(SUPPORTED_RELATIONS)}."
            )
        source = self._resolve(scene_model, source_id)
        references = [self._resolve(scene_model, reference_id) for reference_id in reference_ids]
        self._validate_reference_count(normalized_relation, references)
        source_pose = self._pose(source)
        reference_poses = [self._pose(reference) for reference in references]
        frame = self._common_frame([source_pose, *reference_poses])
        source_size = self._dimensions(source)
        reference_sizes = [self._dimensions(reference) for reference in references]

        if normalized_relation == "between":
            target = self._between(source_pose, source_size, reference_poses, reference_sizes)
        elif normalized_relation in {"right_of", "left_of"}:
            target = self._side_of(
                source_pose,
                source_size,
                reference_poses[0],
                reference_sizes[0],
                direction=1.0 if normalized_relation == "right_of" else -1.0,
            )
        else:
            target = self._on_top_of(source_pose, source_size, reference_poses[0], reference_sizes[0])

        estimated_dimensions = source_size.estimated or any(size.estimated for size in reference_sizes)
        return {
            "status": "SUCCESS",
            "message": f"Computed a plan-only {normalized_relation} placement pose.",
            "source_object_id": source.object_id,
            "reference_object_ids": [reference.object_id for reference in references],
            "relation": normalized_relation,
            "source_pose": source_pose.model_dump(),
            "reference_poses": [pose.model_dump() for pose in reference_poses],
            "target_pose": target.model_dump(),
            "requires_confirmation": True,
            "assumptions": {
                "right_axis": f"+{self.right_axis}",
                "horizontal_clearance_m": self.horizontal_clearance_m,
                "vertical_clearance_m": self.vertical_clearance_m,
                "object_dimensions_estimated": estimated_dimensions,
                "target_pose_is_plan_only": True,
            },
        }

    def _between(
        self,
        source_pose: ObjectPose,
        source_size: ObjectDimensions,
        references: list[ObjectPose],
        reference_sizes: list[ObjectDimensions],
    ) -> ObjectPose:
        first, second = references
        table_z = min(
            source_pose.z - source_size.z / 2.0,
            first.z - reference_sizes[0].z / 2.0,
            second.z - reference_sizes[1].z / 2.0,
        )
        return ObjectPose(
            frame=source_pose.frame,
            x=(first.x + second.x) / 2.0,
            y=(first.y + second.y) / 2.0,
            z=table_z + source_size.z / 2.0 + self.vertical_clearance_m,
        )

    def _side_of(
        self,
        source_pose: ObjectPose,
        source_size: ObjectDimensions,
        reference_pose: ObjectPose,
        reference_size: ObjectDimensions,
        *,
        direction: float,
    ) -> ObjectPose:
        table_z = min(
            source_pose.z - source_size.z / 2.0,
            reference_pose.z - reference_size.z / 2.0,
        )
        source_extent = source_size.x if self.right_axis == "x" else source_size.y
        reference_extent = reference_size.x if self.right_axis == "x" else reference_size.y
        offset = direction * (source_extent + reference_extent) / 2.0
        offset += direction * self.horizontal_clearance_m
        x, y = reference_pose.x, reference_pose.y
        if self.right_axis == "x":
            x += offset
        else:
            y += offset
        return ObjectPose(
            frame=source_pose.frame,
            x=x,
            y=y,
            z=table_z + source_size.z / 2.0 + self.vertical_clearance_m,
        )

    def _on_top_of(
        self,
        source_pose: ObjectPose,
        source_size: ObjectDimensions,
        reference_pose: ObjectPose,
        reference_size: ObjectDimensions,
    ) -> ObjectPose:
        return ObjectPose(
            frame=source_pose.frame,
            x=reference_pose.x,
            y=reference_pose.y,
            z=(
                reference_pose.z
                + reference_size.z / 2.0
                + source_size.z / 2.0
                + self.vertical_clearance_m
            ),
        )

    @staticmethod
    def _scene(scene: SceneModel | dict[str, Any]) -> SceneModel:
        return scene if isinstance(scene, SceneModel) else SceneModel.model_validate(scene)

    @staticmethod
    def _validate_reference_count(relation: str, references: list[SceneObject]) -> None:
        expected = 2 if relation == "between" else 1
        if len(references) != expected:
            raise SpatialReasoningError(
                f"Relation {relation!r} requires exactly {expected} reference object(s)."
            )

    @staticmethod
    def _resolve(scene: SceneModel, identifier: str) -> SceneObject:
        resolved = scene.find_by_id(identifier)
        if resolved is None:
            matches = scene.find_by_label(identifier)
            if len(matches) == 1:
                resolved = matches[0]
            elif len(matches) > 1:
                raise SpatialReasoningError(
                    f"Object label {identifier!r} is ambiguous; use its object_id."
                )
        if resolved is None:
            raise SpatialReasoningError(f"Localized object {identifier!r} is not in the current scene.")
        return resolved

    @staticmethod
    def _pose(scene_object: SceneObject) -> ObjectPose:
        if not scene_object.pose_available or scene_object.pose is None:
            raise SpatialReasoningError(
                f"Object {scene_object.object_id!r} has no verified spatial pose."
            )
        return scene_object.pose

    @staticmethod
    def _common_frame(poses: list[ObjectPose]) -> str:
        frames = {pose.frame for pose in poses}
        if len(frames) != 1:
            raise SpatialReasoningError(
                f"Objects are not expressed in a common frame: {sorted(frames)}."
            )
        return poses[0].frame

    def _dimensions(self, scene_object: SceneObject) -> ObjectDimensions:
        raw = scene_object.metadata.get("dimensions_m")
        if isinstance(raw, dict):
            values = [raw.get(axis) for axis in ("x", "y", "z")]
        else:
            values = raw if isinstance(raw, (list, tuple)) else []
        if len(values) >= 3:
            try:
                dimensions = [float(value) for value in values[:3]]
            except (TypeError, ValueError):
                dimensions = []
            if all(value > 0.0 for value in dimensions):
                height_estimation = scene_object.metadata.get("height_estimation", {})
                measured_from_point_cloud = isinstance(height_estimation, dict) and bool(
                    height_estimation.get("usable", False)
                )
                return ObjectDimensions(*dimensions, estimated=measured_from_point_cloud)
        footprint = scene_object.metadata.get("footprint_dimensions_m")
        if isinstance(footprint, dict):
            values = [footprint.get(axis) for axis in ("x", "y")]
        else:
            values = footprint if isinstance(footprint, (list, tuple)) else []
        if len(values) >= 2:
            try:
                x, y = (float(values[0]), float(values[1]))
            except (TypeError, ValueError):
                x = y = 0.0
            if x > 0.0 and y > 0.0:
                # A single masked RGB-D view measures the footprint well but
                # generally cannot observe object thickness.  Keep the height
                # conservative until a table-plane/multi-view estimator exists.
                return ObjectDimensions(x, y, self.default_object_size_m, estimated=True)
        default = self.default_object_size_m
        return ObjectDimensions(default, default, default, estimated=True)
