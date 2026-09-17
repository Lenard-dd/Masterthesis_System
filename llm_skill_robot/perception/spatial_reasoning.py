"""Deterministic relative-placement pose planning from localized objects.

An LLM may choose a symbolic relation (for example ``between``), but this
module owns all metric computation.  It deliberately produces a plan-only
pose and never sends a robot command.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from llm_skill_robot.perception.scene_model import ObjectPose, SceneModel, SceneObject
from llm_skill_robot.perception.spatial_constraints import (
    AxisAssignmentSolver,
    SpatialConstraintCompiler,
)


SpatialRelation = Literal["between", "right_of", "left_of", "on_top_of", "inside"]
SUPPORTED_RELATIONS = {"between", "right_of", "left_of", "on_top_of", "inside"}


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
        inside_wall_estimate_m: float = 0.008,
        inside_object_clearance_m: float = 0.006,
        inside_release_clearance_m: float = 0.040,
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
        self.inside_wall_estimate_m = float(inside_wall_estimate_m)
        self.inside_object_clearance_m = float(inside_object_clearance_m)
        self.inside_release_clearance_m = float(inside_release_clearance_m)

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
        elif normalized_relation == "on_top_of":
            target = self._on_top_of(source_pose, source_size, reference_poses[0], reference_sizes[0])
        else:
            target, inside_safety = self._inside_above_opening(
                source_pose, source_size, references[0], reference_poses[0], reference_sizes[0],
            )

        estimated_dimensions = source_size.estimated or any(size.estimated for size in reference_sizes)
        constraint_set = SpatialConstraintCompiler(
            right_axis=self.right_axis,
            horizontal_clearance_m=self.horizontal_clearance_m,
        ).compile(normalized_relation, [reference.object_id for reference in references])
        return {
            "status": "SUCCESS",
            "message": f"Computed a plan-only {normalized_relation} placement pose.",
            "source_object_id": source.object_id,
            "source_label": source.label,
            "reference_object_ids": [reference.object_id for reference in references],
            "relation": normalized_relation,
            "source_pose": source_pose.model_dump(),
            "reference_poses": [pose.model_dump() for pose in reference_poses],
            "target_pose": target.model_dump(),
            "requires_confirmation": True,
            "geometry_constraints": constraint_set.as_dict(),
            **({"inside_safety": inside_safety, "placement_mode": "above_opening_drop"}
               if normalized_relation == "inside" else {}),
            "assumptions": {
                "right_axis": f"+{self.right_axis}",
                "horizontal_clearance_m": self.horizontal_clearance_m,
                "vertical_clearance_m": self.vertical_clearance_m,
                "object_dimensions_estimated": estimated_dimensions,
                "target_pose_is_plan_only": True,
            },
        }

    def _inside_above_opening(
        self, source_pose: ObjectPose, source_size: ObjectDimensions,
        container: SceneObject, container_pose: ObjectPose, container_size: ObjectDimensions,
    ) -> tuple[ObjectPose, dict[str, Any]]:
        """Plan a low drop above the opening; the gripper never enters the box."""
        raw_opening = container.metadata.get("container_opening_dimensions_m")
        measured = isinstance(raw_opening, dict) and all(
            raw_opening.get(axis) is not None for axis in ("x", "y")
        )
        if measured:
            opening_x = float(raw_opening["x"])
            opening_y = float(raw_opening["y"])
        else:
            opening_x = container_size.x - 2.0 * self.inside_wall_estimate_m
            opening_y = container_size.y - 2.0 * self.inside_wall_estimate_m
        required_x = source_size.x + 2.0 * self.inside_object_clearance_m
        required_y = source_size.y + 2.0 * self.inside_object_clearance_m
        if opening_x <= 0.0 or opening_y <= 0.0:
            raise SpatialReasoningError("Container opening dimensions are unavailable or invalid.")
        if required_x > opening_x or required_y > opening_y:
            raise SpatialReasoningError(
                "Source object plus inside-placement clearance does not fit through the container opening "
                f"(required {required_x:.3f} x {required_y:.3f} m, available "
                f"{opening_x:.3f} x {opening_y:.3f} m)."
            )
        rim_z = container_pose.z + container_size.z / 2.0
        target = ObjectPose(
            frame=source_pose.frame,
            x=container_pose.x,
            y=container_pose.y,
            z=rim_z + source_size.z / 2.0 + self.inside_release_clearance_m,
        )
        return target, {
            "mode": "above_opening_drop",
            "gripper_enters_container": False,
            "opening_dimensions_m": {"x": opening_x, "y": opening_y},
            "opening_dimensions_measured": measured,
            "required_object_envelope_m": {"x": required_x, "y": required_y},
            "release_clearance_above_rim_m": self.inside_release_clearance_m,
            "rim_z_m": rim_z,
            "minimum_object_bottom_z_m": rim_z + self.inside_release_clearance_m,
            "source_height_m": source_size.z,
            "approach": "vertical_along_opening_normal",
            "retreat": "vertical",
        }

    def compute_composite(
        self,
        scene: SceneModel | dict[str, Any],
        *,
        source_id: str,
        relation_specs: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Solve compatible relations as one deterministic placement pose.

        Each relation contributes only the axes it owns. Conflicting axis
        assignments fail closed instead of silently choosing one relation.
        """
        if not relation_specs:
            raise SpatialReasoningError("At least one relation constraint is required.")
        scene_model = self._scene(scene)
        source = self._resolve(scene_model, source_id)
        source_pose = self._pose(source)
        assignments: list[dict[str, float]] = []
        compiled: list[dict[str, Any]] = []
        all_reference_ids: list[str] = []
        for spec in relation_specs:
            relation = str(spec.get("relation", "")).strip().casefold()
            reference_ids = [str(item) for item in spec.get("reference_ids", [])]
            constraint_set = SpatialConstraintCompiler(
                right_axis=self.right_axis,
                horizontal_clearance_m=self.horizontal_clearance_m,
            ).compile(relation, reference_ids)
            compiled.append(constraint_set.as_dict())
            all_reference_ids.extend(reference_ids)
            if relation in SUPPORTED_RELATIONS:
                single = self.compute(
                    scene_model, source_id=source_id, relation=relation,
                    reference_ids=reference_ids,
                )["target_pose"]
                axes = {
                    "between": ("x", "y", "z"),
                    "right_of": (self.right_axis, "z"),
                    "left_of": (self.right_axis, "z"),
                    "on_top_of": ("x", "y", "z"),
                    "inside": ("x", "y", "z"),
                }[relation]
                assignments.append({axis: float(single[axis]) for axis in axes})
            else:
                reference = self._resolve(scene_model, reference_ids[0])
                reference_pose = self._pose(reference)
                axis = {
                    "aligned_x_with": "x", "aligned_y_with": "y", "same_height_as": "z",
                }[relation]
                assignments.append({axis: float(getattr(reference_pose, axis))})
        merged = AxisAssignmentSolver().merge(
            {"x": source_pose.x, "y": source_pose.y, "z": source_pose.z}, assignments,
        )
        target = ObjectPose(frame=source_pose.frame, **merged)
        return {
            "status": "SUCCESS",
            "message": "Computed one plan-only pose satisfying the combined constraints.",
            "source_object_id": source.object_id,
            "source_label": source.label,
            "reference_object_ids": list(dict.fromkeys(all_reference_ids)),
            "relation": "composite",
            "relation_specs": relation_specs,
            "geometry_constraints": compiled,
            "source_pose": source_pose.model_dump(),
            "target_pose": target.model_dump(),
            "requires_confirmation": True,
            "assumptions": {"target_pose_is_plan_only": True, "constraint_solver": "axis_assignment_v1"},
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
