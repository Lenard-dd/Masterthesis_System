"""Build conservative end-effector poses for a verified relative placement.

The spatial planner returns an object-centre target.  MoveIt, however, needs
an end-effector target.  This module preserves the measured translation from
the localized source centre to the reviewed grasp pose, while retaining the
reviewed grasp orientation.  It is planning-only and sends no robot command.
"""

from __future__ import annotations

from math import sqrt
from typing import Any


class PlaceMotionError(ValueError):
    """Raised when a relative object pose cannot safely become an EE pose."""


class PlaceMotionBuilder:
    def __init__(
        self,
        *,
        approach_height_m: float = 0.05,
        max_grasp_offset_m: float = 0.15,
        tool0_to_grasp_contact_m: tuple[float, float, float] = (0.0, 0.0, 0.199),
        inside_contact_clearance_above_rim_m: float = 0.04,
    ) -> None:
        if approach_height_m <= 0.0:
            raise ValueError("approach_height_m must be positive.")
        if max_grasp_offset_m <= 0.0:
            raise ValueError("max_grasp_offset_m must be positive.")
        self.approach_height_m = float(approach_height_m)
        self.max_grasp_offset_m = float(max_grasp_offset_m)
        self.tool0_to_grasp_contact_m = tuple(float(value) for value in tool0_to_grasp_contact_m)
        if len(self.tool0_to_grasp_contact_m) != 3:
            raise ValueError("tool0_to_grasp_contact_m must contain three values.")
        if inside_contact_clearance_above_rim_m < 0.0:
            raise ValueError("inside_contact_clearance_above_rim_m must not be negative.")
        self.inside_contact_clearance_above_rim_m = float(inside_contact_clearance_above_rim_m)

    def build(
        self,
        *,
        source_pose: dict[str, Any],
        target_pose: dict[str, Any],
        grasp_pose: dict[str, Any],
        placement_safety: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        source = self._object_pose(source_pose, "source_pose")
        target = self._object_pose(target_pose, "target_pose")
        grasp = self._ee_pose(grasp_pose)
        if source["frame"] != target["frame"] or source["frame"] != grasp["frame"]:
            raise PlaceMotionError("Source, target, and grasp poses must use the same frame.")
        offset = {
            axis: grasp["position"][axis] - source[axis]
            for axis in ("x", "y", "z")
        }
        offset_norm = sqrt(sum(value * value for value in offset.values()))
        if offset_norm > self.max_grasp_offset_m:
            raise PlaceMotionError(
                f"Reviewed grasp offset {offset_norm:.3f} m exceeds the {self.max_grasp_offset_m:.3f} m limit."
            )
        contact = {
            "frame": source["frame"],
            "position": {
                axis: target[axis] + offset[axis]
                for axis in ("x", "y", "z")
            },
            "orientation": dict(grasp["orientation"]),
        }
        if placement_safety:
            mode = placement_safety.get("mode")
            if mode != "above_opening_drop" or placement_safety.get("gripper_enters_container") is not False:
                raise PlaceMotionError("Only non-inserting above-opening inside placement is supported.")
            try:
                minimum_object_bottom = float(placement_safety["minimum_object_bottom_z_m"])
                target_object_bottom = target["z"] - float(placement_safety["source_height_m"]) / 2.0
            except (KeyError, TypeError, ValueError) as exc:
                raise PlaceMotionError("Inside placement safety metadata is incomplete.") from exc
            if target_object_bottom + 1e-6 < minimum_object_bottom:
                raise PlaceMotionError("Inside release pose would put the object below its safe rim clearance.")
            try:
                rim_z = float(placement_safety["rim_z_m"])
            except (KeyError, TypeError, ValueError) as exc:
                raise PlaceMotionError("Inside placement safety metadata has no valid rim height.") from exc

            # Checking only the object centre is insufficient for a small box:
            # the Robotiq fingers can still straddle the rim while opening.  Use
            # the calibrated tool0-to-contact transform and lift the complete
            # release pose until the grasp/contact region clears the rim.
            contact_point = self._local_point_in_pose(contact, self.tool0_to_grasp_contact_m)
            minimum_contact_z = rim_z + self.inside_contact_clearance_above_rim_m
            upward_adjustment = max(0.0, minimum_contact_z - contact_point["z"])
            contact["position"]["z"] += upward_adjustment
            placement_safety = {
                **placement_safety,
                "minimum_grasp_contact_z_m": minimum_contact_z,
                "grasp_contact_clearance_above_rim_m": self.inside_contact_clearance_above_rim_m,
                "release_pose_upward_adjustment_m": upward_adjustment,
                "release_grasp_contact_z_m": contact_point["z"] + upward_adjustment,
            }
        approach = self._vertical_offset(contact, self.approach_height_m)
        retreat = self._vertical_offset(contact, self.approach_height_m)
        return {
            "status": "SUCCESS",
            "message": "Built plan-only place approach, contact, and retreat end-effector poses.",
            "source_to_gripper_offset_m": offset,
            "source_to_gripper_offset_norm_m": offset_norm,
            "approach_distance_m": self.approach_height_m,
            "place_approach_pose": approach,
            "place_contact_pose": contact,
            "place_retreat_pose": retreat,
            "requires_confirmation": True,
            **({"inside_safety": dict(placement_safety)} if placement_safety else {}),
        }

    @staticmethod
    def _vertical_offset(pose: dict[str, Any], distance: float) -> dict[str, Any]:
        result = {
            "frame": pose["frame"],
            "position": dict(pose["position"]),
            "orientation": dict(pose["orientation"]),
        }
        result["position"]["z"] += distance
        return result

    @staticmethod
    def _local_point_in_pose(
        pose: dict[str, Any], local_point: tuple[float, float, float]
    ) -> dict[str, float]:
        q = pose["orientation"]
        x, y, z, w = (float(q[axis]) for axis in ("x", "y", "z", "w"))
        norm = sqrt(x * x + y * y + z * z + w * w)
        if norm <= 1e-9:
            raise PlaceMotionError("Grasp orientation quaternion has zero norm.")
        x, y, z, w = x / norm, y / norm, z / norm, w / norm
        rotation = (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        )
        rotated = tuple(
            sum(rotation[row][column] * local_point[column] for column in range(3))
            for row in range(3)
        )
        return {
            axis: float(pose["position"][axis]) + rotated[index]
            for index, axis in enumerate(("x", "y", "z"))
        }

    @staticmethod
    def _object_pose(value: dict[str, Any], name: str) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise PlaceMotionError(f"{name} is required.")
        frame = value.get("frame")
        if not isinstance(frame, str) or not frame:
            raise PlaceMotionError(f"{name}.frame is required.")
        try:
            coordinates = {axis: float(value[axis]) for axis in ("x", "y", "z")}
        except (KeyError, TypeError, ValueError) as exc:
            raise PlaceMotionError(f"{name} must contain finite x/y/z coordinates.") from exc
        return {"frame": frame, **coordinates}

    @staticmethod
    def _ee_pose(value: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise PlaceMotionError("A reviewed grasp end-effector pose is required.")
        frame, position, orientation = value.get("frame"), value.get("position"), value.get("orientation")
        if not isinstance(frame, str) or not isinstance(position, dict) or not isinstance(orientation, dict):
            raise PlaceMotionError("Grasp pose must contain frame, position, and orientation.")
        try:
            return {
                "frame": frame,
                "position": {axis: float(position[axis]) for axis in ("x", "y", "z")},
                "orientation": {axis: float(orientation[axis]) for axis in ("x", "y", "z", "w")},
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise PlaceMotionError("Grasp pose has invalid position or orientation values.") from exc
