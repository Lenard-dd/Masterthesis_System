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
    def __init__(self, *, approach_height_m: float = 0.05, max_grasp_offset_m: float = 0.15) -> None:
        if approach_height_m <= 0.0:
            raise ValueError("approach_height_m must be positive.")
        if max_grasp_offset_m <= 0.0:
            raise ValueError("max_grasp_offset_m must be positive.")
        self.approach_height_m = float(approach_height_m)
        self.max_grasp_offset_m = float(max_grasp_offset_m)

    def build(
        self,
        *,
        source_pose: dict[str, Any],
        target_pose: dict[str, Any],
        grasp_pose: dict[str, Any],
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
