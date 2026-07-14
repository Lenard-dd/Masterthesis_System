"""Independent tabletop-clearance checks for real grasp execution.

This module deliberately does not publish a PlanningScene object or modify a
URDF. MoveIt still checks the full trajectory against its collision scene,
while this check rejects TCP targets that are too close to the measured table.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


def _as_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric.") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


@dataclass(frozen=True)
class TabletopSafety:
    """A calibrated tabletop plane and gripper-contact clearance policy."""

    enabled: bool = False
    frame: str = "base_link"
    normal: tuple[float, float, float] = (0.0, 0.0, 1.0)
    d: float = 0.0
    collision_margin_m: float = 0.0
    min_pregrasp_clearance_m: float = 0.0
    min_grasp_clearance_m: float = 0.0
    min_retreat_clearance_m: float = 0.0
    close_sweep_compensation_m: float = 0.0
    max_approach_tilt_deg: float = 90.0
    tool0_to_grasp_contact_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    grasp_contact_backoff_m: float = 0.0
    finger_axial_close_sweep_m: float = 0.0

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> "TabletopSafety":
        values = (config or {}).get("tabletop_safety", config or {})
        if not isinstance(values, dict):
            raise ValueError("tabletop_safety configuration must be a mapping.")
        normal_cfg = values.get("plane", {}).get("normal", values.get("normal", {}))
        if not isinstance(normal_cfg, dict):
            raise ValueError("tabletop_safety.plane.normal must be a mapping.")
        normal = tuple(
            _as_float(normal_cfg.get(axis), f"plane.normal.{axis}")
            for axis in ("x", "y", "z")
        )
        length = math.sqrt(sum(component * component for component in normal))
        if length <= 1e-9:
            raise ValueError("tabletop_safety plane normal must not be zero.")
        normal = tuple(component / length for component in normal)
        plane = values.get("plane", {})
        d = _as_float(plane.get("d", values.get("d", 0.0)), "plane.d") / length
        clearance = values.get("clearance", {})
        if not isinstance(clearance, dict):
            raise ValueError("tabletop_safety.clearance must be a mapping.")
        contact_offset = values.get("tool0_to_grasp_contact_m", [0.0, 0.0, 0.0])
        if not isinstance(contact_offset, (list, tuple)) or len(contact_offset) != 3:
            raise ValueError("tool0_to_grasp_contact_m must contain three values.")
        return cls(
            enabled=bool(values.get("enabled", False)),
            frame=str(values.get("frame", "base_link")),
            normal=normal,
            d=d,
            collision_margin_m=_as_float(
                values.get("collision_margin_m", 0.0), "collision_margin_m"
            ),
            min_pregrasp_clearance_m=_as_float(
                clearance.get("pregrasp_m", 0.0), "clearance.pregrasp_m"
            ),
            min_grasp_clearance_m=_as_float(
                clearance.get("grasp_m", 0.0), "clearance.grasp_m"
            ),
            min_retreat_clearance_m=_as_float(
                clearance.get("retreat_m", 0.0), "clearance.retreat_m"
            ),
            close_sweep_compensation_m=_as_float(
                values.get("close_sweep_compensation_m", 0.0),
                "close_sweep_compensation_m",
            ),
            max_approach_tilt_deg=_as_float(
                values.get("max_approach_tilt_deg", 90.0),
                "max_approach_tilt_deg",
            ),
            tool0_to_grasp_contact_m=tuple(
                _as_float(value, "tool0_to_grasp_contact_m")
                for value in contact_offset
            ),
            grasp_contact_backoff_m=_as_float(
                values.get("grasp_contact_backoff_m", 0.0),
                "grasp_contact_backoff_m",
            ),
            finger_axial_close_sweep_m=_as_float(
                values.get("finger_axial_close_sweep_m", 0.0),
                "finger_axial_close_sweep_m",
            ),
        )

    def apply_grasp_contact_backoff(
        self, poses: dict[str, Any]
    ) -> dict[str, Any]:
        """Move final grasp and retreat back along the gripper approach axis.

        Pregrasp remains unchanged, so the supervised approach simply ends a
        little earlier on the same approach line. Retreat receives the same
        translation to preserve its displacement from the adjusted final grasp.
        """
        if self.grasp_contact_backoff_m < 0.0:
            raise ValueError("grasp_contact_backoff_m must be non-negative.")
        if self.grasp_contact_backoff_m <= 0.0:
            return poses
        pregrasp = poses.get("pregrasp")
        grasp = poses.get("grasp")
        if not all(
            isinstance(pose, dict) and isinstance(pose.get("position"), dict)
            for pose in (pregrasp, grasp)
        ):
            raise ValueError(
                "grasp contact backoff requires valid pregrasp and grasp positions."
            )
        vector = tuple(
            _as_float(pregrasp["position"].get(axis), f"pregrasp.position.{axis}")
            - _as_float(grasp["position"].get(axis), f"grasp.position.{axis}")
            for axis in ("x", "y", "z")
        )
        length = math.sqrt(sum(component * component for component in vector))
        if length <= 1e-9:
            raise ValueError("pregrasp and grasp positions must differ.")
        offset = tuple(
            component / length * self.grasp_contact_backoff_m
            for component in vector
        )
        adjusted = dict(poses)
        for name in ("grasp", "retreat"):
            pose = poses.get(name)
            if not isinstance(pose, dict) or not isinstance(pose.get("position"), dict):
                raise ValueError(
                    f"grasp contact backoff requires a valid {name} pose position."
                )
            position = dict(pose["position"])
            for index, axis in enumerate(("x", "y", "z")):
                position[axis] = (
                    _as_float(position.get(axis), f"{name}.position.{axis}")
                    + offset[index]
                )
            adjusted[name] = {**pose, "position": position}
        return adjusted

    def signed_distance_m(self, position: dict[str, Any]) -> float:
        point = tuple(
            _as_float(position.get(axis), f"position.{axis}")
            for axis in ("x", "y", "z")
        )
        return sum(
            component * coordinate
            for component, coordinate in zip(self.normal, point)
        ) + self.d

    def validate_pose(self, pose: dict[str, Any], stage: str) -> dict[str, Any]:
        if not self.enabled:
            return {"allowed": True, "reason": "Tabletop safety is disabled."}
        frame = (
            pose.get("frame") or pose.get("frame_id")
            if isinstance(pose, dict)
            else None
        )
        if frame != self.frame:
            return {
                "allowed": False,
                "reason": f"Tabletop safety requires pose frame == {self.frame}.",
            }
        if not isinstance(pose.get("position"), dict):
            return {
                "allowed": False,
                "reason": "Tabletop safety requires a target position.",
            }
        try:
            contact_point = self.grasp_contact_point(pose)
            measured = self.signed_distance_m(contact_point)
        except ValueError as exc:
            return {"allowed": False, "reason": str(exc)}
        required = self._required_clearance(stage)
        if stage == "grasp":
            required += self.close_sweep_compensation_m
        if measured + 1e-9 < required:
            return {
                "allowed": False,
                "reason": (
                    f"{stage} gripper contact-point plane clearance {measured:.4f} m "
                    f"is below the "
                    f"required conservative clearance {required:.4f} m."
                ),
                "measured_clearance_m": measured,
                "required_clearance_m": required,
                "grasp_contact_point": contact_point,
            }
        return {
            "allowed": True,
            "measured_clearance_m": measured,
            "required_clearance_m": required,
            "grasp_contact_point": contact_point,
        }

    def grasp_contact_point(self, pose: dict[str, Any]) -> dict[str, float]:
        """Transform the configured tool0-local grasp point into pose frame."""
        position = pose.get("position")
        orientation = pose.get("orientation")
        if not isinstance(position, dict) or not isinstance(orientation, dict):
            raise ValueError(
                "Gripper contact-point safety requires position and orientation."
            )
        origin = tuple(
            _as_float(position.get(axis), f"position.{axis}")
            for axis in ("x", "y", "z")
        )
        rotation = self._quaternion_rotation(orientation)
        rotated_offset = tuple(
            sum(
                rotation[row][column] * self.tool0_to_grasp_contact_m[column]
                for column in range(3)
            )
            for row in range(3)
        )
        return {
            axis: origin[index] + rotated_offset[index]
            for index, axis in enumerate(("x", "y", "z"))
        }

    def validate_grasp_plan(
        self, plan: Any, *, motion_skill: str | None = None
    ) -> dict[str, Any]:
        if not self.enabled:
            return {"allowed": True, "reason": "Tabletop safety is disabled."}
        names = (
            ("pregrasp_pose", "pregrasp"),
            ("grasp_pose", "grasp"),
            ("retreat_pose", "retreat"),
        )
        results: dict[str, Any] = {}
        for key, stage in names:
            pose = plan.get(key) if isinstance(plan, dict) else getattr(plan, key, None)
            if pose is None:
                continue
            result = self.validate_pose(pose, stage)
            results[stage] = result
            if not result["allowed"]:
                return {
                    "allowed": False,
                    "reason": result["reason"],
                    "checks": results,
                }

        pregrasp = (
            plan.get("pregrasp_pose")
            if isinstance(plan, dict)
            else getattr(plan, "pregrasp_pose", None)
        )
        grasp = (
            plan.get("grasp_pose")
            if isinstance(plan, dict)
            else getattr(plan, "grasp_pose", None)
        )
        if pregrasp is not None and grasp is not None:
            direction = self._approach_direction(pregrasp, grasp)
            if direction is None:
                return {
                    "allowed": False,
                    "reason": "Tabletop safety cannot calculate pregrasp-to-grasp direction.",
                    "checks": results,
                }
            dot = max(
                -1.0,
                min(
                    1.0,
                    sum(a * b for a, b in zip(direction, self.normal)),
                ),
            )
            tilt_deg = math.degrees(math.acos(dot))
            if tilt_deg > self.max_approach_tilt_deg + 1e-9:
                return {
                    "allowed": False,
                    "reason": (
                        f"Approach tilt {tilt_deg:.1f} deg exceeds tabletop "
                        f"limit {self.max_approach_tilt_deg:.1f} deg."
                    ),
                    "checks": results,
                    "approach_tilt_deg": tilt_deg,
                }
            results["approach"] = {"allowed": True, "tilt_deg": tilt_deg}
            sweep_result = self._validate_finger_close_sweep(grasp, direction)
            results["finger_close_sweep"] = sweep_result
            if not sweep_result["allowed"]:
                return {
                    "allowed": False,
                    "reason": sweep_result["reason"],
                    "checks": results,
                }
        return {"allowed": True, "checks": results}

    def _validate_finger_close_sweep(
        self,
        grasp: dict[str, Any],
        backoff_direction: tuple[float, float, float],
    ) -> dict[str, Any]:
        """Check the fingertip endpoint after closing along the approach axis."""
        if self.finger_axial_close_sweep_m < 0.0:
            return {
                "allowed": False,
                "reason": "finger_axial_close_sweep_m must be non-negative.",
            }
        try:
            open_point = self.grasp_contact_point(grasp)
            closed_point = {
                axis: open_point[axis]
                - backoff_direction[index] * self.finger_axial_close_sweep_m
                for index, axis in enumerate(("x", "y", "z"))
            }
            measured = self.signed_distance_m(closed_point)
        except ValueError as exc:
            return {"allowed": False, "reason": str(exc)}
        required = self._required_clearance("grasp") + self.close_sweep_compensation_m
        if measured + 1e-9 < required:
            return {
                "allowed": False,
                "reason": (
                    "Closed-finger sweep endpoint plane clearance "
                    f"{measured:.4f} m is below the required conservative "
                    f"clearance {required:.4f} m."
                ),
                "measured_clearance_m": measured,
                "required_clearance_m": required,
                "closed_contact_point": closed_point,
            }
        return {
            "allowed": True,
            "measured_clearance_m": measured,
            "required_clearance_m": required,
            "closed_contact_point": closed_point,
        }

    def _required_clearance(self, stage: str) -> float:
        requested = {
            "pregrasp": self.min_pregrasp_clearance_m,
            "grasp": self.min_grasp_clearance_m,
            "retreat": self.min_retreat_clearance_m,
        }.get(stage, self.min_grasp_clearance_m)
        return self.collision_margin_m + requested

    @staticmethod
    def _quaternion_rotation(
        quaternion: dict[str, Any],
    ) -> tuple[tuple[float, float, float], ...]:
        x, y, z, w = (
            _as_float(quaternion.get(axis), f"orientation.{axis}")
            for axis in ("x", "y", "z", "w")
        )
        length = math.sqrt(x * x + y * y + z * z + w * w)
        if length <= 1e-9:
            raise ValueError("orientation quaternion must not be zero.")
        x, y, z, w = x / length, y / length, z / length, w / length
        return (
            (
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ),
            (
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ),
            (
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ),
        )

    @staticmethod
    def _approach_direction(
        pregrasp: dict[str, Any], grasp: dict[str, Any]
    ) -> tuple[float, float, float] | None:
        try:
            vector = tuple(
                _as_float(
                    pregrasp["position"][axis], f"pregrasp.position.{axis}"
                )
                - _as_float(grasp["position"][axis], f"grasp.position.{axis}")
                for axis in ("x", "y", "z")
            )
        except (KeyError, TypeError, ValueError):
            return None
        length = math.sqrt(sum(component * component for component in vector))
        if length <= 1e-9:
            return None
        return tuple(component / length for component in vector)
