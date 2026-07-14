"""Independent tabletop-clearance checks for real grasp execution.

This module deliberately does not publish a PlanningScene object or modify a
URDF.  It is an extra, calibrated guard around the shared MoveIt table model:
MoveIt still checks the full trajectory against its collision scene, while this
check rejects grasp targets that are too close to the *measured* tabletop.
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
    """A calibrated tabletop plane and conservative grasp clearance policy."""

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
    finger_envelope_enabled: bool = False
    tool0_to_gripper_base_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    finger_joint_min_rad: float = 0.0
    finger_joint_max_rad: float = 0.7
    finger_joint_samples: int = 15
    finger_extra_clearance_m: float = 0.0

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> "TabletopSafety":
        """Build a normalized policy from ``config/tabletop_safety.yaml``."""
        values = (config or {}).get("tabletop_safety", config or {})
        if not isinstance(values, dict):
            raise ValueError("tabletop_safety configuration must be a mapping.")
        normal_cfg = values.get("plane", {}).get("normal", values.get("normal", {}))
        if not isinstance(normal_cfg, dict):
            raise ValueError("tabletop_safety.plane.normal must be a mapping.")
        normal = tuple(_as_float(normal_cfg.get(axis), f"plane.normal.{axis}") for axis in ("x", "y", "z"))
        length = math.sqrt(sum(component * component for component in normal))
        if length <= 1e-9:
            raise ValueError("tabletop_safety plane normal must not be zero.")
        normal = tuple(component / length for component in normal)
        plane = values.get("plane", {})
        d = _as_float(plane.get("d", values.get("d", 0.0)), "plane.d") / length
        clearance = values.get("clearance", {})
        if not isinstance(clearance, dict):
            raise ValueError("tabletop_safety.clearance must be a mapping.")
        envelope = values.get("robotiq_2f_140_finger_envelope", {})
        if not isinstance(envelope, dict):
            raise ValueError("tabletop_safety.robotiq_2f_140_finger_envelope must be a mapping.")
        base_offset = envelope.get("tool0_to_gripper_base_m", [0.0, 0.0, 0.0])
        if not isinstance(base_offset, (list, tuple)) or len(base_offset) != 3:
            raise ValueError("finger-envelope tool0_to_gripper_base_m must contain three values.")
        samples = int(envelope.get("joint_samples", 15))
        if samples < 2:
            raise ValueError("finger-envelope joint_samples must be at least two.")
        return cls(
            enabled=bool(values.get("enabled", False)),
            frame=str(values.get("frame", "base_link")),
            normal=normal,
            d=d,
            collision_margin_m=_as_float(values.get("collision_margin_m", 0.0), "collision_margin_m"),
            min_pregrasp_clearance_m=_as_float(clearance.get("pregrasp_m", 0.0), "clearance.pregrasp_m"),
            min_grasp_clearance_m=_as_float(clearance.get("grasp_m", 0.0), "clearance.grasp_m"),
            min_retreat_clearance_m=_as_float(clearance.get("retreat_m", 0.0), "clearance.retreat_m"),
            close_sweep_compensation_m=_as_float(values.get("close_sweep_compensation_m", 0.0), "close_sweep_compensation_m"),
            max_approach_tilt_deg=_as_float(values.get("max_approach_tilt_deg", 90.0), "max_approach_tilt_deg"),
            finger_envelope_enabled=bool(envelope.get("enabled", False)),
            tool0_to_gripper_base_m=tuple(_as_float(value, "finger-envelope tool0_to_gripper_base_m") for value in base_offset),
            finger_joint_min_rad=_as_float(envelope.get("joint_min_rad", 0.0), "finger-envelope joint_min_rad"),
            finger_joint_max_rad=_as_float(envelope.get("joint_max_rad", 0.7), "finger-envelope joint_max_rad"),
            finger_joint_samples=samples,
            finger_extra_clearance_m=_as_float(envelope.get("extra_clearance_m", 0.0), "finger-envelope extra_clearance_m"),
        )

    def signed_distance_m(self, position: dict[str, Any]) -> float:
        """Return distance above the calibrated plane along its upward normal."""
        point = tuple(_as_float(position.get(axis), f"position.{axis}") for axis in ("x", "y", "z"))
        return sum(component * coordinate for component, coordinate in zip(self.normal, point)) + self.d

    def validate_pose(self, pose: dict[str, Any], stage: str) -> dict[str, Any]:
        if not self.enabled:
            return {"allowed": True, "reason": "Tabletop safety is disabled."}
        frame = pose.get("frame") or pose.get("frame_id") if isinstance(pose, dict) else None
        if frame != self.frame:
            return {"allowed": False, "reason": f"Tabletop safety requires pose frame == {self.frame}."}
        if not isinstance(pose.get("position"), dict):
            return {"allowed": False, "reason": "Tabletop safety requires a target position."}
        try:
            measured = self.signed_distance_m(pose["position"])
        except ValueError as exc:
            return {"allowed": False, "reason": str(exc)}
        required = self._required_clearance(stage)
        if stage == "grasp":
            required += self.close_sweep_compensation_m
        if measured + 1e-9 < required:
            return {
                "allowed": False,
                "reason": (
                    f"{stage} TCP plane clearance {measured:.4f} m is below the "
                    f"required conservative clearance {required:.4f} m."
                ),
                "measured_clearance_m": measured,
                "required_clearance_m": required,
            }
        finger_check = self._validate_finger_envelope(pose, stage)
        if not finger_check["allowed"]:
            return finger_check
        return {
            "allowed": True,
            "measured_clearance_m": measured,
            "required_clearance_m": required,
            "finger_envelope": finger_check,
        }

    def validate_grasp_plan(self, plan: Any, *, motion_skill: str | None = None) -> dict[str, Any]:
        """Check all available targets and the supervised approach direction.

        The target is the configured MoveIt TCP, not a claim about every finger
        mesh vertex.  The added close-sweep compensation makes the grasp TCP
        policy deliberately higher than the open-gripper target policy; MoveIt
        remains responsible for full link/trajectory collision checking.
        """
        if not self.enabled:
            return {"allowed": True, "reason": "Tabletop safety is disabled."}
        names = (("pregrasp_pose", "pregrasp"), ("grasp_pose", "grasp"), ("retreat_pose", "retreat"))
        results: dict[str, Any] = {}
        for key, stage in names:
            pose = plan.get(key) if isinstance(plan, dict) else getattr(plan, key, None)
            if pose is None:
                continue
            result = self.validate_pose(pose, stage)
            results[stage] = result
            if not result["allowed"]:
                return {"allowed": False, "reason": result["reason"], "checks": results}

        # A straight, near-normal approach avoids dragging long open fingers
        # sideways across the table.  Only apply this to a plan containing both
        # poses; named targets and standalone lifts are not constrained here.
        pregrasp = plan.get("pregrasp_pose") if isinstance(plan, dict) else getattr(plan, "pregrasp_pose", None)
        grasp = plan.get("grasp_pose") if isinstance(plan, dict) else getattr(plan, "grasp_pose", None)
        if pregrasp is not None and grasp is not None:
            direction = self._approach_direction(pregrasp, grasp)
            if direction is None:
                return {"allowed": False, "reason": "Tabletop safety cannot calculate pregrasp-to-grasp direction.", "checks": results}
            # Direction points from grasp toward pregrasp (the safe/up direction).
            dot = max(-1.0, min(1.0, sum(a * b for a, b in zip(direction, self.normal))))
            tilt_deg = math.degrees(math.acos(dot))
            if tilt_deg > self.max_approach_tilt_deg + 1e-9:
                return {
                    "allowed": False,
                    "reason": f"Approach tilt {tilt_deg:.1f} deg exceeds tabletop limit {self.max_approach_tilt_deg:.1f} deg.",
                    "checks": results,
                    "approach_tilt_deg": tilt_deg,
                }
            results["approach"] = {"allowed": True, "tilt_deg": tilt_deg}
        return {"allowed": True, "checks": results}

    def _required_clearance(self, stage: str) -> float:
        requested = {
            "pregrasp": self.min_pregrasp_clearance_m,
            "grasp": self.min_grasp_clearance_m,
            "retreat": self.min_retreat_clearance_m,
        }.get(stage, self.min_grasp_clearance_m)
        return self.collision_margin_m + requested

    def _validate_finger_envelope(self, pose: dict[str, Any], stage: str) -> dict[str, Any]:
        """Check Robotiq finger-pad swept points, not merely the TCP.

        The constants below are the 2F-140 pad kinematic chain from the shared
        Robotiq xacro.  They are used only as a conservative safety proxy; no
        robot description, RViz geometry, or MoveIt collision object is
        changed.  At the grasp pose we check every sampled joint angle between
        open and configured close.  This catches the observed downward sweep
        of both fingers while closing.
        """
        if not self.finger_envelope_enabled:
            return {"allowed": True, "enabled": False}
        orientation = pose.get("orientation")
        if not isinstance(orientation, dict):
            return {"allowed": False, "reason": "Finger-envelope safety requires target orientation."}
        try:
            rotation = self._quaternion_rotation(orientation)
            origin = tuple(_as_float(pose["position"][axis], f"position.{axis}") for axis in ("x", "y", "z"))
        except (KeyError, TypeError, ValueError) as exc:
            return {"allowed": False, "reason": f"Finger-envelope safety cannot transform target pose: {exc}"}
        # Pregrasp occurs open; retreat occurs closed.  During contact close,
        # all sampled angles must remain clear of the measured plane.
        if stage == "pregrasp":
            joint_values = (self.finger_joint_min_rad,)
        elif stage == "retreat":
            joint_values = (self.finger_joint_max_rad,)
        else:
            step = (self.finger_joint_max_rad - self.finger_joint_min_rad) / (self.finger_joint_samples - 1)
            joint_values = tuple(self.finger_joint_min_rad + index * step for index in range(self.finger_joint_samples))
        required = self._required_clearance(stage) + self.finger_extra_clearance_m
        lowest: tuple[float, str, float] | None = None
        for joint_angle in joint_values:
            for name, local_point in self._robotiq_pad_corners(joint_angle):
                rotated = self._rotate(rotation, local_point)
                point = tuple(origin[index] + rotated[index] for index in range(3))
                clearance = sum(component * coordinate for component, coordinate in zip(self.normal, point)) + self.d
                if lowest is None or clearance < lowest[0]:
                    lowest = (clearance, name, joint_angle)
        assert lowest is not None
        if lowest[0] + 1e-9 < required:
            return {
                "allowed": False,
                "reason": (
                    f"{stage} Robotiq finger envelope clearance {lowest[0]:.4f} m "
                    f"({lowest[1]}, joint={lowest[2]:.3f} rad) is below required {required:.4f} m."
                ),
                "measured_clearance_m": lowest[0],
                "required_clearance_m": required,
                "lowest_proxy": lowest[1],
                "joint_angle_rad": lowest[2],
            }
        return {
            "allowed": True,
            "enabled": True,
            "minimum_clearance_m": lowest[0],
            "required_clearance_m": required,
            "lowest_proxy": lowest[1],
            "joint_angle_rad": lowest[2],
        }

    def _robotiq_pad_corners(self, joint_angle: float) -> tuple[tuple[str, tuple[float, float, float]], ...]:
        """Return the eight collision-box corners of each 2F-140 finger pad.

        Coordinates are transformed into tool0 using the fixed mount offset.
        The collision pad is 0.03 x 0.07 x 0.0075 m in the Robotiq xacro.
        """
        half = (0.015, 0.035, 0.00375)
        corners = tuple((x, y, z) for x in (-half[0], half[0]) for y in (-half[1], half[1]) for z in (-half[2], half[2]))
        result: list[tuple[str, tuple[float, float, float]]] = []
        for side in ("left", "right"):
            transform = self._robotiq_pad_transform(joint_angle, side)
            for index, corner in enumerate(corners):
                point = self._transform_point(transform, corner)
                point = tuple(point[axis] + self.tool0_to_gripper_base_m[axis] for axis in range(3))
                result.append((f"{side}_pad_corner_{index}", point))
        return tuple(result)

    @staticmethod
    def _robotiq_pad_transform(q: float, side: str) -> tuple[tuple[float, ...], ...]:
        # Base -> outer knuckle -> outer finger -> inner finger -> pad.
        reflect = -1.0 if side == "left" else 1.0
        base_y = -0.030601 if side == "left" else 0.030601
        base_rotation = TabletopSafety._matmul3(
            TabletopSafety._rotation_x(math.pi / 2.0 + 0.725),
            TabletopSafety._rotation_z(0.0 if side == "left" else math.pi),
        )
        # The left finger joint uses axis -X; the right joint mimics -q about
        # +X.  Both therefore have the same effective -q rotation here.
        transform = TabletopSafety._transform((0.0, base_y, 0.054905), TabletopSafety._matmul3(base_rotation, TabletopSafety._rotation_x(-q)))
        transform = TabletopSafety._matmul(transform, TabletopSafety._transform((0.0, 0.01821998610742, 0.0260018192872234), TabletopSafety._identity_rotation()))
        transform = TabletopSafety._matmul(transform, TabletopSafety._transform((0.0, 0.0817554015893473, -0.0282203446692936), TabletopSafety._matmul3(TabletopSafety._rotation_x(-0.725), TabletopSafety._rotation_x(q))))
        return TabletopSafety._matmul(transform, TabletopSafety._transform((0.0, 0.0457554015893473, -0.0272203446692936), TabletopSafety._identity_rotation()))

    @staticmethod
    def _quaternion_rotation(quaternion: dict[str, Any]) -> tuple[tuple[float, ...], ...]:
        x, y, z, w = (_as_float(quaternion.get(axis), f"orientation.{axis}") for axis in ("x", "y", "z", "w"))
        length = math.sqrt(x * x + y * y + z * z + w * w)
        if length <= 1e-9:
            raise ValueError("orientation quaternion must not be zero.")
        x, y, z, w = x / length, y / length, z / length, w / length
        return (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        )

    @staticmethod
    def _rotation_x(angle: float) -> tuple[tuple[float, ...], ...]:
        cosine, sine = math.cos(angle), math.sin(angle)
        return ((1.0, 0.0, 0.0), (0.0, cosine, -sine), (0.0, sine, cosine))

    @staticmethod
    def _rotation_z(angle: float) -> tuple[tuple[float, ...], ...]:
        cosine, sine = math.cos(angle), math.sin(angle)
        return ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0))

    @staticmethod
    def _identity_rotation() -> tuple[tuple[float, ...], ...]:
        return ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))

    @staticmethod
    def _transform(translation: tuple[float, float, float], rotation: tuple[tuple[float, ...], ...]) -> tuple[tuple[float, ...], ...]:
        return tuple(tuple(rotation[row][column] for column in range(3)) + (translation[row],) for row in range(3)) + ((0.0, 0.0, 0.0, 1.0),)

    @staticmethod
    def _matmul(first: tuple[tuple[float, ...], ...], second: tuple[tuple[float, ...], ...]) -> tuple[tuple[float, ...], ...]:
        return tuple(tuple(sum(first[row][index] * second[index][column] for index in range(4)) for column in range(4)) for row in range(4))

    @staticmethod
    def _matmul3(first: tuple[tuple[float, ...], ...], second: tuple[tuple[float, ...], ...]) -> tuple[tuple[float, ...], ...]:
        return tuple(tuple(sum(first[row][index] * second[index][column] for index in range(3)) for column in range(3)) for row in range(3))

    @staticmethod
    def _transform_point(transform: tuple[tuple[float, ...], ...], point: tuple[float, float, float]) -> tuple[float, float, float]:
        return tuple(sum(transform[row][column] * point[column] for column in range(3)) + transform[row][3] for row in range(3))

    @staticmethod
    def _rotate(rotation: tuple[tuple[float, ...], ...], point: tuple[float, float, float]) -> tuple[float, float, float]:
        return tuple(sum(rotation[row][column] * point[column] for column in range(3)) for row in range(3))

    @staticmethod
    def _approach_direction(pregrasp: dict[str, Any], grasp: dict[str, Any]) -> tuple[float, float, float] | None:
        try:
            vector = tuple(
                _as_float(pregrasp["position"][axis], f"pregrasp.position.{axis}")
                - _as_float(grasp["position"][axis], f"grasp.position.{axis}")
                for axis in ("x", "y", "z")
            )
        except (KeyError, TypeError, ValueError):
            return None
        length = math.sqrt(sum(component * component for component in vector))
        if length <= 1e-9:
            return None
        return tuple(component / length for component in vector)
