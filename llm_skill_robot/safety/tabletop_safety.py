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
    narrow_grasp_clearance_tiers: tuple[tuple[float, float], ...] = ()
    finger_envelope_enabled: bool = False
    tool0_to_gripper_base_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    finger_joint_min_rad: float = 0.0
    finger_joint_max_rad: float = 0.7
    finger_joint_samples: int = 15
    finger_extra_clearance_m: float = 0.0
    max_finger_envelope_lift_m: float = 0.012
    finger_envelope_hard_reject: bool = False

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
        raw_tiers = values.get("narrow_grasp_clearance_tiers", []) or []
        if not isinstance(raw_tiers, list):
            raise ValueError("narrow_grasp_clearance_tiers must be a list.")
        tiers: list[tuple[float, float]] = []
        for index, tier in enumerate(raw_tiers):
            if not isinstance(tier, dict):
                raise ValueError(f"narrow_grasp_clearance_tiers[{index}] must be a mapping.")
            maximum = _as_float(tier.get("max_width_m"), f"narrow_grasp_clearance_tiers[{index}].max_width_m")
            extra_clearance = _as_float(tier.get("extra_clearance_m"), f"narrow_grasp_clearance_tiers[{index}].extra_clearance_m")
            if maximum <= 0.0 or extra_clearance < 0.0:
                raise ValueError("Narrow-grasp width limits must be positive and clearances non-negative.")
            tiers.append((maximum, extra_clearance))
        tiers.sort(key=lambda item: item[0])
        envelope = values.get("robotiq_2f_140_finger_envelope", {}) or {}
        if not isinstance(envelope, dict):
            raise ValueError("robotiq_2f_140_finger_envelope must be a mapping.")
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
            narrow_grasp_clearance_tiers=tuple(tiers),
            finger_envelope_enabled=bool(envelope.get("enabled", False)),
            tool0_to_gripper_base_m=tuple(
                _as_float(value, "finger-envelope tool0_to_gripper_base_m")
                for value in base_offset
            ),
            finger_joint_min_rad=_as_float(envelope.get("joint_min_rad", 0.0), "finger-envelope joint_min_rad"),
            finger_joint_max_rad=_as_float(envelope.get("joint_max_rad", 0.7), "finger-envelope joint_max_rad"),
            finger_joint_samples=samples,
            finger_extra_clearance_m=_as_float(envelope.get("extra_clearance_m", 0.0), "finger-envelope extra_clearance_m"),
            max_finger_envelope_lift_m=_as_float(envelope.get("max_auto_lift_m", 0.012), "finger-envelope max_auto_lift_m"),
            finger_envelope_hard_reject=bool(envelope.get("hard_reject", False)),
        )

    def apply_grasp_contact_backoff(
        self, poses: dict[str, Any], *, gripper_width_m: float | None = None
    ) -> dict[str, Any]:
        """Move final grasp and retreat back along the gripper approach axis.

        Pregrasp remains unchanged, so the supervised approach simply ends a
        little earlier on the same approach line. Retreat receives the same
        translation to preserve its displacement from the adjusted final grasp.
        """
        if self.grasp_contact_backoff_m < 0.0:
            raise ValueError("grasp_contact_backoff_m must be non-negative.")
        if self.grasp_contact_backoff_m <= 0.0:
            adjusted = poses
        else:
            adjusted = self._apply_approach_backoff(poses)
        return self._apply_minimum_table_clearance(adjusted, gripper_width_m)

    def _apply_approach_backoff(self, poses: dict[str, Any]) -> dict[str, Any]:
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

    def narrow_grasp_extra_clearance(self, gripper_width_m: float | None) -> float:
        """Return calibrated extra fingertip clearance for a narrow grasp."""
        if gripper_width_m is None:
            return 0.0
        width = _as_float(gripper_width_m, "gripper_width_m")
        if width < 0.0:
            raise ValueError("gripper_width_m must be non-negative.")
        for maximum, clearance in self.narrow_grasp_clearance_tiers:
            if width <= maximum:
                return clearance
        return 0.0

    def _apply_minimum_table_clearance(
        self, poses: dict[str, Any], gripper_width_m: float | None
    ) -> dict[str, Any]:
        """Lift grasp/retreat just enough for the closed fingertip sweep.

        The correction follows the calibrated table normal, so it remains
        valid when the physical table is slightly tilted in ``base_link``.
        """
        if not self.enabled:
            return poses
        grasp = poses.get("grasp")
        pregrasp = poses.get("pregrasp")
        if not isinstance(grasp, dict) or not isinstance(pregrasp, dict):
            raise ValueError("Table-clearance lift requires pregrasp and grasp poses.")
        direction = self._approach_direction(pregrasp, grasp)
        if direction is None:
            raise ValueError("Table-clearance lift cannot calculate the approach direction.")
        extra = self.narrow_grasp_extra_clearance(gripper_width_m)
        if self.finger_envelope_enabled:
            envelope = self._finger_envelope_minimum(grasp, "grasp")
            measured = float(envelope["minimum_clearance_m"])
            required = self._required_clearance("grasp") + self.finger_extra_clearance_m + extra
        else:
            open_point = self.grasp_contact_point(grasp)
            closed_point = {
                axis: open_point[axis] - direction[index] * self.finger_axial_close_sweep_m
                for index, axis in enumerate(("x", "y", "z"))
            }
            measured = self.signed_distance_m(closed_point)
            required = self._required_clearance("grasp") + self.close_sweep_compensation_m + extra
        lift = max(0.0, required - measured)
        if lift <= 1e-9:
            return poses
        if self.finger_envelope_enabled and lift > self.max_finger_envelope_lift_m:
            if self.finger_envelope_hard_reject:
                raise ValueError(
                    f"Robotiq finger envelope requires {lift:.4f} m upward correction, "
                    f"above the {self.max_finger_envelope_lift_m:.4f} m safe auto-lift limit."
                )
            # The pad-box proxy intentionally samples the whole closing range
            # and can overestimate intrusion for a candidate's actual contact
            # width.  Use it as a bounded correction instead of rejecting all
            # otherwise reachable candidates.
            lift = self.max_finger_envelope_lift_m
        adjusted = dict(poses)
        for name in ("grasp", "retreat"):
            pose = poses.get(name)
            if not isinstance(pose, dict) or not isinstance(pose.get("position"), dict):
                raise ValueError(f"Table-clearance lift requires a valid {name} pose.")
            position = dict(pose["position"])
            for index, axis in enumerate(("x", "y", "z")):
                position[axis] = _as_float(position.get(axis), f"{name}.position.{axis}") + self.normal[index] * lift
            adjusted[name] = {**pose, "position": position}
        return adjusted

    def _finger_envelope_minimum(self, pose: dict[str, Any], stage: str) -> dict[str, Any]:
        """Return the lowest Robotiq pad-box corner over the closing sweep."""
        orientation = pose.get("orientation")
        position = pose.get("position")
        if not isinstance(orientation, dict) or not isinstance(position, dict):
            raise ValueError("Finger-envelope safety requires position and orientation.")
        rotation = self._quaternion_rotation(orientation)
        origin = tuple(_as_float(position.get(axis), f"position.{axis}") for axis in ("x", "y", "z"))
        if stage == "pregrasp":
            joint_values = (self.finger_joint_min_rad,)
        elif stage == "retreat":
            joint_values = (self.finger_joint_max_rad,)
        else:
            step = (self.finger_joint_max_rad - self.finger_joint_min_rad) / (self.finger_joint_samples - 1)
            joint_values = tuple(self.finger_joint_min_rad + index * step for index in range(self.finger_joint_samples))
        lowest: tuple[float, str, float] | None = None
        for joint_angle in joint_values:
            for name, local_point in self._robotiq_pad_corners(joint_angle):
                rotated = self._rotate(rotation, local_point)
                point = tuple(origin[index] + rotated[index] for index in range(3))
                clearance = sum(component * coordinate for component, coordinate in zip(self.normal, point)) + self.d
                if lowest is None or clearance < lowest[0]:
                    lowest = (clearance, name, joint_angle)
        assert lowest is not None
        return {
            "minimum_clearance_m": lowest[0],
            "lowest_proxy": lowest[1],
            "joint_angle_rad": lowest[2],
        }

    def _robotiq_pad_corners(self, joint_angle: float) -> tuple[tuple[str, tuple[float, float, float]], ...]:
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
        base_y = -0.030601 if side == "left" else 0.030601
        base_rotation = TabletopSafety._matmul3(
            TabletopSafety._rotation_x(math.pi / 2.0 + 0.725),
            TabletopSafety._rotation_z(0.0 if side == "left" else math.pi),
        )
        transform = TabletopSafety._transform((0.0, base_y, 0.054905), TabletopSafety._matmul3(base_rotation, TabletopSafety._rotation_x(-q)))
        transform = TabletopSafety._matmul(transform, TabletopSafety._transform((0.0, 0.01821998610742, 0.0260018192872234), TabletopSafety._identity_rotation()))
        transform = TabletopSafety._matmul(transform, TabletopSafety._transform((0.0, 0.0817554015893473, -0.0282203446692936), TabletopSafety._matmul3(TabletopSafety._rotation_x(-0.725), TabletopSafety._rotation_x(q))))
        return TabletopSafety._matmul(transform, TabletopSafety._transform((0.0, 0.0457554015893473, -0.0272203446692936), TabletopSafety._identity_rotation()))

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
        finger_envelope = None
        if self.finger_envelope_enabled and self.finger_envelope_hard_reject:
            try:
                finger_envelope = self._finger_envelope_minimum(pose, stage)
            except ValueError as exc:
                return {"allowed": False, "reason": str(exc)}
            finger_required = self._required_clearance(stage) + self.finger_extra_clearance_m
            if finger_envelope["minimum_clearance_m"] + 1e-9 < finger_required:
                return {
                    "allowed": False,
                    "reason": (
                        f"{stage} Robotiq finger envelope clearance "
                        f"{finger_envelope['minimum_clearance_m']:.4f} m "
                        f"({finger_envelope['lowest_proxy']}, joint="
                        f"{finger_envelope['joint_angle_rad']:.3f} rad) is below "
                        f"required {finger_required:.4f} m."
                    ),
                    "measured_clearance_m": finger_envelope["minimum_clearance_m"],
                    "required_clearance_m": finger_required,
                    "finger_envelope": finger_envelope,
                }
        return {
            "allowed": True,
            "measured_clearance_m": measured,
            "required_clearance_m": required,
            "grasp_contact_point": contact_point,
            **({"finger_envelope": finger_envelope} if finger_envelope else {}),
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
    def _transform(translation, rotation) -> tuple[tuple[float, ...], ...]:
        return tuple(
            tuple(rotation[row][column] for column in range(3)) + (translation[row],)
            for row in range(3)
        ) + ((0.0, 0.0, 0.0, 1.0),)

    @staticmethod
    def _matmul(first, second) -> tuple[tuple[float, ...], ...]:
        return tuple(tuple(sum(first[row][index] * second[index][column] for index in range(4)) for column in range(4)) for row in range(4))

    @staticmethod
    def _matmul3(first, second) -> tuple[tuple[float, ...], ...]:
        return tuple(tuple(sum(first[row][index] * second[index][column] for index in range(3)) for column in range(3)) for row in range(3))

    @staticmethod
    def _transform_point(transform, point) -> tuple[float, float, float]:
        return tuple(sum(transform[row][column] * point[column] for column in range(3)) + transform[row][3] for row in range(3))

    @staticmethod
    def _rotate(rotation, point) -> tuple[float, float, float]:
        return tuple(sum(rotation[row][column] * point[column] for column in range(3)) for row in range(3))

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
