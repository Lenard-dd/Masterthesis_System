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
    """A calibrated tabletop plane and conservative TCP-clearance policy."""

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
        )

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
        return {
            "allowed": True,
            "measured_clearance_m": measured,
            "required_clearance_m": required,
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
        return {"allowed": True, "checks": results}

    def _required_clearance(self, stage: str) -> float:
        requested = {
            "pregrasp": self.min_pregrasp_clearance_m,
            "grasp": self.min_grasp_clearance_m,
            "retreat": self.min_retreat_clearance_m,
        }.get(stage, self.min_grasp_clearance_m)
        return self.collision_margin_m + requested

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
