"""Auditable geometric constraints for deterministic relative placement.

Language models may select symbolic relations, but only this module converts
supported relations into metric axis assignments.  It produces one pose and
never samples placement candidates or commands robot motion.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal


ConstraintKind = Literal[
    "axis_equal", "axis_separation", "axis_midpoint", "table_support", "surface_support",
]


@dataclass(frozen=True)
class GeometryConstraint:
    kind: ConstraintKind
    axes: tuple[str, ...]
    reference_ids: tuple[str, ...]
    direction: float = 0.0
    clearance_m: float = 0.0
    description: str = ""

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["axes"] = list(self.axes)
        value["reference_ids"] = list(self.reference_ids)
        return value


@dataclass(frozen=True)
class RelationConstraintSet:
    relation: str
    constraints: tuple[GeometryConstraint, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "relation": self.relation,
            "constraints": [constraint.as_dict() for constraint in self.constraints],
        }


class SpatialConstraintCompiler:
    """Compile supported symbolic relations into explicit base constraints."""

    def __init__(self, *, right_axis: str = "x", horizontal_clearance_m: float = 0.03) -> None:
        if right_axis not in {"x", "y"}:
            raise ValueError("right_axis must be 'x' or 'y'.")
        self.right_axis = right_axis
        self.horizontal_clearance_m = float(horizontal_clearance_m)

    def compile(self, relation: str, reference_ids: list[str]) -> RelationConstraintSet:
        relation = str(relation).strip().casefold()
        references = tuple(str(item) for item in reference_ids)
        if relation == "between":
            return RelationConstraintSet(relation, (
                GeometryConstraint("axis_midpoint", ("x", "y"), references,
                                   description="centre between two references"),
                GeometryConstraint("table_support", ("z",), references,
                                   description="preserve tabletop support"),
            ))
        if relation in {"right_of", "left_of"}:
            return RelationConstraintSet(relation, (
                GeometryConstraint(
                    "axis_separation", (self.right_axis,), references,
                    direction=1.0 if relation == "right_of" else -1.0,
                    clearance_m=self.horizontal_clearance_m,
                    description=f"separate along {self.right_axis}",
                ),
                GeometryConstraint("table_support", ("z",), references,
                                   description="preserve tabletop support"),
            ))
        if relation == "on_top_of":
            return RelationConstraintSet(relation, (
                GeometryConstraint("axis_equal", ("x", "y"), references,
                                   description="align horizontal centres"),
                GeometryConstraint("surface_support", ("z",), references,
                                   description="support source on reference top surface"),
            ))
        if relation in {"aligned_x_with", "aligned_y_with", "same_height_as"}:
            axis = {"aligned_x_with": "x", "aligned_y_with": "y", "same_height_as": "z"}[relation]
            return RelationConstraintSet(relation, (
                GeometryConstraint("axis_equal", (axis,), references,
                                   description=f"align {axis} coordinate"),
            ))
        raise ValueError(f"Unsupported constraint relation {relation!r}.")


class ConstraintConflictError(ValueError):
    """Raised when combined constraints assign incompatible axis values."""


class AxisAssignmentSolver:
    """Merge independently computed axis assignments into one target pose."""

    def __init__(self, *, tolerance_m: float = 0.002) -> None:
        self.tolerance_m = float(tolerance_m)

    def merge(self, seed: dict[str, float], assignments: list[dict[str, float]]) -> dict[str, float]:
        result = dict(seed)
        assigned: dict[str, float] = {}
        for assignment in assignments:
            for axis, raw_value in assignment.items():
                if axis not in {"x", "y", "z"}:
                    raise ValueError(f"Unsupported pose axis {axis!r}.")
                value = float(raw_value)
                previous = assigned.get(axis)
                if previous is not None and abs(previous - value) > self.tolerance_m:
                    raise ConstraintConflictError(
                        f"Combined constraints conflict on {axis}: {previous:.4f} vs {value:.4f} m."
                    )
                assigned[axis] = value
                result[axis] = value
        return result
