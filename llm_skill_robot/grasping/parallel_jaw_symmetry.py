"""Resolve the 180-degree axial symmetry of a parallel-jaw gripper."""

from __future__ import annotations

import copy
import math
from typing import Any


def current_closing_axis(config: dict[str, Any], transformer: Any | None) -> list[float] | None:
    """Return the directed gripper closing axis in the transformer's target frame."""
    configured = _vector(config.get("current_gripper_closing_axis"))
    if configured is not None:
        return configured
    frame = str(config.get("current_gripper_frame") or "").strip()
    lookup = getattr(transformer, "_lookup_transform", None)
    if not frame or not callable(lookup):
        return None
    transform = lookup(frame, timeout_sec=float(config.get("current_gripper_tf_timeout_sec", 1.0)))
    if transform is None:
        return None
    axis = _vector(config.get("current_gripper_axis", [0.0, 0.0, 1.0]))
    if axis is None:
        return None
    rotation = transform.transform.rotation
    return _rotate_vector(axis, (rotation.x, rotation.y, rotation.z, rotation.w))


def canonicalize_sparse_parallel_jaw_candidates(
    candidates: list[Any], current_axis: Any | None,
) -> tuple[list[Any], list[Any]]:
    """Choose the nearer of each Robotiq grasp's two equivalent axial poses.

    The returned fallback poses are kept separate so callers can avoid sending
    them to MoveIt unless every preferred representative fails.
    """
    reference = _planar_unit(current_axis)
    if reference is None:
        return list(candidates), []

    primary_by_key: dict[tuple[Any, ...], Any] = {}
    passthrough: list[Any] = []
    for candidate in candidates:
        if not _is_parallel_jaw_candidate(candidate):
            passthrough.append(candidate)
            continue
        metadata = _value(candidate, "metadata", {}) or {}
        if metadata.get("parallel_jaw_symmetry_role") == "fallback":
            passthrough.append(candidate)
            continue
        primary = copy.deepcopy(candidate)
        closing = _closing_axis(primary)
        closing_xy = _planar_unit(closing)
        if closing_xy is None:
            passthrough.append(candidate)
            continue
        rotated = closing_xy[0] * reference[0] + closing_xy[1] * reference[1] < 0.0
        if rotated:
            primary = _rotate_about_local_approach(primary)
        _mark(primary, "preferred", rotated)
        key = _symmetry_key(primary)
        old = primary_by_key.get(key)
        if old is None or _score(primary) > _score(old):
            primary_by_key[key] = primary

    primaries = passthrough + list(primary_by_key.values())
    fallbacks = []
    for primary in primary_by_key.values():
        fallback = _rotate_about_local_approach(primary)
        _mark(fallback, "fallback", True)
        fallbacks.append(fallback)
    return primaries, fallbacks


def _rotate_about_local_approach(candidate: Any) -> Any:
    result = copy.deepcopy(candidate)
    transform = _value(result, "transform")
    if isinstance(transform, list) and len(transform) >= 3:
        for row in range(3):
            if isinstance(transform[row], list) and len(transform[row]) >= 3:
                transform[row][0] = -float(transform[row][0])
                transform[row][1] = -float(transform[row][1])
    pose = _value(result, "pose")
    if isinstance(pose, dict) and isinstance(pose.get("orientation"), dict):
        orientation = pose["orientation"]
        try:
            q = tuple(float(orientation.get(name, default)) for name, default in (
                ("x", 0.0), ("y", 0.0), ("z", 0.0), ("w", 1.0)
            ))
            # q_new = q * quaternion(local Z, pi).
            x, y, z, w = q
            orientation.update({"x": y, "y": -x, "z": w, "w": -z})
        except (TypeError, ValueError):
            pass
    for name in ("closing_axis", "closing_axis_base", "width_axis", "gripper_width_axis"):
        value = _value(result, name)
        vector = _vector(value)
        if vector is not None:
            replacement = [-component for component in vector]
            if isinstance(value, dict):
                value.update(dict(zip(("x", "y", "z"), replacement)))
            elif isinstance(result, dict):
                result[name] = replacement
    return result


def _mark(candidate: Any, role: str, rotated: bool) -> None:
    if not isinstance(candidate, dict):
        return
    metadata = dict(candidate.get("metadata") or {})
    base_id = str(
        metadata.get("parallel_jaw_symmetry_base_candidate_id")
        or candidate.get("candidate_id", "")
    )
    metadata.update({
        "parallel_jaw_symmetry_role": role,
        "parallel_jaw_symmetry_rotation_deg": 180 if rotated else 0,
        "parallel_jaw_symmetry_base_candidate_id": base_id,
    })
    candidate["metadata"] = metadata
    if role == "fallback" and base_id:
        candidate["candidate_id"] = f"{base_id}__sym180"


def _symmetry_key(candidate: Any) -> tuple[Any, ...]:
    pose = _value(candidate, "pose", {}) or {}
    position = pose.get("position", {}) if isinstance(pose, dict) else {}
    closing = _planar_unit(_closing_axis(candidate)) or (0.0, 0.0)
    approach = _unit(_vector(_value(candidate, "approach_vector"))) or (0.0, 0.0, 0.0)
    return tuple(round(float(value), 4) for value in (
        position.get("x", 0.0), position.get("y", 0.0), position.get("z", 0.0),
        *closing, *approach,
    ))


def _closing_axis(candidate: Any) -> list[float] | None:
    for name in ("closing_axis_base", "closing_axis", "width_axis", "gripper_width_axis"):
        value = _vector(_value(candidate, name))
        if value is not None:
            return value
    return None


def _is_parallel_jaw_candidate(candidate: Any) -> bool:
    metadata = _value(candidate, "metadata", {}) or {}
    if not isinstance(metadata, dict):
        return False
    gripper = str(metadata.get("gripper_name") or "").lower()
    return (
        gripper in {"robotiq_2f_140", "robotiq-2f-140"}
        or metadata.get("generation_branch") == "obb_sparse"
    )


def _score(candidate: Any) -> float:
    try:
        return float(_value(candidate, "score", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _value(candidate: Any, name: str, default: Any = None) -> Any:
    return candidate.get(name, default) if isinstance(candidate, dict) else getattr(candidate, name, default)


def _vector(value: Any) -> list[float] | None:
    if isinstance(value, dict):
        try:
            return [float(value[name]) for name in ("x", "y", "z")]
        except (KeyError, TypeError, ValueError):
            return None
    try:
        values = list(value)
        return [float(values[index]) for index in range(3)] if len(values) >= 3 else None
    except (TypeError, ValueError):
        return None


def _unit(value: list[float] | None) -> tuple[float, float, float] | None:
    if value is None:
        return None
    norm = math.sqrt(sum(component * component for component in value))
    return tuple(component / norm for component in value) if norm > 1e-9 else None


def _planar_unit(value: Any) -> tuple[float, float] | None:
    vector = _vector(value)
    if vector is None:
        return None
    norm = math.hypot(vector[0], vector[1])
    return (vector[0] / norm, vector[1] / norm) if norm > 1e-9 else None


def _rotate_vector(vector: list[float], quaternion: tuple[float, float, float, float]) -> list[float]:
    x, y, z, w = quaternion
    vx, vy, vz = vector
    tx, ty, tz = 2.0 * (y * vz - z * vy), 2.0 * (z * vx - x * vz), 2.0 * (x * vy - y * vx)
    return [
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    ]
