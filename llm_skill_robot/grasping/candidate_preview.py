"""Grasp pose preparation and MoveIt screening shared by real and simulated workflows."""

from __future__ import annotations

import math
from typing import Any, Callable
from llm_skill_robot.grasping.graspgenx_grasp_planner import screen_candidate_order
from llm_skill_robot.grasping.parallel_jaw_symmetry import canonicalize_sparse_parallel_jaw_candidates, current_closing_axis
from llm_skill_robot.perception.tf_pose_transformer import TFPoseTransformer


def _candidate_value(candidate: Any, name: str, default: Any = None) -> Any:
    return candidate.get(name, default) if isinstance(candidate, dict) else getattr(candidate, name, default)


def create_grasp_pose(candidate: Any, fallback_frame: str = "base_link") -> dict:
    pose = _candidate_value(candidate, "pose")
    frame = str(
        _candidate_value(candidate, "pose_frame", None)
        or (pose.get("frame") if isinstance(pose, dict) else None)
        or fallback_frame
    )
    if isinstance(pose, dict) and isinstance(pose.get("position"), dict):
        position = {key: float(pose["position"][key]) for key in ("x", "y", "z")}
        source_orientation = pose.get("orientation", {})
        orientation = {
            "x": float(source_orientation.get("x", 0.0)),
            "y": float(source_orientation.get("y", 0.0)),
            "z": float(source_orientation.get("z", 0.0)),
            "w": float(source_orientation.get("w", 1.0)),
        }
        return {"frame": frame, "position": position, "orientation": orientation}

    matrix = _candidate_value(candidate, "transform", pose)
    if _is_transform_matrix(matrix):
        return {
            "frame": frame,
            "position": {
                "x": float(matrix[0][3]),
                "y": float(matrix[1][3]),
                "z": float(matrix[2][3]),
            },
            "orientation": _matrix_to_quaternion(matrix),
        }
    raise ValueError("Selected candidate does not provide a usable grasp pose.")

def create_preview_poses(
    candidate: Any,
    config: dict | None = None,
    fallback_frame: str = "base_link",
) -> dict[str, dict]:
    config = config or {}
    grasp = create_grasp_pose(candidate, fallback_frame)
    approach = _normalized_vector(
        _candidate_value(candidate, "approach_vector", None)
        or _rotate_local_z(grasp["orientation"])
    )
    pregrasp_distance = float(config.get("pregrasp_distance_m", 0.10))
    retreat_distance = float(config.get("retreat_distance_m", 0.10))
    retreat_vector = _normalized_vector(config.get("retreat_vector", [0.0, 0.0, 1.0]))
    pregrasp = _offset_pose(grasp, approach, -pregrasp_distance)
    retreat = _offset_pose(grasp, retreat_vector, retreat_distance)
    return {"pregrasp": pregrasp, "grasp": grasp, "retreat": retreat}

def localize_graspgenx_candidate(candidate: Any, point_cloud: dict) -> dict:
    """Convert a GraspGenX centered-point-cloud candidate back to camera frame."""
    frame = point_cloud.get("frame")
    centroid = point_cloud.get("centroid_m")
    if not frame or not isinstance(centroid, dict):
        return dict(candidate) if isinstance(candidate, dict) else candidate
    pose_frame = _candidate_value(candidate, "pose_frame")
    pose = _candidate_value(candidate, "pose", {}) or {}
    if pose_frame != "object_point_cloud" and _candidate_value(pose, "frame") != "object_point_cloud":
        return dict(candidate) if isinstance(candidate, dict) else candidate

    localized = dict(candidate)
    dx, dy, dz = float(centroid["x"]), float(centroid["y"]), float(centroid["z"])
    matrix = _candidate_value(candidate, "transform")
    if _is_transform_matrix(matrix):
        transform = [list(row) for row in matrix]
        transform[0][3] = float(transform[0][3]) + dx
        transform[1][3] = float(transform[1][3]) + dy
        transform[2][3] = float(transform[2][3]) + dz
        localized["transform"] = transform

    if isinstance(pose, dict) and isinstance(pose.get("position"), dict):
        localized_pose = {
            "frame": frame,
            "position": {
                "x": float(pose["position"]["x"]) + dx,
                "y": float(pose["position"]["y"]) + dy,
                "z": float(pose["position"]["z"]) + dz,
            },
            "orientation": dict(pose.get("orientation", {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0})),
        }
        localized["pose"] = localized_pose
    localized["pose_frame"] = frame
    metadata = dict(localized.get("metadata", {}))
    metadata["object_point_cloud_frame"] = "object_point_cloud"
    metadata["localized_frame"] = frame
    metadata["point_cloud_centroid_m"] = dict(centroid)
    localized["metadata"] = metadata
    return localized

def transform_pose_sequence_to_target(
    poses: dict[str, dict],
    transformer: TFPoseTransformer | None,
    target_frame: str,
) -> dict[str, dict] | None:
    if not target_frame or all(pose.get("frame") == target_frame for pose in poses.values()):
        return poses
    if transformer is None or not transformer.available:
        return None
    transformed = {}
    for name, pose in poses.items():
        transformed_pose = transformer.transform_pose_dict(pose)
        if transformed_pose is None:
            return None
        transformed[name] = transformed_pose
    return transformed

def apply_grasp_frame_to_ee_transform(
    poses: dict[str, dict],
    config: dict | None = None,
) -> dict[str, dict]:
    """Convert planned GraspGenX grasp-frame poses into MoveIt EE/tool0 poses.

    GraspGenX candidates describe a grasp frame. MoveIt, however, plans the
    configured end-effector frame, normally ``tool0``. The configured transform
    follows:

        T_base_ee = T_base_grasp * T_grasp_ee

    Translation is expressed in the GraspGenX grasp frame.
    """
    config = config or {}
    if not config.get("enabled", False):
        return poses
    return {
        name: _apply_local_pose_transform(pose, config)
        for name, pose in poses.items()
    }

def _environment_trajectory_poses(candidate: Any, point_cloud: dict) -> dict[str, dict] | None:
    trajectory = _candidate_value(candidate, "trajectory", None)
    if not isinstance(trajectory, dict):
        return None
    names = {
        "pregrasp": "pregrasp_transform",
        "grasp": "grasp_transform",
        "retreat": "retreat_transform",
    }
    if not all(_is_transform_matrix(trajectory.get(key)) for key in names.values()):
        return None
    source_frame = str(trajectory.get("frame") or _candidate_value(candidate, "pose_frame", ""))
    output_frame = source_frame
    offset = (0.0, 0.0, 0.0)
    if source_frame == "object_point_cloud":
        centroid = point_cloud.get("centroid_m")
        if not isinstance(centroid, dict) or not point_cloud.get("frame"):
            raise ValueError("Environment trajectory requires the object-cloud centroid and frame.")
        offset = tuple(float(centroid[axis]) for axis in ("x", "y", "z"))
        output_frame = str(point_cloud["frame"])
    poses: dict[str, dict] = {}
    for name, key in names.items():
        matrix = trajectory[key]
        poses[name] = {
            "frame": output_frame,
            "position": {
                "x": float(matrix[0][3]) + offset[0],
                "y": float(matrix[1][3]) + offset[1],
                "z": float(matrix[2][3]) + offset[2],
            },
            "orientation": _matrix_to_quaternion(matrix),
        }
    return poses

def prepare_candidate_preview(
    candidate: Any,
    point_cloud: dict,
    preview_config: dict | None,
    target_frame: str,
    transformer: TFPoseTransformer | None,
) -> dict:
    """Build localized, TF-transformed MoveIt EE preview poses for a candidate."""
    preview_config = preview_config or {}
    poses = _environment_trajectory_poses(candidate, point_cloud)
    uses_environment_trajectory = poses is not None
    if poses is None:
        localized = localize_graspgenx_candidate(candidate, point_cloud)
        poses = create_preview_poses(
            localized,
            preview_config,
            fallback_frame=point_cloud.get("frame") or target_frame or "base_link",
        )
    if target_frame:
        transformed_poses = transform_pose_sequence_to_target(
            poses, transformer, target_frame
        )
        if transformed_poses is None:
            raise ValueError(
                f"TF transform from {poses['grasp']['frame']} to {target_frame} "
                "is unavailable."
            )
        poses = transformed_poses
    poses = apply_grasp_frame_to_ee_transform(
        poses,
        preview_config.get("grasp_frame_to_ee_frame", {}),
    )
    if (
        not uses_environment_trajectory
        and str(preview_config.get("retreat_vector_frame", "pose_frame")).lower() in {
            "target_frame",
            "base_link",
        }
    ):
        retreat_vector = _normalized_vector(
            preview_config.get("retreat_vector", [0.0, 0.0, 1.0])
        )
        retreat_distance = float(preview_config.get("retreat_distance_m", 0.10))
        poses = dict(poses)
        poses["retreat"] = _offset_pose(poses["grasp"], retreat_vector, retreat_distance)
    if uses_environment_trajectory:
        poses = dict(poses)
        poses["_trajectory_source"] = "graspgenx_environment_screened"
        poses["_execution_backoff_applied"] = True
    return poses

def planning_variant_kwargs(
    variant: str,
    screening_config: dict | None = None,
) -> dict:
    """Return MoveIt pose constraint kwargs for a named plan-only variant."""
    screening_config = screening_config or {}
    position_tolerance = float(screening_config.get("position_tolerance_m", 0.01))
    if variant == "full_pose":
        return {
            "position_tolerance": position_tolerance,
            "orientation_tolerance": float(
                screening_config.get("full_orientation_tolerance_rad", 0.05)
            ),
        }
    if variant == "relaxed_orientation":
        return {
            "position_tolerance": position_tolerance,
            "orientation_tolerance": float(
                screening_config.get("relaxed_orientation_tolerance_rad", 0.75)
            ),
        }
    if variant == "position_only":
        return {
            "position_tolerance": position_tolerance,
            "use_orientation_constraint": False,
        }
    raise ValueError(f"Unknown MoveIt planning variant '{variant}'.")

def moveit_planning_kwargs(preview_config: dict | None = None) -> dict:
    """Return MoveIt request tuning kwargs from plan_only_preview config."""
    preview_config = preview_config or {}
    moveit_config = preview_config.get("moveit", {}) or {}
    kwargs = {}
    pipeline_id = moveit_config.get("pipeline_id")
    if pipeline_id:
        kwargs["pipeline_id"] = str(pipeline_id)
    planner_id = moveit_config.get("planner_id")
    if planner_id:
        kwargs["planner_id"] = str(planner_id)
    if "num_planning_attempts" in moveit_config:
        kwargs["num_planning_attempts"] = int(moveit_config["num_planning_attempts"])
    if "allowed_planning_time_sec" in moveit_config:
        kwargs["allowed_planning_time"] = float(
            moveit_config["allowed_planning_time_sec"]
        )
    if "service_timeout_sec" in moveit_config:
        kwargs["service_timeout_sec"] = float(moveit_config["service_timeout_sec"])
    return kwargs

def moveit_stage_planning_kwargs(
    preview_config: dict | None,
    stage: str,
    base_kwargs: dict | None = None,
) -> dict:
    """Return per-stage MoveIt tuning, e.g. Pilz PTP for pregrasp and LIN after."""
    preview_config = preview_config or {}
    moveit_config = preview_config.get("moveit", {}) or {}
    kwargs = dict(base_kwargs or moveit_planning_kwargs(preview_config))
    stage_planner_ids = moveit_config.get("stage_planner_ids", {}) or {}
    if stage in stage_planner_ids:
        kwargs["planner_id"] = str(stage_planner_ids[stage])
    stage_pipeline_ids = moveit_config.get("stage_pipeline_ids", {}) or {}
    if stage in stage_pipeline_ids:
        kwargs["pipeline_id"] = str(stage_pipeline_ids[stage])
    stage_overrides = moveit_config.get("stage_overrides", {}) or {}
    if stage in stage_overrides:
        override = stage_overrides[stage] or {}
        if "planner_id" in override:
            kwargs["planner_id"] = str(override["planner_id"])
        if "pipeline_id" in override:
            kwargs["pipeline_id"] = str(override["pipeline_id"])
        if "num_planning_attempts" in override:
            kwargs["num_planning_attempts"] = int(override["num_planning_attempts"])
        if "allowed_planning_time_sec" in override:
            kwargs["allowed_planning_time"] = float(
                override["allowed_planning_time_sec"]
            )
        if "service_timeout_sec" in override:
            kwargs["service_timeout_sec"] = float(override["service_timeout_sec"])
    return kwargs

def screen_grasp_candidates_for_plan_only(
    candidates: list[Any],
    point_cloud: dict,
    preview_config: dict | None,
    target_frame: str,
    transformer: TFPoseTransformer | None,
    backend,
    trajectory_validator,
    velocity_scale: float = 0.05,
    acceleration_scale: float = 0.05,
    pose_adjuster: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    candidate_pose_adjuster: Callable[[dict[str, Any], Any], dict[str, Any]] | None = None,
    pose_validator: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> dict:
    """Select a GraspGenX candidate using MoveIt reachability and validation.

    This function sends plan-only requests to MoveIt. It never executes a
    trajectory. Candidates are still primarily generated and scored by
    GraspGenX, but the final selected candidate must also be plausible for the
    current UR5 joint state.
    """
    preview_config = preview_config or {}
    screening_config = preview_config.get("candidate_screening", {}) or {}
    if not screening_config.get("enabled", True):
        return {"selected": None, "poses": None, "variant": None, "attempts": []}

    variants = list(
        screening_config.get(
            "variants", ["full_pose", "relaxed_orientation", "position_only"]
        )
    )
    if not screening_config.get("allow_position_only_fallback", True):
        variants = [variant for variant in variants if variant != "position_only"]
    max_candidates = max(1, int(screening_config.get("max_candidates", 12)))
    base_plan_kwargs = moveit_planning_kwargs(preview_config)
    valid_candidates = [
        candidate
        for candidate in candidates
        if _candidate_value(candidate, "valid", True) is not False
    ]
    symmetry_fallbacks = []
    symmetry_config = screening_config.get("grasp_preferences", {}) or {}
    if symmetry_config.get("parallel_jaw_symmetry_enabled", True):
        valid_candidates, symmetry_fallbacks = canonicalize_sparse_parallel_jaw_candidates(
            valid_candidates,
            current_closing_axis(symmetry_config, transformer),
        )
    valid_candidates = screen_candidate_order(
        valid_candidates, max_candidates,
        int(screening_config.get("sparse_screen_quota", 12)),
    )
    selected_ids = {
        str(_candidate_value(candidate, "candidate_id", ""))
        for candidate in valid_candidates
    }
    symmetry_fallbacks = [
        candidate for candidate in symmetry_fallbacks
        if str((_candidate_value(candidate, "metadata", {}) or {}).get(
            "parallel_jaw_symmetry_base_candidate_id", ""
        )) in selected_ids
    ]

    attempts = []
    accepted_strict = []
    accepted_position_only = []
    for candidate_index, candidate in enumerate(valid_candidates, start=1):
        candidate_id = _candidate_value(candidate, "candidate_id", f"candidate_{candidate_index}")
        candidate_score = float(_candidate_value(candidate, "score", 0.0) or 0.0)
        try:
            poses = prepare_candidate_preview(
                candidate,
                point_cloud,
                preview_config,
                target_frame,
                transformer,
            )
            if candidate_pose_adjuster is not None:
                poses = candidate_pose_adjuster(poses, candidate)
            elif pose_adjuster is not None:
                poses = pose_adjuster(poses)
        except ValueError as exc:
            attempts.append(
                {
                    "candidate": candidate,
                    "candidate_id": candidate_id,
                    "score": candidate_score,
                    "variant": "prepare",
                    "accepted": False,
                    "message": str(exc),
                    "validation": {"decision": "BLOCK", "reasons": [str(exc)]},
                    "points": 0,
                    "max_joint_delta_rad": None,
                }
            )
            continue

        pose_validation = (
            pose_validator(poses)
            if pose_validator is not None
            else {"allowed": True, "reason": "No additional pose validator."}
        )
        if not bool(pose_validation.get("allowed", False)):
            reason = str(
                pose_validation.get("reason")
                or "Candidate was blocked by the additional pose validator."
            )
            attempts.append(
                {
                    "candidate": candidate,
                    "candidate_id": candidate_id,
                    "score": candidate_score,
                    "poses": poses,
                    "variant": "tabletop_safety",
                    "accepted": False,
                    "message": reason,
                    "validation": {
                        "decision": "BLOCK",
                        "reasons": [reason],
                        "tabletop_safety": pose_validation,
                    },
                    "tabletop_safety": pose_validation,
                    "points": 0,
                    "max_joint_delta_rad": None,
                }
            )
            continue

        metadata = _candidate_value(candidate, "metadata", {}) or {}
        candidate_variants = ["full_pose"] if metadata.get("generation_branch") == "obb_sparse" else variants
        for variant in candidate_variants:
            kwargs = {
                **base_plan_kwargs,
                **planning_variant_kwargs(variant, screening_config),
            }
            stage_kwargs = {
                stage: moveit_stage_planning_kwargs(preview_config, stage, kwargs)
                for stage in ("pregrasp", "grasp", "retreat")
            }
            sequence = plan_pose_sequence(
                backend,
                poses,
                trajectory_validator,
                velocity_scale=velocity_scale,
                acceleration_scale=acceleration_scale,
                plan_kwargs=kwargs,
                stage_plan_kwargs=stage_kwargs,
                max_replanning_attempts=int(
                    screening_config.get("sequence_replanning_attempts", 1)
                ),
            )
            summaries = [
                stage["plan"].get("summary", {})
                for stage in sequence.get("stages", [])
            ]
            summary = _sequence_screening_summary(summaries)
            validation = (
                {"decision": "ALLOW", "reasons": [], "warnings": []}
                if sequence.get("success")
                else _failed_sequence_validation(sequence)
            )
            failed_stage = _failed_sequence_stage(sequence)
            attempt = {
                "candidate": candidate,
                "candidate_id": candidate_id,
                "score": candidate_score,
                "poses": poses,
                "variant": variant,
                "plan_kwargs": kwargs,
                "sequence": sequence,
                "plan": sequence["stages"][-1]["plan"] if sequence.get("stages") else {},
                "summary": summary,
                "validation": validation,
                "tabletop_safety": pose_validation,
                "accepted": bool(sequence.get("success"))
                and validation.get("decision") == "ALLOW",
                "message": (
                    "full sequence reachable"
                    if sequence.get("success")
                    else f"failed at {failed_stage}"
                ),
                "failed_stage": failed_stage,
                "points": int(summary.get("num_trajectory_points") or 0),
                "max_joint_delta_rad": summary.get("max_joint_delta_rad"),
                "total_joint_delta_rad": summary.get("total_joint_delta_rad"),
                "wrist_3_joint_delta_rad": summary.get("wrist_3_joint_delta_rad"),
                "joint_deltas_rad": summary.get("joint_deltas_rad", {}),
            }
            attempts.append(attempt)
            if attempt["accepted"]:
                if variant == "position_only":
                    accepted_position_only.append(attempt)
                else:
                    accepted_strict.append(attempt)

    def ranking_key(attempt: dict) -> tuple:
        max_delta = attempt.get("max_joint_delta_rad")
        total_delta = attempt.get("total_joint_delta_rad")
        wrist_delta = attempt.get("wrist_3_joint_delta_rad")
        wrist_weight = max(0.0, float(screening_config.get("wrist_rotation_weight", 2.0)))
        if total_delta is None:
            motion_cost = float(max_delta) if max_delta is not None else float("inf")
        else:
            motion_cost = float(total_delta) + wrist_weight * float(wrist_delta or 0.0)
        return (
            motion_cost,
            float(max_delta) if max_delta is not None else float("inf"),
            int(attempt.get("points") or 999999),
            -float(attempt.get("score") or 0.0),
        )

    selected_attempts = accepted_strict or accepted_position_only
    if not selected_attempts:
        if symmetry_fallbacks:
            fallback_config = dict(preview_config)
            fallback_screening = dict(screening_config)
            fallback_preferences = dict(symmetry_config)
            fallback_preferences["parallel_jaw_symmetry_enabled"] = False
            fallback_screening["grasp_preferences"] = fallback_preferences
            fallback_screening["max_candidates"] = min(max_candidates, len(symmetry_fallbacks))
            fallback_config["candidate_screening"] = fallback_screening
            fallback_result = screen_grasp_candidates_for_plan_only(
                symmetry_fallbacks, point_cloud, fallback_config, target_frame,
                transformer, backend, trajectory_validator, velocity_scale,
                acceleration_scale, pose_adjuster,
                candidate_pose_adjuster=candidate_pose_adjuster,
                pose_validator=pose_validator,
            )
            fallback_result["attempts"] = attempts + fallback_result.get("attempts", [])
            return fallback_result
        return {"selected": None, "poses": None, "variant": None, "attempts": attempts}

    if symmetry_config.get("prefer_reachable_obb_top_down", True):
        maximum_preferred_tilt = float(
            symmetry_config.get("preferred_obb_max_tilt_deg", 2.0)
        )
        preferred_attempts = []
        for attempt in selected_attempts:
            candidate = attempt.get("candidate")
            metadata = _candidate_value(candidate, "metadata", {}) or {}
            approach = _candidate_value(candidate, "approach_vector", None)
            try:
                vector = [float(approach[index]) for index in range(3)]
                length = math.sqrt(sum(component * component for component in vector))
                tilt = math.degrees(math.acos(min(1.0, abs(vector[2]) / length)))
            except (TypeError, ValueError, IndexError, ZeroDivisionError):
                tilt = float("inf")
            if (
                metadata.get("generation_branch") == "obb_sparse"
                and tilt <= maximum_preferred_tilt
            ):
                preferred_attempts.append(attempt)
        if preferred_attempts:
            selected_attempts = preferred_attempts

    selected_attempt = sorted(selected_attempts, key=ranking_key)[0]
    selected_metadata = _candidate_value(selected_attempt["candidate"], "metadata", {}) or {}
    wrist_delta = selected_attempt.get("wrist_3_joint_delta_rad")
    wrist_threshold = float(
        symmetry_config.get("symmetry_replan_wrist_threshold_rad", 2.09439510239)
    )
    if (
        symmetry_fallbacks
        and selected_metadata.get("parallel_jaw_symmetry_role") == "preferred"
        and wrist_delta is not None
        and float(wrist_delta) >= wrist_threshold
    ):
        selected_base_id = str(
            selected_metadata.get("parallel_jaw_symmetry_base_candidate_id")
            or selected_attempt.get("candidate_id", "")
        )
        matching_fallback = next(
            (
                candidate for candidate in symmetry_fallbacks
                if str((_candidate_value(candidate, "metadata", {}) or {}).get(
                    "parallel_jaw_symmetry_base_candidate_id", ""
                )) == selected_base_id
            ),
            None,
        )
        if matching_fallback is not None:
            fallback_config = dict(preview_config)
            fallback_screening = dict(screening_config)
            fallback_preferences = dict(symmetry_config)
            fallback_preferences["parallel_jaw_symmetry_enabled"] = False
            fallback_screening["grasp_preferences"] = fallback_preferences
            fallback_screening["max_candidates"] = 1
            fallback_config["candidate_screening"] = fallback_screening
            fallback_result = screen_grasp_candidates_for_plan_only(
                [matching_fallback], point_cloud, fallback_config, target_frame,
                transformer, backend, trajectory_validator, velocity_scale,
                acceleration_scale, pose_adjuster,
                candidate_pose_adjuster=candidate_pose_adjuster,
                pose_validator=pose_validator,
            )
            attempts += fallback_result.get("attempts", [])
            fallback_attempt = fallback_result.get("attempt")
            if fallback_attempt is not None and ranking_key(fallback_attempt) < ranking_key(selected_attempt):
                selected_attempt = fallback_attempt
    return {
        "selected": selected_attempt["candidate"],
        "poses": selected_attempt["poses"],
        "variant": selected_attempt["variant"],
        "plan_kwargs": selected_attempt["plan_kwargs"],
        "attempt": selected_attempt,
        "attempts": attempts,
    }

def _sequence_screening_summary(summaries: list[dict]) -> dict:
    if not summaries:
        return {
            "success": False,
            "num_trajectory_points": 0,
            "max_joint_delta_rad": None,
            "total_joint_delta_rad": None,
            "wrist_3_joint_delta_rad": None,
            "joint_deltas_rad": {},
            "trajectory_preview": {},
        }
    max_delta_values = [
        summary.get("max_joint_delta_rad")
        for summary in summaries
        if summary.get("max_joint_delta_rad") is not None
    ]
    joint_deltas: dict[str, float] = {}
    joint_abs_deltas: dict[str, float] = {}
    total_joint_delta = 0.0
    found_joint_data = False
    for summary in summaries:
        preview = summary.get("trajectory_preview", {}) or {}
        first = preview.get("first_point_positions") or []
        last = preview.get("last_point_positions") or []
        names = summary.get("joint_names") or []
        count = min(len(first), len(last), len(names))
        for index in range(count):
            delta = float(last[index]) - float(first[index])
            name = str(names[index])
            joint_deltas[name] = joint_deltas.get(name, 0.0) + delta
            joint_abs_deltas[name] = joint_abs_deltas.get(name, 0.0) + abs(delta)
            total_joint_delta += abs(delta)
            found_joint_data = True
    return {
        "success": all(summary.get("success", False) for summary in summaries),
        "num_trajectory_points": sum(
            int(summary.get("num_trajectory_points") or 0)
            for summary in summaries
        ),
        "max_joint_delta_rad": max(max_delta_values) if max_delta_values else None,
        "total_joint_delta_rad": total_joint_delta if found_joint_data else None,
        "wrist_3_joint_delta_rad": (
            joint_abs_deltas["wrist_3_joint"]
            if "wrist_3_joint" in joint_abs_deltas else None
        ),
        "joint_deltas_rad": joint_deltas,
        "trajectory_preview": {},
    }

def _failed_sequence_stage(sequence: dict) -> str:
    for stage in sequence.get("stages", []):
        if (
            not stage["plan"].get("success", False)
            or stage["validation"].get("decision") != "ALLOW"
        ):
            return str(stage.get("stage"))
    return "sequence"

def _failed_sequence_validation(sequence: dict) -> dict:
    failed_stage = _failed_sequence_stage(sequence)
    reasons = [f"Sequence planning failed at {failed_stage}."]
    for stage in sequence.get("stages", []):
        if str(stage.get("stage")) != failed_stage:
            continue
        plan = stage.get("plan", {})
        if plan.get("message"):
            reasons.append(str(plan.get("message")))
        reasons.extend(stage.get("validation", {}).get("reasons") or [])
        break
    return {"decision": "BLOCK", "reasons": reasons, "warnings": []}

def plan_pose_sequence(
    backend,
    poses: dict[str, dict],
    trajectory_validator,
    velocity_scale: float = 0.05,
    acceleration_scale: float = 0.05,
    plan_kwargs: dict | None = None,
    stage_plan_kwargs: dict[str, dict] | None = None,
    max_replanning_attempts: int = 1,
) -> dict:
    """Plan and validate three segments; this function has no execution call."""
    results = []
    start_state = None
    plan_kwargs = plan_kwargs or {}
    stage_plan_kwargs = stage_plan_kwargs or {}
    attempts_per_stage = max(1, int(max_replanning_attempts))
    for step_id, label in enumerate(("pregrasp", "grasp", "retreat"), start=1):
        pose = poses[label]
        current_plan_kwargs = {
            **plan_kwargs,
            **(stage_plan_kwargs.get(label) or {}),
        }
        attempts = []
        for attempt_index in range(1, attempts_per_stage + 1):
            result = backend.plan_to_pose(
                frame=pose["frame"],
                position=pose["position"],
                orientation=pose["orientation"],
                velocity_scale=velocity_scale,
                acceleration_scale=acceleration_scale,
                skill_id=f"plan_{label}_pose",
                step_id=step_id * 100 + attempt_index,
                start_state=start_state,
                **current_plan_kwargs,
            )
            validation = trajectory_validator.validate_motion_plan_summary(
                result["summary"]
            )
            attempts.append(
                {
                    "stage": label,
                    "plan": result,
                    "validation": validation,
                    "attempt_index": attempt_index,
                }
            )
        accepted = [
            attempt
            for attempt in attempts
            if attempt["plan"].get("success", False)
            and attempt["validation"].get("decision") == "ALLOW"
        ]
        selected_stage = sorted(accepted or attempts, key=_plan_attempt_ranking_key)[0]
        selected_stage["attempts"] = attempts
        results.append(selected_stage)
        if (
            not selected_stage["plan"].get("success", False)
            or selected_stage["validation"].get("decision") != "ALLOW"
        ):
            break
        end_state_getter = getattr(backend, "get_plan_end_state", None)
        start_state = (
            end_state_getter(selected_stage["plan"]["plan_id"])
            if callable(end_state_getter)
            else None
        )
    return {
        "success": len(results) == 3 and all(
            item["plan"].get("success", False)
            and item["validation"].get("decision") == "ALLOW"
            for item in results
        ),
        "stages": results,
    }

def _plan_attempt_ranking_key(attempt: dict) -> tuple:
    summary = attempt.get("plan", {}).get("summary", {}) or {}
    max_delta = summary.get("max_joint_delta_rad")
    return (
        float(max_delta) if max_delta is not None else float("inf"),
        int(summary.get("num_trajectory_points") or 999999),
    )

def _is_transform_matrix(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 4
        and all(isinstance(row, (list, tuple)) and len(row) == 4 for row in value)
    )

def _matrix_to_quaternion(matrix) -> dict[str, float]:
    trace = float(matrix[0][0]) + float(matrix[1][1]) + float(matrix[2][2])
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (float(matrix[2][1]) - float(matrix[1][2])) / scale
        y = (float(matrix[0][2]) - float(matrix[2][0])) / scale
        z = (float(matrix[1][0]) - float(matrix[0][1])) / scale
    else:
        diagonal = [float(matrix[index][index]) for index in range(3)]
        index = diagonal.index(max(diagonal))
        if index == 0:
            scale = math.sqrt(1.0 + diagonal[0] - diagonal[1] - diagonal[2]) * 2.0
            x, y, z, w = 0.25 * scale, (matrix[0][1] + matrix[1][0]) / scale, (matrix[0][2] + matrix[2][0]) / scale, (matrix[2][1] - matrix[1][2]) / scale
        elif index == 1:
            scale = math.sqrt(1.0 + diagonal[1] - diagonal[0] - diagonal[2]) * 2.0
            x, y, z, w = (matrix[0][1] + matrix[1][0]) / scale, 0.25 * scale, (matrix[1][2] + matrix[2][1]) / scale, (matrix[0][2] - matrix[2][0]) / scale
        else:
            scale = math.sqrt(1.0 + diagonal[2] - diagonal[0] - diagonal[1]) * 2.0
            x, y, z, w = (matrix[0][2] + matrix[2][0]) / scale, (matrix[1][2] + matrix[2][1]) / scale, 0.25 * scale, (matrix[1][0] - matrix[0][1]) / scale
    return {"x": float(x), "y": float(y), "z": float(z), "w": float(w)}

def _normalized_vector(value: Any) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError("Approach/retreat vector must contain three values.")
    vector = [float(component) for component in value]
    magnitude = math.sqrt(sum(component * component for component in vector))
    if not math.isfinite(magnitude) or magnitude <= 0.0:
        raise ValueError("Approach/retreat vector must be finite and non-zero.")
    return [component / magnitude for component in vector]

def _rotate_local_z(quaternion: dict) -> list[float]:
    x, y, z, w = (float(quaternion[key]) for key in ("x", "y", "z", "w"))
    return [2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)]

def _apply_local_pose_transform(pose: dict, config: dict) -> dict:
    translation = _three_float_config(config, "translation_m", [0.0, 0.0, 0.0])
    rotation_rpy = _three_float_config(config, "rotation_rpy_rad", [0.0, 0.0, 0.0])
    pose_quaternion = _normalize_quaternion_dict(pose["orientation"])
    rotated_translation = _rotate_vector_by_quaternion(translation, pose_quaternion)
    offset_quaternion = _rpy_to_quaternion(*rotation_rpy)
    output_quaternion = _multiply_quaternions(pose_quaternion, offset_quaternion)
    return {
        "frame": pose["frame"],
        "position": {
            axis: float(pose["position"][axis]) + rotated_translation[index]
            for index, axis in enumerate(("x", "y", "z"))
        },
        "orientation": _quaternion_tuple_to_dict(output_quaternion),
    }

def _three_float_config(config: dict, key: str, default: list[float]) -> list[float]:
    value = config.get(key, default)
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{key} must contain exactly three values.")
    vector = [float(component) for component in value]
    if not all(math.isfinite(component) for component in vector):
        raise ValueError(f"{key} values must be finite.")
    return vector

def _normalize_quaternion_dict(quaternion: dict) -> tuple[float, float, float, float]:
    values = tuple(float(quaternion[key]) for key in ("x", "y", "z", "w"))
    norm = math.sqrt(sum(component * component for component in values))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("Pose orientation quaternion must be finite and non-zero.")
    return tuple(component / norm for component in values)

def _quaternion_tuple_to_dict(
    quaternion: tuple[float, float, float, float]
) -> dict[str, float]:
    x, y, z, w = _normalize_quaternion_dict(
        {"x": quaternion[0], "y": quaternion[1], "z": quaternion[2], "w": quaternion[3]}
    )
    return {"x": x, "y": y, "z": z, "w": w}

def _multiply_quaternions(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return (
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    )

def _rpy_to_quaternion(
    roll: float,
    pitch: float,
    yaw: float,
) -> tuple[float, float, float, float]:
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return _normalize_quaternion_dict(
        {
            "x": sr * cp * cy - cr * sp * sy,
            "y": cr * sp * cy + sr * cp * sy,
            "z": cr * cp * sy - sr * sp * cy,
            "w": cr * cp * cy + sr * sp * sy,
        }
    )

def _rotate_vector_by_quaternion(
    vector: list[float],
    quaternion: tuple[float, float, float, float],
) -> list[float]:
    q_vector = (float(vector[0]), float(vector[1]), float(vector[2]), 0.0)
    q_conjugate = (-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3])
    rotated = _multiply_quaternions(
        _multiply_quaternions(quaternion, q_vector),
        q_conjugate,
    )
    return [rotated[0], rotated[1], rotated[2]]

def _offset_pose(pose: dict, vector: list[float], distance: float) -> dict:
    return {
        "frame": pose["frame"],
        "position": {
            axis: float(pose["position"][axis]) + vector[index] * distance
            for index, axis in enumerate(("x", "y", "z"))
        },
        "orientation": dict(pose["orientation"]),
    }
