"""GraspGenX MoveIt planning and RViz preview with real-arm execution blocked."""

from __future__ import annotations

import math
import pprint
import sys
import time
from typing import Any

from llm_skill_robot.core.trajectory_validator import TrajectoryValidator
from llm_skill_robot.core.trajectory_visualizer import TrajectoryVisualizer
from llm_skill_robot.grasping.graspgenx_grasp_planner import GraspGenXGraspPlanner
from llm_skill_robot.grasping.graspgenx_subprocess_adapter import GraspGenXSubprocessAdapter
from llm_skill_robot.perception.object_point_cloud_builder import ObjectPointCloudBuilder
from llm_skill_robot.perception.perception_factory import create_perception_pipeline
from llm_skill_robot.perception.tf_pose_transformer import TFPoseTransformer
from llm_skill_robot.robot.ur5_moveit_plan_backend import UR5MoveItPlanBackend
from llm_skill_robot.ros_graspgenx_gripper_width_test import (
    _candidate_value,
    _load_yaml,
    _select_highest_confidence,
)
from llm_skill_robot.ros_real_gripper_test import confirmation_is_yes
from llm_skill_robot.safety.hardware_mode import (
    HardwareMode,
    assert_arm_execution_allowed,
    load_hardware_mode,
)


ARM_BLOCKED_MESSAGE = "REAL ARM EXECUTION BLOCKED BY hardware_mode.yaml"


def wait_for_camera_messages(camera_backend, node, rclpy_module) -> dict:
    """Wait until the D435i backend has RGB, depth, and CameraInfo cached."""
    base = float(getattr(camera_backend, "config", {}).get("wait_timeout_sec", 5.0))
    timeout_sec = max(base * 3.0, 5.0)
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        status = camera_message_status(camera_backend)
        if (
            status["rgb_received"]
            and status["depth_received"]
            and status["camera_info_received"]
        ):
            return status
        rclpy_module.spin_once(node, timeout_sec=0.1)
    return camera_message_status(camera_backend)


def camera_message_status(camera_backend) -> dict[str, bool]:
    return {
        "rgb_received": getattr(camera_backend, "latest_rgb_msg", None) is not None,
        "depth_received": getattr(camera_backend, "latest_depth_msg", None) is not None,
        "camera_info_received": getattr(camera_backend, "latest_camera_info_msg", None)
        is not None,
    }


def print_camera_status(status: dict[str, bool]) -> None:
    print("Camera status:")
    print(f"RGB received: {status['rgb_received']}")
    print(f"Depth received: {status['depth_received']}")
    print(f"CameraInfo received: {status['camera_info_received']}")


def print_detection_debug(detection: Any) -> None:
    """Print perception result details before selecting the best object."""
    status = getattr(detection, "status", None) or (
        detection.get("status") if isinstance(detection, dict) else None
    )
    message = getattr(detection, "message", None) or (
        detection.get("message") if isinstance(detection, dict) else None
    )
    if hasattr(detection, "objects"):
        objects = getattr(detection, "objects") or []
    elif isinstance(detection, dict):
        objects = detection.get("objects", []) or []
    else:
        objects = []

    print(f"Detector status: {status}")
    print(f"Detector message: {message}")
    print(f"Raw objects count: {len(objects)}")
    for index, obj in enumerate(objects, start=1):
        if hasattr(obj, "metadata"):
            metadata = getattr(obj, "metadata", {}) or {}
            object_id = getattr(obj, "object_id", None)
            confidence = getattr(obj, "confidence", None)
        elif isinstance(obj, dict):
            metadata = obj.get("metadata", {}) or {}
            object_id = obj.get("object_id")
            confidence = obj.get("confidence")
        else:
            metadata = {}
            object_id = None
            confidence = None
        print(
            f"  Object {index}: id={object_id} confidence={confidence} "
            f"mask={metadata.get('mask_path')} overlay={metadata.get('overlay_path')}"
        )


def assert_plan_only_mode(mode: HardwareMode) -> None:
    if mode.arm.allow_arm_execution is not False or mode.arm.real_arm_enabled is not False:
        raise PermissionError("HW4 requires real arm execution to be disabled.")
    if not mode.arm.allow_plan_only:
        raise PermissionError("Plan-only arm requests are disabled by hardware_mode.yaml.")


def attempt_real_arm_execution(mode: HardwareMode, confirmed: bool = False) -> None:
    """Central fail-closed hook for any accidental execution attempt."""
    if not confirmed:
        raise PermissionError("Real arm execution was not confirmed.")
    assert_arm_execution_allowed(mode)
    raise PermissionError("HW4 never executes real arm trajectories.")


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


def describe_grasp_frame_to_ee_transform(config: dict | None = None) -> str:
    config = config or {}
    if not config.get("enabled", False):
        return "disabled; using GraspGenX grasp frame directly as MoveIt EE target"
    return (
        "enabled; T_base_ee = T_base_grasp * T_grasp_ee, "
        f"translation_m={config.get('translation_m', [0.0, 0.0, 0.0])}, "
        f"rotation_rpy_rad={config.get('rotation_rpy_rad', [0.0, 0.0, 0.0])}"
    )


def prepare_candidate_preview(
    candidate: Any,
    point_cloud: dict,
    preview_config: dict | None,
    target_frame: str,
    transformer: TFPoseTransformer | None,
) -> dict:
    """Build localized, TF-transformed MoveIt EE preview poses for a candidate."""
    preview_config = preview_config or {}
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
    return apply_grasp_frame_to_ee_transform(
        poses,
        preview_config.get("grasp_frame_to_ee_frame", {}),
    )


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
    valid_candidates = sorted(
        valid_candidates,
        key=lambda candidate: float(_candidate_value(candidate, "score", 0.0) or 0.0),
        reverse=True,
    )[:max_candidates]

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

        for variant in variants:
            kwargs = {
                **base_plan_kwargs,
                **planning_variant_kwargs(variant, screening_config),
            }
            result = backend.plan_to_pose(
                frame=poses["pregrasp"]["frame"],
                position=poses["pregrasp"]["position"],
                orientation=poses["pregrasp"]["orientation"],
                velocity_scale=velocity_scale,
                acceleration_scale=acceleration_scale,
                skill_id=f"screen_pregrasp_{variant}",
                step_id=2000 + len(attempts) + 1,
                **kwargs,
            )
            summary = result.get("summary", {})
            validation = trajectory_validator.validate_motion_plan_summary(summary)
            attempt = {
                "candidate": candidate,
                "candidate_id": candidate_id,
                "score": candidate_score,
                "poses": poses,
                "variant": variant,
                "plan_kwargs": kwargs,
                "plan": result,
                "validation": validation,
                "accepted": bool(result.get("success"))
                and validation.get("decision") == "ALLOW",
                "message": result.get("message"),
                "points": int(summary.get("num_trajectory_points") or 0),
                "max_joint_delta_rad": summary.get("max_joint_delta_rad"),
            }
            attempts.append(attempt)
            if attempt["accepted"]:
                if variant == "position_only":
                    accepted_position_only.append(attempt)
                else:
                    accepted_strict.append(attempt)

    selected_attempts = accepted_strict or accepted_position_only
    if not selected_attempts:
        return {"selected": None, "poses": None, "variant": None, "attempts": attempts}

    def ranking_key(attempt: dict) -> tuple:
        max_delta = attempt.get("max_joint_delta_rad")
        return (
            float(max_delta) if max_delta is not None else float("inf"),
            int(attempt.get("points") or 999999),
            -float(attempt.get("score") or 0.0),
        )

    selected_attempt = sorted(selected_attempts, key=ranking_key)[0]
    return {
        "selected": selected_attempt["candidate"],
        "poses": selected_attempt["poses"],
        "variant": selected_attempt["variant"],
        "plan_kwargs": selected_attempt["plan_kwargs"],
        "attempt": selected_attempt,
        "attempts": attempts,
    }


def plan_pose_sequence(
    backend,
    poses: dict[str, dict],
    trajectory_validator,
    velocity_scale: float = 0.05,
    acceleration_scale: float = 0.05,
    plan_kwargs: dict | None = None,
) -> dict:
    """Plan and validate three segments; this function has no execution call."""
    results = []
    start_state = None
    plan_kwargs = plan_kwargs or {}
    for step_id, label in enumerate(("pregrasp", "grasp", "retreat"), start=1):
        pose = poses[label]
        result = backend.plan_to_pose(
            frame=pose["frame"],
            position=pose["position"],
            orientation=pose["orientation"],
            velocity_scale=velocity_scale,
            acceleration_scale=acceleration_scale,
            skill_id=f"plan_{label}_pose",
            step_id=step_id,
            start_state=start_state,
            **plan_kwargs,
        )
        validation = trajectory_validator.validate_motion_plan_summary(result["summary"])
        results.append({"stage": label, "plan": result, "validation": validation})
        if not result.get("success", False) or validation.get("decision") != "ALLOW":
            break
        end_state_getter = getattr(backend, "get_plan_end_state", None)
        start_state = end_state_getter(result["plan_id"]) if callable(end_state_getter) else None
    return {
        "success": len(results) == 3 and all(
            item["plan"].get("success", False)
            and item["validation"].get("decision") == "ALLOW"
            for item in results
        ),
        "stages": results,
    }


def diagnose_pregrasp_planning(
    backend,
    pose: dict,
    trajectory_validator,
    velocity_scale: float = 0.05,
    acceleration_scale: float = 0.05,
    plan_kwargs: dict | None = None,
) -> list[dict]:
    """Run non-executing MoveIt plan variants to isolate pose failure causes."""
    plan_kwargs = plan_kwargs or {}
    variants = [
        {
            "name": "full_pose",
            "description": "position + GraspGenX orientation",
            "kwargs": {"position_tolerance": 0.005, "orientation_tolerance": 0.05},
        },
        {
            "name": "relaxed_orientation",
            "description": "position + loose orientation tolerance",
            "kwargs": {"position_tolerance": 0.01, "orientation_tolerance": 0.75},
        },
        {
            "name": "position_only",
            "description": "position constraint only; no orientation constraint",
            "kwargs": {
                "position_tolerance": 0.01,
                "use_orientation_constraint": False,
            },
        },
    ]
    results = []
    for index, variant in enumerate(variants, start=1):
        result = backend.plan_to_pose(
            frame=pose["frame"],
            position=pose["position"],
            orientation=pose["orientation"],
            velocity_scale=velocity_scale,
            acceleration_scale=acceleration_scale,
            skill_id=f"diagnose_pregrasp_{variant['name']}",
            step_id=100 + index,
            **plan_kwargs,
            **variant["kwargs"],
        )
        validation = trajectory_validator.validate_motion_plan_summary(
            result["summary"]
        )
        results.append(
            {
                "variant": variant["name"],
                "description": variant["description"],
                "plan": result,
                "validation": validation,
            }
        )
    return results


def _publish_sequence_preview(backend, sequence: dict, visualizer) -> dict:
    entries = []
    for stage in sequence.get("stages", []):
        plan_id = stage["plan"].get("plan_id")
        entry = backend.get_plan(plan_id) if plan_id else None
        if entry is not None:
            entries.append(entry)
    return visualizer.publish_sequence(entries)


def _print_plan_results(sequence: dict) -> None:
    for stage in sequence.get("stages", []):
        plan = stage["plan"]
        print(f"{stage['stage']} planning success/failure: {'SUCCESS' if plan.get('success') else 'FAILURE'}")
        summary = plan.get("summary", {})
        print(f"{stage['stage']} planning message: {plan.get('message')}")
        print(f"{stage['stage']} trajectory points: {summary.get('num_trajectory_points')}")
        diagnostics = (summary.get("trajectory_preview", {}) or {}).get(
            "moveit_diagnostics"
        )
        if diagnostics:
            print(f"{stage['stage']} MoveIt diagnostics:")
            pprint.pprint(diagnostics, sort_dicts=False)
        print(f"{stage['stage']} trajectory validation result:")
        pprint.pprint(stage["validation"], sort_dicts=False)


def _print_pregrasp_diagnostics(results: list[dict]) -> None:
    print("Pregrasp planning diagnostics:")
    for item in results:
        plan = item["plan"]
        summary = plan.get("summary", {})
        diagnostics = (summary.get("trajectory_preview", {}) or {}).get(
            "moveit_diagnostics", {}
        )
        print(
            f"- {item['variant']}: "
            f"{'SUCCESS' if plan.get('success') else 'FAILURE'} "
            f"({item['description']})"
        )
        print(f"  message: {plan.get('message')}")
        print(f"  points: {summary.get('num_trajectory_points')}")
        if diagnostics:
            print(
                "  moveit: "
                f"{diagnostics.get('error_code_name')} "
                f"({diagnostics.get('error_code_value')})"
            )
        if item["validation"].get("reasons"):
            print(f"  validation: {item['validation'].get('reasons')}")


def _print_candidate_screening(screening: dict) -> None:
    attempts = screening.get("attempts", [])
    if not attempts:
        print("MoveIt candidate screening: no attempts were made.")
        return

    print("MoveIt candidate screening:")
    for item in attempts:
        reasons = item.get("validation", {}).get("reasons") or []
        status = "ALLOW" if item.get("accepted") else "BLOCK"
        details = []
        if item.get("points") is not None:
            details.append(f"points={item.get('points')}")
        if item.get("max_joint_delta_rad") is not None:
            details.append(f"max_joint_delta={item.get('max_joint_delta_rad')}")
        if reasons:
            details.append(f"reasons={reasons}")
        print(
            f"- {item.get('candidate_id')} score={item.get('score'):.3f} "
            f"variant={item.get('variant')}: {status}"
            + (f" ({'; '.join(details)})" if details else "")
        )

    selected = screening.get("attempt")
    if selected:
        print(
            "Selected MoveIt-reachable candidate: "
            f"{selected.get('candidate_id')} "
            f"variant={selected.get('variant')} "
            f"score={selected.get('score'):.3f} "
            f"points={selected.get('points')} "
            f"max_joint_delta={selected.get('max_joint_delta_rad')}"
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


def main() -> None:
    try:
        import rclpy
    except ImportError:
        print("Error: rclpy not available. Please source ROS 2 setup.", file=sys.stderr)
        return

    hardware_mode = load_hardware_mode()
    try:
        assert_plan_only_mode(hardware_mode)
    except PermissionError as exc:
        print(f"Safety error: {exc}")
        return
    print(ARM_BLOCKED_MESSAGE)

    perception_config = _load_yaml("perception.yaml")
    grasping_config = _load_yaml("grasping.yaml")
    safety_config = _load_yaml("safety.yaml")
    grasping = grasping_config.get("grasping", {})
    preview_config = grasping.get("plan_only_preview", {})
    node = None
    backend = None
    try:
        rclpy.init()
        node = rclpy.create_node("ros_graspgenx_plan_only_demo")
        perception_pipeline = create_perception_pipeline(node=node, config=perception_config)
        point_cloud_builder = ObjectPointCloudBuilder(grasping.get("point_cloud", {}))
        planner = GraspGenXGraspPlanner(
            GraspGenXSubprocessAdapter(grasping.get("graspgenx_subprocess", {}))
        )
        backend = UR5MoveItPlanBackend(
            node=node,
            dry_run=False,
            execution_mode="plan_only",
            hardware_mode=hardware_mode,
        )
        validator = TrajectoryValidator(safety_config)
        visualizer = TrajectoryVisualizer(node=node)

        query = input("Enter object query:\n> ").strip()
        if not query:
            print("No object query entered.")
            return
        print("\nWaiting for D435i RGB/depth/camera_info messages...")
        camera_backend = perception_pipeline.backend
        camera_status = wait_for_camera_messages(camera_backend, node, rclpy)
        print_camera_status(camera_status)
        if not (
            camera_status["rgb_received"]
            and camera_status["depth_received"]
            and camera_status["camera_info_received"]
        ):
            print(
                "No point cloud. Aligned depth frame and camera info are required. "
                "Check RealSense topics and align_depth.enable:=true."
            )
            return
        detection = perception_pipeline.detect_object(query, require_pose=False)
        print_detection_debug(detection)
        selected_object = _select_highest_confidence(detection.get("objects", []))
        if selected_object is None:
            print("No object detected.")
            return
        depth_getter = getattr(camera_backend, "_get_latest_depth_image", None)
        point_cloud = point_cloud_builder.build(
            selected_object,
            depth_getter() if callable(depth_getter) else None,
            getattr(camera_backend, "latest_camera_info_msg", None),
            frame=getattr(camera_backend, "config", {}).get("depth_frame"),
        )
        if not point_cloud.get("success", False):
            print(f"No point cloud. {point_cloud.get('message', '')}".strip())
            return
        graspgenx_config = grasping.get("graspgenx_subprocess", {})
        if graspgenx_config.get("require_server_confirmation", False):
            print(
                "\nObject point cloud is ready. If you are using an 8GB GPU, "
                "start or resume the GraspGenX server now so SAM3 and GraspGenX "
                "do not load on CUDA at the same time."
            )
            review_server = input(
                "Confirm GraspGenX server is ready. Type YES to generate candidates: "
            )
            if not confirmation_is_yes(review_server):
                print("GraspGenX generation not approved. Stopped safely.")
                return
        generation = planner.generate_candidates(point_cloud["point_cloud_path"])
        print(f"GraspGenX status: {generation.get('status')}")
        if generation.get("message"):
            print(f"GraspGenX message: {generation.get('message')}")
        candidates = generation.get("candidates", [])
        print(f"GraspGenX candidates returned: {len(candidates)}")
        best_candidate = planner.select_best_candidate(candidates)
        if best_candidate is None:
            print("No valid GraspGenX candidates.")
            return

        target_frame = perception_config.get("perception", {}).get("target_frame", "base_link")
        transformer = None
        if target_frame:
            transformer = getattr(camera_backend, "pose_transformer", None)
            if transformer is None or getattr(transformer, "target_frame", None) != target_frame:
                transformer = TFPoseTransformer(node, target_frame=target_frame)
        ee_transform_config = preview_config.get("grasp_frame_to_ee_frame", {})
        print(
            "Grasp frame -> MoveIt EE transform: "
            f"{describe_grasp_frame_to_ee_transform(ee_transform_config)}"
        )
        base_moveit_kwargs = moveit_planning_kwargs(preview_config)
        if base_moveit_kwargs:
            print("MoveIt plan-only request tuning:")
            pprint.pprint(base_moveit_kwargs, sort_dicts=False)

        velocity_scale = float(preview_config.get("velocity_scale", 0.05))
        acceleration_scale = float(preview_config.get("acceleration_scale", 0.05))
        screening = screen_grasp_candidates_for_plan_only(
            candidates,
            point_cloud,
            preview_config,
            target_frame,
            transformer,
            backend,
            validator,
            velocity_scale=velocity_scale,
            acceleration_scale=acceleration_scale,
        )
        _print_candidate_screening(screening)
        selected = screening.get("selected")
        poses = screening.get("poses")
        selected_plan_kwargs = screening.get("plan_kwargs") or {}
        selected_variant = screening.get("variant")
        if selected is None or poses is None:
            print(
                "No MoveIt-reachable GraspGenX candidate passed trajectory validation. "
                "Full RViz sequence preview was not requested."
            )
            print(ARM_BLOCKED_MESSAGE)
            print("No arm trajectory was executed. No gripper command was sent.")
            return

        print(f"selected candidate score: {_candidate_value(selected, 'score')}")
        print(f"selected MoveIt planning variant: {selected_variant}")
        if selected_variant == "position_only":
            print(
                "Warning: selected preview uses position-only constraints; "
                "the exact GraspGenX gripper orientation is not enforced."
            )
        print(f"gripper_width_m: {_candidate_value(selected, 'gripper_width_m')}")
        print("pregrasp pose:")
        pprint.pprint(poses["pregrasp"], sort_dicts=False)
        print("grasp pose:")
        pprint.pprint(poses["grasp"], sort_dicts=False)
        print("retreat pose:")
        pprint.pprint(poses["retreat"], sort_dicts=False)
        review = input("Review poses. Type YES to request plan-only RViz preview: ")
        if not confirmation_is_yes(review):
            print("Review not approved. No MoveIt plans requested.")
            return

        diagnostic_results = diagnose_pregrasp_planning(
            backend,
            poses["pregrasp"],
            validator,
            velocity_scale=velocity_scale,
            acceleration_scale=acceleration_scale,
            plan_kwargs=base_moveit_kwargs,
        )
        _print_pregrasp_diagnostics(diagnostic_results)

        sequence = plan_pose_sequence(
            backend,
            poses,
            validator,
            velocity_scale=velocity_scale,
            acceleration_scale=acceleration_scale,
            plan_kwargs=selected_plan_kwargs,
        )
        _print_plan_results(sequence)
        if sequence["success"]:
            preview = _publish_sequence_preview(backend, sequence, visualizer)
            print(f"RViz preview: {preview.get('message')}")
        else:
            print("Planning or trajectory validation failed; RViz sequence preview was not published.")
        print(ARM_BLOCKED_MESSAGE)
        print("No arm trajectory was executed. No gripper command was sent.")
    except (KeyboardInterrupt, ValueError) as exc:
        print(f"Stopped safely: {exc}")
    finally:
        if backend is not None:
            backend.shutdown()
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
