#!/usr/bin/env python3
"""Run GraspGenX ZMQ inference and print LLM_Ros candidate JSON.

This script is intentionally small and side-effect free: it does not start the
GraspGenX model server. Start the server in the ``graspgenx`` environment first,
then this runner connects to it, sends a point cloud, and writes a JSON object to
stdout for :class:`GraspGenXSubprocessAdapter`.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Connect to a GraspGenX ZMQ server and return grasp candidates as JSON."
    )
    parser.add_argument("--point-cloud", required=True, help="Input .ply/.xyz/.npy point cloud.")
    parser.add_argument(
        "--scene-point-cloud", default=None,
        help="Environment-only scene cloud in the same source frame; target mask must be removed.",
    )
    parser.add_argument("--collision-threshold-m", type=float, default=0.006)
    parser.add_argument("--collision-samples", type=int, default=2500)
    parser.add_argument("--collision-batch-size", type=int, default=4)
    parser.add_argument(
        "--collision-closed-finger-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--collision-closed-finger-samples", type=int, default=3500)
    parser.add_argument("--collision-closed-finger-threshold-m", type=float, default=0.0065)
    parser.add_argument("--collision-closed-finger-backoff-m", type=float, default=0.040)
    parser.add_argument("--environment-constrained-generation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--constraint-min-backoff-m", type=float, default=0.025)
    parser.add_argument("--constraint-max-backoff-m", type=float, default=0.050)
    parser.add_argument("--constraint-backoff-step-m", type=float, default=0.0025)
    parser.add_argument("--constraint-min-target-contact-points", type=int, default=12)
    parser.add_argument("--constraint-min-target-contact-fraction", type=float, default=0.20)
    parser.add_argument("--constraint-pad-contact-depth-m", type=float, default=0.045)
    parser.add_argument("--constraint-support-plane-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--constraint-support-plane-margin-m", type=float, default=0.012)
    parser.add_argument("--constraint-support-plane-clearance-m", type=float, default=0.011)
    parser.add_argument("--constraint-support-plane-inlier-threshold-m", type=float, default=0.003)
    parser.add_argument("--constraint-support-plane-max-tilt-deg", type=float, default=20.0)
    parser.add_argument("--constraint-support-plane-min-inliers", type=int, default=24)
    parser.add_argument("--constraint-pregrasp-distance-m", type=float, default=0.060)
    parser.add_argument("--constraint-retreat-distance-m", type=float, default=0.100)
    parser.add_argument("--constraint-path-samples", type=int, default=5)
    parser.add_argument("--constraint-path-surface-samples", type=int, default=1800)
    parser.add_argument("--constraint-yaw-bin-deg", type=float, default=15.0)
    parser.add_argument("--constraint-max-per-yaw-bin", type=int, default=2)
    parser.add_argument("--constraint-min-obb-candidates", type=int, default=12)
    parser.add_argument(
        "--collision-fallback-to-object-only",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--output-dir", required=True, help="Directory for optional/debug outputs.")
    parser.add_argument("--host", default="localhost", help="GraspGenX server host.")
    parser.add_argument("--port", type=int, default=5556, help="GraspGenX server port.")
    parser.add_argument(
        "--gripper-name",
        default="robotiq_2f_140",
        help="GraspGenX gripper name, e.g. robotiq_2f_140.",
    )
    parser.add_argument("--num-grasps", type=int, default=200)
    parser.add_argument("--grasp-threshold", type=float, default=-1.0)
    parser.add_argument("--topk-num-grasps", type=int, default=100)
    parser.add_argument("--timeout-ms", type=int, default=60000)
    parser.add_argument("--strategy", choices=["diffusion", "sparse"], default="diffusion")
    parser.add_argument("--sparse-quota", type=int, default=24)
    parser.add_argument("--moe-num-yaws", type=int, default=36)
    parser.add_argument("--moe-z-offsets-cm", default="0")
    parser.add_argument(
        "--default-gripper-width-m",
        type=float,
        default=None,
        help=(
            "Optional width to attach to every candidate. GraspGenX ZMQ returns "
            "poses/scores, not Robotiq aperture widths."
        ),
    )
    parser.add_argument(
        "--estimate-gripper-width",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Estimate gripper_width_m from point-cloud extent along the candidate "
            "gripper opening axis when GraspGenX does not provide a width."
        ),
    )
    parser.add_argument(
        "--width-axis-index",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Candidate rotation-matrix column used as gripper opening axis.",
    )
    parser.add_argument(
        "--width-clearance-m",
        type=float,
        default=0.01,
        help="Clearance added to estimated object width.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        point_cloud, point_cloud_metadata = load_point_cloud_with_metadata(
            args.point_cloud
        )
        collision_metadata = {
            "requested": bool(args.scene_point_cloud),
            "applied": False,
            "fallback_used": False,
        }
        with redirect_stdout(sys.stderr):
            from graspgenx.serving.zmq_client import GraspGenXClient

            with GraspGenXClient(
                host=args.host, port=args.port, timeout_ms=args.timeout_ms
            ) as client:
                metadata = client.server_metadata
                strategy_used = "diffusion"
                sources = None
                if args.strategy == "sparse" and "infer_sparse" in metadata.get("supported_actions", []):
                    response = client._request({
                        "action": "infer_sparse",
                        "point_cloud": point_cloud,
                        "gripper_name": args.gripper_name,
                        "num_grasps": args.num_grasps,
                        "grasp_threshold": args.grasp_threshold,
                        "topk_num_grasps": args.topk_num_grasps,
                        "sparse_quota": args.sparse_quota,
                        "moe_num_yaws": args.moe_num_yaws,
                        "moe_z_offsets_cm": [float(x) for x in args.moe_z_offsets_cm.split(",")],
                    })
                    grasps = response["grasps"]
                    confidences = response["confidences"]
                    sources = response["sources"]
                    strategy_used = "sparse"
                else:
                    grasps, confidences = client.infer(
                        point_cloud,
                        gripper_name=args.gripper_name,
                        num_grasps=args.num_grasps,
                        grasp_threshold=args.grasp_threshold,
                        topk_num_grasps=args.topk_num_grasps,
                    )
            raw_grasp_array = np.asarray(grasps, dtype=np.float32)
            raw_score_array = np.asarray(confidences, dtype=np.float32).reshape(-1)
            raw_sources = (
                list(sources) if sources is not None
                else ["diffusion"] * raw_grasp_array.shape[0]
            )
            candidate_lifecycle = initialize_candidate_lifecycle(
                raw_grasp_array, raw_score_array, raw_sources,
                width_axis_index=args.width_axis_index,
            )
            source_indices = np.arange(raw_grasp_array.shape[0], dtype=int)
            if args.scene_point_cloud:
                try:
                    from graspgenx.utils.collision_filter import filter_colliding_grasps
                    from graspgenx.x_grippers import resolve_gripper_info

                    environment_points = load_point_cloud_uncentered(args.scene_point_cloud)
                    object_centroid = point_cloud_metadata["original_centroid_m"]
                    centroid_vector = np.asarray([
                        object_centroid["x"], object_centroid["y"], object_centroid["z"],
                    ], dtype=np.float32)
                    centered_environment = environment_points - centroid_vector
                    residual_points_removed = 0
                    if strategy_used == "sparse":
                        centered_environment, residual_points_removed = remove_target_residuals(
                            centered_environment, point_cloud,
                        )
                    grasp_array = np.asarray(grasps, dtype=np.float32)
                    score_array = np.asarray(confidences, dtype=np.float32).reshape(-1)
                    gripper = resolve_gripper_info(args.gripper_name)
                    if args.environment_constrained_generation and strategy_used == "sparse":
                        constrained = constrain_grasps_to_environment(
                            filter_colliding_grasps,
                            centered_environment,
                            point_cloud,
                            grasp_array,
                            gripper,
                            static_threshold=args.collision_threshold_m,
                            static_samples=args.collision_samples,
                            static_batch_size=args.collision_batch_size,
                            closed_threshold=args.collision_closed_finger_threshold_m,
                            closed_samples=args.collision_closed_finger_samples,
                            closed_enabled=args.collision_closed_finger_enabled,
                            min_backoff_m=args.constraint_min_backoff_m,
                            max_backoff_m=args.constraint_max_backoff_m,
                            backoff_step_m=args.constraint_backoff_step_m,
                            min_target_contact_points=args.constraint_min_target_contact_points,
                            min_target_contact_fraction=args.constraint_min_target_contact_fraction,
                            pad_contact_depth_m=args.constraint_pad_contact_depth_m,
                            support_plane_enabled=args.constraint_support_plane_enabled,
                            support_plane_margin_m=args.constraint_support_plane_margin_m,
                            support_plane_clearance_m=args.constraint_support_plane_clearance_m,
                            support_plane_inlier_threshold_m=args.constraint_support_plane_inlier_threshold_m,
                            support_plane_max_tilt_deg=args.constraint_support_plane_max_tilt_deg,
                            support_plane_min_inliers=args.constraint_support_plane_min_inliers,
                            pregrasp_distance_m=args.constraint_pregrasp_distance_m,
                            retreat_distance_m=args.constraint_retreat_distance_m,
                            path_samples=args.constraint_path_samples,
                            path_surface_samples=args.constraint_path_surface_samples,
                        )
                        mask = constrained["mask"]
                        selected_backoffs = constrained["selected_backoffs_m"]
                        trajectory_specs = constrained["trajectory_specs"]
                        static_mask = constrained["static_final_mask"]
                        closed_mask = constrained["closed_final_mask"]
                        for item, stages in zip(candidate_lifecycle, constrained["candidate_stages"]):
                            item["environment"] = stages
                    else:
                        collision_pose_array = apply_grasp_backoff(
                            grasp_array, args.collision_closed_finger_backoff_m,
                        )
                        static_mask = np.asarray(filter_colliding_grasps(
                            scene_pc=centered_environment,
                            grasp_poses=collision_pose_array,
                            gripper_collision_mesh=gripper.collision_mesh,
                            collision_threshold=args.collision_threshold_m,
                            num_collision_samples=args.collision_samples,
                            batch_size=args.collision_batch_size,
                        ), dtype=bool)
                        if args.collision_closed_finger_enabled:
                            closed_mask = filter_grasps_against_closed_surface(
                                centered_environment,
                                grasp_array,
                                gripper.close_pointcloud,
                                collision_threshold=args.collision_closed_finger_threshold_m,
                                surface_samples=args.collision_closed_finger_samples,
                                backoff_m=args.collision_closed_finger_backoff_m,
                            )
                            mask = static_mask & closed_mask
                        else:
                            closed_mask = np.ones(grasp_array.shape[0], dtype=bool)
                            mask = static_mask
                        selected_backoffs = np.full(
                            grasp_array.shape[0], args.collision_closed_finger_backoff_m,
                            dtype=np.float32,
                        )
                        trajectory_specs = [None] * grasp_array.shape[0]
                        for index, item in enumerate(candidate_lifecycle):
                            item["environment"] = {
                                "static_passed": bool(static_mask[index]),
                                "closed_passed": bool(closed_mask[index]),
                                "selected_safe_trajectory": bool(mask[index]),
                                "first_rejection_stage": (
                                    None if mask[index] else
                                    "static" if not static_mask[index] else "closed"
                                ),
                            }
                    safe_source_indices = source_indices[mask]
                    grasps = grasp_array[mask]
                    confidences = score_array[mask]
                    selected_backoffs = np.asarray(selected_backoffs)[mask].tolist()
                    trajectory_specs = [spec for spec, keep in zip(trajectory_specs, mask) if keep]
                    if sources is not None:
                        sources = np.asarray(sources)[mask].tolist()
                    safe_before_diversity = int(len(grasps))
                    if (
                        args.environment_constrained_generation
                        and strategy_used == "sparse"
                        and len(grasps) > args.sparse_quota
                    ):
                        keep = select_yaw_diverse_candidates(
                            grasps,
                            confidences,
                            maximum=args.sparse_quota,
                            yaw_bin_deg=args.constraint_yaw_bin_deg,
                            max_per_bin=args.constraint_max_per_yaw_bin,
                            width_axis_index=args.width_axis_index,
                            source_labels=sources,
                            preferred_source="obb_sparse",
                            min_preferred=args.constraint_min_obb_candidates,
                        )
                        grasps = np.asarray(grasps)[keep]
                        confidences = np.asarray(confidences)[keep]
                        selected_backoffs = [selected_backoffs[index] for index in keep]
                        trajectory_specs = [trajectory_specs[index] for index in keep]
                        if sources is not None:
                            sources = [sources[index] for index in keep]
                        safe_source_indices = safe_source_indices[keep]
                    final_source_index_set = {int(value) for value in safe_source_indices}
                    for index, item in enumerate(candidate_lifecycle):
                        item["environment_safe"] = bool(mask[index])
                        item["selected_after_yaw_diversity"] = index in final_source_index_set
                    source_indices = np.asarray(safe_source_indices, dtype=int)
                    collision_metadata.update({
                        "applied": True,
                        "scene_point_cloud_path": str(args.scene_point_cloud),
                        "scene_points": int(centered_environment.shape[0]),
                        "target_residual_points_removed": int(residual_points_removed),
                        "candidates_before": int(grasp_array.shape[0]),
                        "candidates_after_static": int(np.count_nonzero(static_mask)),
                        "candidates_after_closed_finger": int(np.count_nonzero(mask)),
                        "candidates_safe_before_diversity": safe_before_diversity,
                        "candidates_after": int(len(grasps)),
                        "collision_threshold_m": float(args.collision_threshold_m),
                        "collision_samples": int(args.collision_samples),
                        "closed_finger_enabled": bool(args.collision_closed_finger_enabled),
                        "closed_finger_samples": int(args.collision_closed_finger_samples),
                        "closed_finger_threshold_m": float(args.collision_closed_finger_threshold_m),
                        "closed_finger_backoff_m": float(args.collision_closed_finger_backoff_m),
                        "static_mesh_uses_executed_backoff": True,
                        "environment_constrained_generation": bool(
                            args.environment_constrained_generation and strategy_used == "sparse"
                        ),
                        "backoff_search_m": {
                            "min": float(args.constraint_min_backoff_m),
                            "max": float(args.constraint_max_backoff_m),
                            "step": float(args.constraint_backoff_step_m),
                        },
                        "target_contact_required_points": int(args.constraint_min_target_contact_points),
                        "target_contact_required_fraction": float(args.constraint_min_target_contact_fraction),
                        "pad_contact_depth_m": float(args.constraint_pad_contact_depth_m),
                        "support_plane_enabled": bool(args.constraint_support_plane_enabled),
                        "support_plane": constrained.get("support_plane") if (
                            args.environment_constrained_generation and strategy_used == "sparse"
                        ) else None,
                        "support_plane_required_clearance_m": float(args.constraint_support_plane_clearance_m),
                        "path_samples": int(args.constraint_path_samples),
                        "yaw_diversity": {
                            "bin_deg": float(args.constraint_yaw_bin_deg),
                            "max_per_bin": int(args.constraint_max_per_yaw_bin),
                            "final_quota": int(args.sparse_quota),
                            "min_obb_candidates": int(args.constraint_min_obb_candidates),
                        },
                    })
                except Exception as collision_exc:
                    if not args.collision_fallback_to_object_only:
                        raise
                    collision_metadata.update({
                        "fallback_used": True,
                        "error": f"{type(collision_exc).__name__}: {collision_exc}",
                    })
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "FAILED",
                    "message": f"{type(exc).__name__}: {exc}",
                    "candidates": [],
                }
            )
        )
        return 0

    candidates = make_candidates(
        grasps=grasps,
        confidences=confidences,
        point_cloud=point_cloud,
        gripper_name=args.gripper_name,
        default_gripper_width_m=args.default_gripper_width_m,
        estimate_gripper_width=args.estimate_gripper_width,
        width_axis_index=args.width_axis_index,
        width_clearance_m=args.width_clearance_m,
        sources=sources,
        execution_backoffs_m=locals().get("selected_backoffs"),
        trajectory_specs=locals().get("trajectory_specs"),
        source_indices=locals().get("source_indices"),
    )
    diagnostics_path = write_candidate_lifecycle_diagnostics(
        args.output_dir,
        args.point_cloud,
        {
            "schema_version": 1,
            "strategy_requested": args.strategy,
            "strategy_used": strategy_used,
            "sparse_fallback": args.strategy == "sparse" and strategy_used != "sparse",
            "point_cloud_path": str(args.point_cloud),
            "scene_point_cloud_path": str(args.scene_point_cloud) if args.scene_point_cloud else None,
            "width_axis_index": int(args.width_axis_index),
            "candidate_count_raw": len(candidate_lifecycle),
            "candidate_count_output": len(candidates),
            "collision": collision_metadata,
            "candidates": candidate_lifecycle,
        },
    )
    status = "SUCCESS" if candidates else "NOT_FOUND"
    print(
        json.dumps(
            {
                "status": status,
                "message": (
                    f"Received {len(candidates)} candidate(s) from GraspGenX "
                    f"server {args.host}:{args.port} for {args.gripper_name}; "
                    f"strategy={strategy_used}"
                    + (" (sparse unavailable on this server; diffusion fallback)"
                       if args.strategy == "sparse" and strategy_used != "sparse" else "")
                    + "."
                ),
                "metadata": {
                    "server": metadata,
                    "num_input_points": int(point_cloud.shape[0]),
                    "gripper_name": args.gripper_name,
                    "point_cloud_preprocessing": point_cloud_metadata,
                    "strategy_requested": args.strategy,
                    "strategy_used": strategy_used,
                    "sparse_fallback": args.strategy == "sparse" and strategy_used != "sparse",
                    "scene_collision": collision_metadata,
                    "candidate_lifecycle_path": diagnostics_path,
                },
                "candidates": candidates,
            }
        )
    )
    return 0


def constrain_grasps_to_environment(
    collision_filter,
    scene_points: np.ndarray,
    target_points: np.ndarray,
    grasp_poses: np.ndarray,
    gripper: Any,
    *,
    static_threshold: float,
    static_samples: int,
    static_batch_size: int,
    closed_threshold: float,
    closed_samples: int,
    closed_enabled: bool,
    min_backoff_m: float,
    max_backoff_m: float,
    backoff_step_m: float,
    min_target_contact_points: int,
    pregrasp_distance_m: float,
    retreat_distance_m: float,
    path_samples: int,
    path_surface_samples: int,
    min_target_contact_fraction: float = 0.0,
    pad_contact_depth_m: float | None = None,
    support_plane_enabled: bool = False,
    support_plane_margin_m: float = 0.012,
    support_plane_clearance_m: float = 0.011,
    support_plane_inlier_threshold_m: float = 0.003,
    support_plane_max_tilt_deg: float = 20.0,
    support_plane_min_inliers: int = 24,
) -> dict[str, Any]:
    """Search per-candidate insertion depth and return complete safe trajectories."""
    poses = np.asarray(grasp_poses, dtype=np.float32)
    backoffs = _inclusive_values(min_backoff_m, max_backoff_m, backoff_step_m)
    variants = np.concatenate(
        [apply_grasp_backoff(poses, value) for value in backoffs], axis=0,
    )
    # Layout is backoff-major; reshape to candidate-major below.
    static_flat = np.asarray(collision_filter(
        scene_pc=scene_points,
        grasp_poses=variants,
        gripper_collision_mesh=gripper.collision_mesh,
        collision_threshold=static_threshold,
        num_collision_samples=static_samples,
        batch_size=static_batch_size,
    ), dtype=bool)
    expected = len(backoffs) * poses.shape[0]
    if static_flat.size != expected:
        raise ValueError("GraspGenX static collision filter returned an unexpected mask size")
    static_by_candidate = static_flat.reshape(len(backoffs), poses.shape[0]).T

    selected = np.zeros(poses.shape[0], dtype=bool)
    selected_backoffs = np.full(poses.shape[0], np.nan, dtype=np.float32)
    static_final = np.zeros(poses.shape[0], dtype=bool)
    closed_final = np.zeros(poses.shape[0], dtype=bool)
    support_final = np.zeros(poses.shape[0], dtype=bool)
    contact_final = np.zeros(poses.shape[0], dtype=bool)
    approach_final = np.zeros(poses.shape[0], dtype=bool)
    retreat_final = np.zeros(poses.shape[0], dtype=bool)
    trajectory_specs: list[dict[str, Any] | None] = [None] * poses.shape[0]
    candidate_stages: list[dict[str, Any]] = []
    scene_tree = _scene_tree(scene_points)
    open_surface = _bounded_surface(gripper.open_pointcloud, path_surface_samples)
    closed_surface = _bounded_surface(gripper.close_pointcloud, path_surface_samples)
    support_plane = (
        estimate_local_support_plane(
            scene_points, target_points,
            footprint_margin_m=support_plane_margin_m,
            inlier_threshold_m=support_plane_inlier_threshold_m,
            max_tilt_deg=support_plane_max_tilt_deg,
            min_inliers=support_plane_min_inliers,
        )
        if support_plane_enabled else None
    )

    for candidate_index, raw_pose in enumerate(poses):
        rejection_counts = {"static": 0, "closed": 0, "support_plane": 0, "contact": 0, "approach": 0, "retreat": 0}
        backoff_trials: list[dict[str, Any]] = []
        for backoff_index, backoff in enumerate(backoffs):
            trial: dict[str, Any] = {
                "backoff_m": float(backoff),
                "static_passed": bool(static_by_candidate[candidate_index, backoff_index]),
            }
            if not static_by_candidate[candidate_index, backoff_index]:
                rejection_counts["static"] += 1
                trial["outcome"] = "rejected_static"
                backoff_trials.append(trial)
                continue
            executed = apply_grasp_backoff(raw_pose[None], backoff)[0]
            trial["executed_transform"] = executed.astype(float).tolist()
            trial["open_surface_min_clearance_m"] = _surface_pose_min_clearance(
                scene_tree, open_surface, executed,
            )
            closed_clearance = _surface_pose_min_clearance(
                scene_tree, closed_surface, executed,
            )
            trial["closed_surface_min_clearance_m"] = closed_clearance
            static_final[candidate_index] = True
            if closed_enabled and closed_clearance <= float(closed_threshold):
                rejection_counts["closed"] += 1
                trial["outcome"] = "rejected_closed"
                backoff_trials.append(trial)
                continue
            closed_final[candidate_index] = True
            if support_plane is not None:
                plane_clearance = _surface_plane_min_clearance(
                    closed_surface, executed, support_plane,
                )
                trial["support_plane_clearance_m"] = plane_clearance
                trial["support_plane"] = support_plane
                if plane_clearance <= float(support_plane_clearance_m):
                    rejection_counts["support_plane"] += 1
                    trial["outcome"] = "rejected_support_plane"
                    backoff_trials.append(trial)
                    continue
            support_final[candidate_index] = True
            contact = target_contact_evidence(
                target_points, executed, gripper.grasp_volume,
                pad_contact_depth_m=pad_contact_depth_m,
            )
            trial["target_contact_point_count"] = int(contact["point_count"])
            trial["target_contact_fraction"] = float(contact["fraction"])
            trial["target_pad_contact_point_count"] = int(contact["point_count"])
            trial["target_pad_contact_fraction"] = float(contact["fraction"])
            if (
                contact["point_count"] < int(min_target_contact_points)
                or contact["fraction"] < float(min_target_contact_fraction)
            ):
                rejection_counts["contact"] += 1
                trial["outcome"] = "rejected_contact"
                backoff_trials.append(trial)
                continue
            contact_final[candidate_index] = True

            pregrasp = _offset_transform_local_z(executed, -float(pregrasp_distance_m))
            retreat = executed.copy()
            retreat[:3, 3] += np.array([0.0, 0.0, float(retreat_distance_m)], dtype=np.float32)
            approach_clearance = _surface_segment_min_clearance(
                scene_tree, open_surface, pregrasp, executed, path_samples,
            )
            trial["approach_min_clearance_m"] = approach_clearance
            if approach_clearance <= float(static_threshold):
                rejection_counts["approach"] += 1
                trial["outcome"] = "rejected_approach"
                backoff_trials.append(trial)
                continue
            approach_final[candidate_index] = True
            retreat_clearance = _surface_segment_min_clearance(
                scene_tree, closed_surface, executed, retreat, path_samples,
            )
            trial["retreat_min_clearance_m"] = retreat_clearance
            if retreat_clearance <= float(closed_threshold):
                rejection_counts["retreat"] += 1
                trial["outcome"] = "rejected_retreat"
                backoff_trials.append(trial)
                continue
            retreat_final[candidate_index] = True

            selected[candidate_index] = True
            selected_backoffs[candidate_index] = float(backoff)
            trial["outcome"] = "selected"
            backoff_trials.append(trial)
            trajectory_specs[candidate_index] = {
                "frame": "object_point_cloud",
                "pregrasp_transform": pregrasp.astype(float).tolist(),
                "grasp_transform": executed.astype(float).tolist(),
                "retreat_transform": retreat.astype(float).tolist(),
                "execution_backoff_m": float(backoff),
                "target_contact_point_count": int(contact["point_count"]),
                "target_contact_fraction": float(contact["fraction"]),
                "target_pad_contact_point_count": int(contact["point_count"]),
                "target_pad_contact_fraction": float(contact["fraction"]),
                "support_plane": support_plane,
                "support_plane_clearance_m": trial.get("support_plane_clearance_m"),
                "path_samples": int(path_samples),
                "intermediate_closing_check": "urdf_finger_envelope_downstream",
                "rejections_before_selection": dict(rejection_counts),
            }
            break

        first_rejection = next(
            (name for name, passed in (
                ("static", static_final[candidate_index]),
                ("closed", closed_final[candidate_index]),
                ("support_plane", support_final[candidate_index]),
                ("contact", contact_final[candidate_index]),
                ("approach", approach_final[candidate_index]),
                ("retreat", retreat_final[candidate_index]),
            ) if not passed),
            None,
        )
        candidate_stages.append({
            "static_passed": bool(static_final[candidate_index]),
            "closed_passed": bool(closed_final[candidate_index]),
            "support_plane_passed": bool(support_final[candidate_index]),
            "contact_passed": bool(contact_final[candidate_index]),
            "approach_passed": bool(approach_final[candidate_index]),
            "retreat_passed": bool(retreat_final[candidate_index]),
            "selected_safe_trajectory": bool(selected[candidate_index]),
            "selected_backoff_m": (
                float(selected_backoffs[candidate_index])
                if np.isfinite(selected_backoffs[candidate_index]) else None
            ),
            "first_rejection_stage": first_rejection,
            "rejection_counts_by_backoff": dict(rejection_counts),
            "backoff_trials": backoff_trials,
            "selected_trajectory": trajectory_specs[candidate_index],
        })

    return {
        "mask": selected,
        "selected_backoffs_m": selected_backoffs,
        "trajectory_specs": trajectory_specs,
        "static_final_mask": static_final,
        "closed_final_mask": closed_final,
        "candidate_stages": candidate_stages,
        "support_plane": support_plane,
    }


def target_contact_evidence(
    target_points: np.ndarray, grasp_pose: np.ndarray, grasp_volume: Any, *,
    pad_contact_depth_m: float | None = None,
) -> dict[str, Any]:
    """Count target points in the distal finger-pad contact region."""
    target = np.asarray(target_points, dtype=np.float32)
    pose = np.asarray(grasp_pose, dtype=np.float32)
    bounds = np.asarray(grasp_volume, dtype=np.float32)
    if bounds.shape != (2, 3):
        raise ValueError(f"Expected grasp volume shape (2, 3), got {bounds.shape}")
    contact_bounds = bounds.copy()
    if pad_contact_depth_m is not None:
        depth = float(pad_contact_depth_m)
        if not math.isfinite(depth) or depth <= 0.0:
            raise ValueError("pad_contact_depth_m must be positive")
        contact_bounds[0, 2] = max(contact_bounds[0, 2], contact_bounds[1, 2] - depth)
    local = (target[:, :3] - pose[:3, 3]) @ pose[:3, :3]
    inside = np.all((local >= contact_bounds[0]) & (local <= contact_bounds[1]), axis=1)
    count = int(np.count_nonzero(inside))
    return {
        "point_count": count,
        "fraction": count / max(1, target.shape[0]),
        "contact_bounds": contact_bounds.astype(float).tolist(),
    }


def estimate_local_support_plane(
    scene_points: np.ndarray, target_points: np.ndarray, *,
    footprint_margin_m: float = 0.012, inlier_threshold_m: float = 0.003,
    max_tilt_deg: float = 20.0, min_inliers: int = 24,
    ransac_trials: int = 96,
) -> dict[str, Any] | None:
    """Estimate the highest horizontal support immediately below the target."""
    scene = np.asarray(scene_points, dtype=np.float64)[:, :3]
    target = np.asarray(target_points, dtype=np.float64)[:, :3]
    if scene.shape[0] < max(3, int(min_inliers)) or target.shape[0] < 3:
        return None
    lo, hi = np.percentile(target[:, :2], [2.0, 98.0], axis=0)
    margin = max(0.0, float(footprint_margin_m))
    xy_mask = np.all((scene[:, :2] >= lo - margin) & (scene[:, :2] <= hi + margin), axis=1)
    ceiling = float(np.percentile(target[:, 2], 15.0)) + max(0.004, float(inlier_threshold_m))
    local = scene[xy_mask & (scene[:, 2] <= ceiling)]
    required = max(3, int(min_inliers))
    if local.shape[0] < required:
        return None
    threshold = max(1e-4, float(inlier_threshold_m))
    max_tilt = math.radians(float(max_tilt_deg))
    rng = np.random.default_rng(0)
    best = None
    for _ in range(max(1, int(ransac_trials))):
        sample = local[rng.choice(local.shape[0], 3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        norm = float(np.linalg.norm(normal))
        if norm <= 1e-9:
            continue
        normal /= norm
        if normal[2] < 0.0:
            normal = -normal
        tilt = math.acos(float(np.clip(normal[2], -1.0, 1.0)))
        if tilt > max_tilt:
            continue
        d = -float(normal @ sample[0])
        residuals = np.abs(local @ normal + d)
        inliers = residuals <= threshold
        count = int(np.count_nonzero(inliers))
        if count < required:
            continue
        fitted = local[inliers]
        design = np.column_stack((fitted[:, 0], fitted[:, 1], np.ones(count)))
        a, b, c = np.linalg.lstsq(design, fitted[:, 2], rcond=None)[0]
        refined = np.array([-a, -b, 1.0], dtype=np.float64)
        refined /= np.linalg.norm(refined)
        refined_tilt = math.acos(float(np.clip(refined[2], -1.0, 1.0)))
        if refined_tilt > max_tilt:
            continue
        refined_d = -float(c) / float(np.linalg.norm(np.array([-a, -b, 1.0])))
        refined_residuals = np.abs(local @ refined + refined_d)
        refined_inliers = refined_residuals <= threshold
        refined_count = int(np.count_nonzero(refined_inliers))
        center_xy = np.median(target[:, :2], axis=0)
        center_height = float(-(refined[0] * center_xy[0] + refined[1] * center_xy[1] + refined_d) / refined[2])
        # Prefer the dominant local surface. Choosing height first can select
        # a sparse band of target-mask leakage a few millimetres above the
        # actual table/platform and falsely reject every grasp. Height only
        # breaks ties between similarly supported planes.
        score = (refined_count, center_height)
        if refined_count >= required and (best is None or score > best[0]):
            rmse = float(np.sqrt(np.mean(refined_residuals[refined_inliers] ** 2)))
            best = (score, refined, refined_d, refined_count, rmse)
    if best is None:
        return None
    _score, normal, d, count, rmse = best
    return {
        "normal": normal.astype(float).tolist(), "d": float(d),
        "inlier_count": int(count), "rmse_m": rmse,
        "local_point_count": int(local.shape[0]),
        "inlier_ratio": float(count / max(1, local.shape[0])),
    }


def _surface_plane_min_clearance(
    surface: np.ndarray, pose: np.ndarray, plane: dict[str, Any],
) -> float:
    transformed = np.asarray(surface) @ pose[:3, :3].T + pose[:3, 3]
    normal = np.asarray(plane["normal"], dtype=np.float64)
    return float(np.min(transformed @ normal + float(plane["d"])))


def select_yaw_diverse_candidates(
    grasp_poses: Any,
    scores: Any,
    *,
    maximum: int,
    yaw_bin_deg: float = 15.0,
    max_per_bin: int = 2,
    width_axis_index: int = 0,
    source_labels: Any | None = None,
    preferred_source: str = "obb_sparse",
    min_preferred: int = 0,
) -> list[int]:
    """Select high-quality candidates without collapsing onto one planar yaw.

    Two-finger grasps are pi-periodic. Binning the projected opening axis over
    [0, pi) is independent of obstacle type and preserves alternative free-space
    directions for later MoveIt screening.
    """
    poses = np.asarray(grasp_poses, dtype=np.float32)
    values = np.asarray(scores, dtype=np.float32).reshape(-1)
    limit = min(max(0, int(maximum)), poses.shape[0], values.size)
    if limit == 0:
        return []
    bin_width = math.radians(float(yaw_bin_deg))
    if not math.isfinite(bin_width) or bin_width <= 0.0:
        raise ValueError("yaw_bin_deg must be positive")
    per_bin = max(1, int(max_per_bin))
    axes = poses[:, :2, int(width_axis_index)]
    yaws = np.mod(np.arctan2(axes[:, 1], axes[:, 0]), math.pi)
    bins = np.floor(yaws / bin_width).astype(int)
    order = sorted(range(min(poses.shape[0], values.size)), key=lambda i: float(values[i]), reverse=True)
    selected: list[int] = []
    counts: dict[int, int] = {}

    def append_diverse(indices: list[int], target_count: int) -> None:
        for index in indices:
            if index in selected:
                continue
            bucket = int(bins[index])
            if counts.get(bucket, 0) >= per_bin:
                continue
            selected.append(index)
            counts[bucket] = counts.get(bucket, 0) + 1
            if len(selected) >= target_count:
                return

    if source_labels is not None and int(min_preferred) > 0:
        labels = list(source_labels)
        preferred_order = [
            index for index in order
            if index < len(labels) and str(labels[index]) == str(preferred_source)
        ]
        append_diverse(preferred_order, min(limit, int(min_preferred)))
    append_diverse(order, limit)
    if len(selected) == limit:
        return selected
    # If too few distinct bins survived the scene checks, fill the remaining
    # slots by score rather than discarding otherwise safe candidates.
    selected_set = set(selected)
    selected.extend(index for index in order if index not in selected_set)
    return selected[:limit]


def _inclusive_values(minimum: float, maximum: float, step: float) -> np.ndarray:
    low, high, increment = float(minimum), float(maximum), float(step)
    if low < 0.0 or high < low or increment <= 0.0:
        raise ValueError("Backoff search requires 0 <= min <= max and step > 0")
    count = int(math.floor((high - low) / increment + 1e-9)) + 1
    values = low + np.arange(count, dtype=np.float32) * increment
    if values[-1] < high - 1e-8:
        values = np.append(values, np.float32(high))
    return values


def _scene_tree(scene_points: np.ndarray):
    from scipy.spatial import cKDTree
    return cKDTree(np.asarray(scene_points, dtype=np.float32))


def _bounded_surface(surface_points: Any, maximum: int) -> np.ndarray:
    surface = np.asarray(surface_points, dtype=np.float32)
    limit = int(maximum)
    if surface.ndim != 2 or surface.shape[1] != 3 or limit < 1:
        raise ValueError("Gripper surface must be (N, 3) and sample limit positive")
    if surface.shape[0] > limit:
        surface = surface[np.linspace(0, surface.shape[0] - 1, limit, dtype=int)]
    return surface


def _surface_pose_collision_free(tree, surface: np.ndarray, pose: np.ndarray, threshold: float) -> bool:
    return _surface_pose_min_clearance(tree, surface, pose) > float(threshold)


def _surface_pose_min_clearance(tree, surface: np.ndarray, pose: np.ndarray) -> float:
    """Return the nearest sampled gripper-surface to scene-point distance."""
    transformed = surface @ pose[:3, :3].T + pose[:3, 3]
    distances, _indices = tree.query(transformed, k=1, workers=-1)
    return float(np.min(distances))


def _surface_segment_collision_free(
    tree, surface: np.ndarray, start: np.ndarray, end: np.ndarray,
    threshold: float, samples: int,
) -> bool:
    return _surface_segment_min_clearance(tree, surface, start, end, samples) > float(threshold)


def _surface_segment_min_clearance(
    tree, surface: np.ndarray, start: np.ndarray, end: np.ndarray, samples: int,
) -> float:
    """Return the minimum sampled clearance along a translational pose segment."""
    count = max(2, int(samples))
    minimum = float("inf")
    for alpha in np.linspace(0.0, 1.0, count):
        pose = start.copy()
        pose[:3, 3] = (1.0 - alpha) * start[:3, 3] + alpha * end[:3, 3]
        minimum = min(minimum, _surface_pose_min_clearance(tree, surface, pose))
    return minimum


def _offset_transform_local_z(pose: np.ndarray, distance_m: float) -> np.ndarray:
    result = np.asarray(pose, dtype=np.float32).copy()
    result[:3, 3] += result[:3, 2] * float(distance_m)
    return result


def remove_target_residuals(
    environment_points: np.ndarray,
    target_points: np.ndarray,
    *,
    footprint_margin_m: float = 0.002,
    target_height_percentile: float = 10.0,
) -> tuple[np.ndarray, int]:
    """Remove leaked target-edge points while retaining its lower support.

    Sparse inference runs in base_link, so XY is the support plane and +Z is
    height. Only points inside the robust target footprint and at target-body
    height are removed. Lower platform/table points remain collision geometry.
    """
    environment = np.asarray(environment_points, dtype=np.float32)
    target = np.asarray(target_points, dtype=np.float32)
    if environment.ndim != 2 or environment.shape[1] != 3:
        raise ValueError("Environment points must have shape (N, 3)")
    if target.ndim != 2 or target.shape[1] != 3 or target.shape[0] < 10:
        raise ValueError("Target points must have shape (N, 3) with at least ten points")
    margin = float(footprint_margin_m)
    if margin < 0.0:
        raise ValueError("Target residual footprint margin must be non-negative")
    lower_xy, upper_xy = np.percentile(target[:, :2], [2.0, 98.0], axis=0)
    lower_xy -= margin
    upper_xy += margin
    target_height = float(np.percentile(target[:, 2], target_height_percentile))
    inside_footprint = np.all(
        (environment[:, :2] >= lower_xy) & (environment[:, :2] <= upper_xy), axis=1,
    )
    leaked_target = inside_footprint & (environment[:, 2] >= target_height)
    return environment[~leaked_target], int(np.count_nonzero(leaked_target))


def apply_grasp_backoff(grasp_poses: np.ndarray, backoff_m: float) -> np.ndarray:
    """Return poses shifted from GraspGenX contact to executed grasp targets."""
    poses = np.asarray(grasp_poses, dtype=np.float32)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"Expected grasp poses shape (N, 4, 4), got {poses.shape}")
    backoff = float(backoff_m)
    if backoff < 0.0:
        raise ValueError("grasp backoff must be non-negative")
    adjusted = poses.copy()
    adjusted[:, :3, 3] -= adjusted[:, :3, 2] * backoff
    return adjusted


def filter_grasps_against_closed_surface(
    scene_points: np.ndarray,
    grasp_poses: np.ndarray,
    closed_surface_points: np.ndarray,
    *,
    collision_threshold: float,
    surface_samples: int,
    backoff_m: float,
) -> np.ndarray:
    """Reject poses whose actual closed-gripper surface touches the scene."""
    from scipy.spatial import cKDTree

    scene = np.asarray(scene_points, dtype=np.float32)
    poses = np.asarray(grasp_poses, dtype=np.float32)
    surface = np.asarray(closed_surface_points, dtype=np.float32)
    if scene.ndim != 2 or scene.shape[1] != 3 or scene.shape[0] == 0:
        raise ValueError("Scene points must have shape (N, 3) and be non-empty")
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"Expected grasp poses shape (N, 4, 4), got {poses.shape}")
    if surface.ndim != 2 or surface.shape[1] != 3 or surface.shape[0] == 0:
        raise ValueError("Closed gripper surface must have shape (N, 3) and be non-empty")
    threshold = float(collision_threshold)
    if threshold < 0.0:
        raise ValueError("collision threshold must be non-negative")
    backoff = float(backoff_m)
    if backoff < 0.0:
        raise ValueError("closed finger backoff must be non-negative")
    maximum = int(surface_samples)
    if maximum < 1:
        raise ValueError("closed finger surface samples must be at least one")
    if poses.shape[0] == 0:
        return np.zeros((0,), dtype=bool)
    if surface.shape[0] > maximum:
        indices = np.linspace(0, surface.shape[0] - 1, maximum, dtype=int)
        surface = surface[indices]

    tree = cKDTree(scene)
    collision_free = []
    for pose in poses:
        # The execution safety layer backs the final tool target away from the
        # object along negative local +Z. Evaluate the closed geometry at that
        # same pose instead of at GraspGenX's uncorrected contact transform.
        executed_translation = pose[:3, 3] - pose[:3, 2] * backoff
        transformed = surface @ pose[:3, :3].T + executed_translation
        distances, _indices = tree.query(
            transformed, k=1, distance_upper_bound=threshold, workers=-1,
        )
        collision_free.append(not np.isfinite(distances).any())
    return np.asarray(collision_free, dtype=bool)


def load_point_cloud(path: str) -> np.ndarray:
    points, _metadata = load_point_cloud_with_metadata(path)
    return points


def load_point_cloud_with_metadata(path: str) -> tuple[np.ndarray, dict]:
    point_cloud_path = Path(path)
    if not point_cloud_path.is_file():
        raise FileNotFoundError(f"Point cloud does not exist: {point_cloud_path}")
    suffix = point_cloud_path.suffix.lower()
    if suffix == ".npy":
        points = np.load(point_cloud_path)
    elif suffix == ".xyz":
        points = np.loadtxt(point_cloud_path, dtype=np.float32)
    elif suffix == ".ply":
        points = load_ascii_ply_xyz(point_cloud_path)
    else:
        raise ValueError(f"Unsupported point cloud format: {suffix}")

    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"Expected point cloud shape (N, 3+), got {points.shape}")
    points = points[:, :3]
    if points.shape[0] == 0:
        raise ValueError("Point cloud is empty.")
    raw_point_count = int(points.shape[0])
    points = points[np.all(np.isfinite(points), axis=1)]
    if points.shape[0] == 0:
        raise ValueError("Point cloud contains no finite XYZ points.")
    centroid = points.mean(axis=0, keepdims=True)
    points = points - centroid
    centered_centroid = points.mean(axis=0)
    metadata = {
        "source_path": str(point_cloud_path),
        "raw_point_count": raw_point_count,
        "finite_point_count": int(points.shape[0]),
        "centered_for_graspgenx": True,
        "centering_method": "subtract_finite_xyz_mean",
        "original_centroid_m": {
            "x": float(centroid[0, 0]),
            "y": float(centroid[0, 1]),
            "z": float(centroid[0, 2]),
        },
        "centered_centroid_m": {
            "x": float(centered_centroid[0]),
            "y": float(centered_centroid[1]),
            "z": float(centered_centroid[2]),
        },
    }
    return points, metadata


def load_point_cloud_uncentered(path: str) -> np.ndarray:
    point_cloud_path = Path(path)
    if not point_cloud_path.is_file():
        raise FileNotFoundError(f"Scene point cloud does not exist: {point_cloud_path}")
    suffix = point_cloud_path.suffix.lower()
    if suffix == ".npy":
        points = np.load(point_cloud_path)
    elif suffix == ".xyz":
        points = np.loadtxt(point_cloud_path, dtype=np.float32)
    elif suffix == ".ply":
        points = load_ascii_ply_xyz(point_cloud_path)
    else:
        raise ValueError(f"Unsupported scene point cloud format: {suffix}")
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"Expected scene point cloud shape (N, 3+), got {points.shape}")
    points = points[:, :3]
    points = points[np.all(np.isfinite(points), axis=1)]
    if points.shape[0] == 0:
        raise ValueError("Scene point cloud contains no finite XYZ points.")
    return points


def load_ascii_ply_xyz(path: Path) -> np.ndarray:
    points: list[list[float]] = []
    with path.open("r", encoding="utf-8") as handle:
        in_body = False
        for line in handle:
            stripped = line.strip()
            if not in_body:
                if stripped == "end_header":
                    in_body = True
                continue
            if not stripped:
                continue
            values = stripped.split()
            if len(values) >= 3:
                points.append([float(values[0]), float(values[1]), float(values[2])])
    return np.asarray(points, dtype=np.float32)


def initialize_candidate_lifecycle(
    grasps: Any,
    scores: Any,
    sources: Any,
    *,
    width_axis_index: int,
) -> list[dict[str, Any]]:
    """Create stable raw-candidate records before any environment filtering."""
    poses = np.asarray(grasps, dtype=np.float32)
    values = np.asarray(scores, dtype=np.float32).reshape(-1)
    records: list[dict[str, Any]] = []
    for index in range(min(poses.shape[0], values.size, len(sources))):
        axis = poses[index, :3, int(width_axis_index)]
        yaw_deg = math.degrees(math.atan2(float(axis[1]), float(axis[0]))) % 180.0
        records.append({
            "raw_index": index,
            "candidate_id": f"graspgenx_raw_{index + 1}",
            "generation_branch": str(sources[index]),
            "score": float(values[index]),
            "opening_axis_yaw_deg_mod_180": yaw_deg,
            "opening_axis": axis.astype(float).tolist(),
            "approach_axis": poses[index, :3, 2].astype(float).tolist(),
            "raw_transform": poses[index].astype(float).tolist(),
            "environment_safe": None,
            "selected_after_yaw_diversity": None,
            "environment": None,
            "moveit": [],
            "gui_selected": False,
        })
    return records


def write_candidate_lifecycle_diagnostics(
    output_dir: str,
    point_cloud_path: str,
    payload: dict[str, Any],
) -> str:
    """Persist one stable JSON artifact that the GUI can later enrich."""
    stem = Path(point_cloud_path).stem
    path = Path(output_dir) / f"{stem}_candidate_lifecycle.json"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)
    return str(path.resolve())


def make_candidates(
    grasps: Any,
    confidences: Any,
    gripper_name: str,
    default_gripper_width_m: float | None,
    point_cloud: Any | None = None,
    estimate_gripper_width: bool = True,
    width_axis_index: int = 1,
    width_clearance_m: float = 0.01,
    sources: Any | None = None,
    execution_backoffs_m: Any | None = None,
    trajectory_specs: Any | None = None,
    source_indices: Any | None = None,
) -> list[dict]:
    grasp_array = np.asarray(grasps, dtype=np.float32)
    score_array = np.asarray(confidences, dtype=np.float32).reshape(-1)
    pc_array = None if point_cloud is None else np.asarray(point_cloud, dtype=np.float32)
    if grasp_array.ndim != 3 or grasp_array.shape[1:] != (4, 4):
        raise ValueError(f"Expected grasps shape (N, 4, 4), got {grasp_array.shape}")
    count = min(int(grasp_array.shape[0]), int(score_array.shape[0]))
    if sources is not None and len(sources) != count:
        raise ValueError("GraspGenX source count does not match grasp count")
    if execution_backoffs_m is not None and len(execution_backoffs_m) != count:
        raise ValueError("Execution backoff count does not match grasp count")
    if trajectory_specs is not None and len(trajectory_specs) != count:
        raise ValueError("Trajectory specification count does not match grasp count")
    if source_indices is not None and len(source_indices) != count:
        raise ValueError("Source-index count does not match grasp count")
    candidates = []
    for index in range(count):
        matrix = grasp_array[index].astype(float).tolist()
        score = float(score_array[index])
        rotation = np.asarray(matrix, dtype=float)[:3, :3]
        approach_vector = rotation[:, 2].astype(float).tolist()
        closing_axis = rotation[:, int(width_axis_index)].astype(float).tolist()
        raw_index = int(source_indices[index]) if source_indices is not None else index
        candidate = {
            "candidate_id": (
                f"graspgenx_raw_{raw_index + 1}"
                if source_indices is not None else f"graspgenx_{index + 1}"
            ),
            "valid": math.isfinite(score),
            "score": score,
            "pose_frame": "object_point_cloud",
            "transform": matrix,
            "pose": matrix_to_pose(matrix, frame="object_point_cloud"),
            "approach_vector": approach_vector,
            "closing_axis": closing_axis,
            "metadata": {
                "source": "graspgenx_zmq",
                "gripper_name": gripper_name,
                "generation_branch": str(sources[index]) if sources is not None else "diffusion",
                "raw_candidate_index": raw_index,
                "opening_axis_yaw_deg_mod_180": (
                    math.degrees(math.atan2(float(closing_axis[1]), float(closing_axis[0])))
                    % 180.0
                ),
            },
        }
        if execution_backoffs_m is not None:
            candidate["execution_backoff_m"] = float(execution_backoffs_m[index])
            candidate["metadata"]["execution_backoff_source"] = "environment_constraint_search"
        if trajectory_specs is not None and isinstance(trajectory_specs[index], dict):
            candidate["trajectory"] = dict(trajectory_specs[index])
            candidate["metadata"]["environment_constrained_trajectory"] = True
        if default_gripper_width_m is not None:
            candidate["gripper_width_m"] = float(default_gripper_width_m)
            candidate["metadata"]["gripper_width_source"] = "configured_default"
        elif estimate_gripper_width and pc_array is not None:
            width = estimate_width_from_point_cloud_projection(
                pc_array,
                matrix,
                axis_index=width_axis_index,
                clearance_m=width_clearance_m,
            )
            if width is not None:
                candidate["gripper_width_m"] = width
                candidate["metadata"]["gripper_width_source"] = (
                    "point_cloud_projection_estimate"
                )
                candidate["metadata"]["gripper_width_axis_index"] = int(width_axis_index)
                candidate["metadata"]["gripper_width_clearance_m"] = float(
                    width_clearance_m
                )
        candidates.append(candidate)
    return candidates


def estimate_width_from_point_cloud_projection(
    point_cloud: np.ndarray,
    transform: list[list[float]],
    axis_index: int = 1,
    clearance_m: float = 0.01,
) -> float | None:
    points = np.asarray(point_cloud, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3 or points.shape[0] == 0:
        return None
    axis = np.asarray(
        [transform[0][axis_index], transform[1][axis_index], transform[2][axis_index]],
        dtype=np.float32,
    )
    norm = float(np.linalg.norm(axis))
    if not math.isfinite(norm) or norm <= 0.0:
        return None
    axis = axis / norm
    projections = points[:, :3] @ axis
    finite = projections[np.isfinite(projections)]
    if finite.size == 0:
        return None
    width = float(finite.max() - finite.min() + max(float(clearance_m), 0.0))
    if not math.isfinite(width) or width <= 0.0:
        return None
    return width


def matrix_to_pose(matrix: list[list[float]], frame: str) -> dict:
    return {
        "frame": frame,
        "position": {
            "x": float(matrix[0][3]),
            "y": float(matrix[1][3]),
            "z": float(matrix[2][3]),
        },
        "orientation": matrix_to_quaternion(matrix),
    }


def matrix_to_quaternion(matrix: list[list[float]]) -> dict[str, float]:
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
            x = 0.25 * scale
            y = (float(matrix[0][1]) + float(matrix[1][0])) / scale
            z = (float(matrix[0][2]) + float(matrix[2][0])) / scale
            w = (float(matrix[2][1]) - float(matrix[1][2])) / scale
        elif index == 1:
            scale = math.sqrt(1.0 + diagonal[1] - diagonal[0] - diagonal[2]) * 2.0
            x = (float(matrix[0][1]) + float(matrix[1][0])) / scale
            y = 0.25 * scale
            z = (float(matrix[1][2]) + float(matrix[2][1])) / scale
            w = (float(matrix[0][2]) - float(matrix[2][0])) / scale
        else:
            scale = math.sqrt(1.0 + diagonal[2] - diagonal[0] - diagonal[1]) * 2.0
            x = (float(matrix[0][2]) + float(matrix[2][0])) / scale
            y = (float(matrix[1][2]) + float(matrix[2][1])) / scale
            z = 0.25 * scale
            w = (float(matrix[1][0]) - float(matrix[0][1])) / scale
    return {"x": float(x), "y": float(y), "z": float(z), "w": float(w)}


if __name__ == "__main__":
    sys.exit(main())
