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
        default=1,
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
        with redirect_stdout(sys.stderr):
            from graspgenx.serving.zmq_client import GraspGenXClient

            with GraspGenXClient(
                host=args.host, port=args.port, timeout_ms=args.timeout_ms
            ) as client:
                metadata = client.server_metadata
                grasps, confidences = client.infer(
                    point_cloud,
                    gripper_name=args.gripper_name,
                    num_grasps=args.num_grasps,
                    grasp_threshold=args.grasp_threshold,
                    topk_num_grasps=args.topk_num_grasps,
                )
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
    )
    status = "SUCCESS" if candidates else "NOT_FOUND"
    print(
        json.dumps(
            {
                "status": status,
                "message": (
                    f"Received {len(candidates)} candidate(s) from GraspGenX "
                    f"server {args.host}:{args.port} for {args.gripper_name}."
                ),
                "metadata": {
                    "server": metadata,
                    "num_input_points": int(point_cloud.shape[0]),
                    "gripper_name": args.gripper_name,
                    "point_cloud_preprocessing": point_cloud_metadata,
                },
                "candidates": candidates,
            }
        )
    )
    return 0


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


def make_candidates(
    grasps: Any,
    confidences: Any,
    gripper_name: str,
    default_gripper_width_m: float | None,
    point_cloud: Any | None = None,
    estimate_gripper_width: bool = True,
    width_axis_index: int = 1,
    width_clearance_m: float = 0.01,
) -> list[dict]:
    grasp_array = np.asarray(grasps, dtype=np.float32)
    score_array = np.asarray(confidences, dtype=np.float32).reshape(-1)
    pc_array = None if point_cloud is None else np.asarray(point_cloud, dtype=np.float32)
    if grasp_array.ndim != 3 or grasp_array.shape[1:] != (4, 4):
        raise ValueError(f"Expected grasps shape (N, 4, 4), got {grasp_array.shape}")
    count = min(int(grasp_array.shape[0]), int(score_array.shape[0]))
    candidates = []
    for index in range(count):
        matrix = grasp_array[index].astype(float).tolist()
        score = float(score_array[index])
        rotation = np.asarray(matrix, dtype=float)[:3, :3]
        approach_vector = rotation[:, 2].astype(float).tolist()
        closing_axis = rotation[:, int(width_axis_index)].astype(float).tolist()
        candidate = {
            "candidate_id": f"graspgenx_{index + 1}",
            "valid": math.isfinite(score),
            "score": score,
            "pose_frame": "object_point_cloud",
            "transform": matrix,
            "pose": matrix_to_pose(matrix, frame="object_point_cloud"),
            "approach_vector": approach_vector,
            "closing_axis": closing_axis,
            "metadata": {"source": "graspgenx_zmq", "gripper_name": gripper_name},
        }
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
