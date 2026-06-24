"""GraspGenX MoveIt planning and RViz preview with real-arm execution blocked."""

from __future__ import annotations

import math
import pprint
import sys
from typing import Any

from llm_skill_robot.core.trajectory_validator import TrajectoryValidator
from llm_skill_robot.core.trajectory_visualizer import TrajectoryVisualizer
from llm_skill_robot.grasping.graspgenx_grasp_planner import GraspGenXGraspPlanner
from llm_skill_robot.grasping.graspgenx_subprocess_adapter import GraspGenXSubprocessAdapter
from llm_skill_robot.perception.object_point_cloud_builder import ObjectPointCloudBuilder
from llm_skill_robot.perception.perception_factory import create_perception_pipeline
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


def plan_pose_sequence(
    backend,
    poses: dict[str, dict],
    trajectory_validator,
    velocity_scale: float = 0.05,
    acceleration_scale: float = 0.05,
) -> dict:
    """Plan and validate three segments; this function has no execution call."""
    results = []
    start_state = None
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
        print(f"{stage['stage']} trajectory validation result:")
        pprint.pprint(stage["validation"], sort_dicts=False)


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
        detection = perception_pipeline.detect_object(query, require_pose=False)
        selected_object = _select_highest_confidence(detection.get("objects", []))
        if selected_object is None:
            print("No object detected.")
            return
        camera_backend = perception_pipeline.backend
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
        generation = planner.generate_candidates(point_cloud["point_cloud_path"])
        candidates = generation.get("candidates", [])
        selected = planner.select_best_candidate(candidates)
        if selected is None:
            print("No valid GraspGenX candidates.")
            return

        poses = create_preview_poses(
            selected,
            preview_config,
            fallback_frame=point_cloud.get("frame") or "base_link",
        )
        print(f"selected candidate score: {_candidate_value(selected, 'score')}")
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

        sequence = plan_pose_sequence(
            backend,
            poses,
            validator,
            velocity_scale=float(preview_config.get("velocity_scale", 0.05)),
            acceleration_scale=float(preview_config.get("acceleration_scale", 0.05)),
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
