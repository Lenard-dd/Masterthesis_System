"""GraspGenX MoveIt planning and RViz preview with real-arm execution blocked."""

from __future__ import annotations

import math
import pprint
import sys
import time
from typing import Any, Callable

from llm_skill_robot.core.trajectory_validator import TrajectoryValidator
from llm_skill_robot.core.trajectory_visualizer import TrajectoryVisualizer
from llm_skill_robot.grasping.graspgenx_grasp_planner import GraspGenXGraspPlanner, generate_candidates_with_context, screen_candidate_order
from llm_skill_robot.grasping.graspgenx_subprocess_adapter import GraspGenXSubprocessAdapter
from llm_skill_robot.grasping.parallel_jaw_symmetry import (
    canonicalize_sparse_parallel_jaw_candidates,
    current_closing_axis,
)
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


# Compatibility exports for existing CLI imports. Runtime callers use the module below.
from llm_skill_robot.grasping.candidate_preview import (
    _apply_local_pose_transform,
    _environment_trajectory_poses,
    _failed_sequence_stage,
    _failed_sequence_validation,
    _is_transform_matrix,
    _matrix_to_quaternion,
    _multiply_quaternions,
    _normalize_quaternion_dict,
    _normalized_vector,
    _offset_pose,
    _plan_attempt_ranking_key,
    _quaternion_tuple_to_dict,
    _rotate_local_z,
    _rotate_vector_by_quaternion,
    _rpy_to_quaternion,
    _sequence_screening_summary,
    _three_float_config,
    apply_grasp_frame_to_ee_transform,
    create_grasp_pose,
    create_preview_poses,
    localize_graspgenx_candidate,
    moveit_planning_kwargs,
    moveit_stage_planning_kwargs,
    plan_pose_sequence,
    planning_variant_kwargs,
    prepare_candidate_preview,
    screen_grasp_candidates_for_plan_only,
    transform_pose_sequence_to_target,
)

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
    """Allow planning-only previews regardless of whether real execution is enabled.

    This demo instantiates MoveIt with ``execution_mode='plan_only'`` and has no
    trajectory execution path.  Keeping this guard scoped to plan-only access
    lets operators preview candidates under full_grasp_supervised configuration
    without changing the real-hardware safety configuration.
    """
    if not mode.arm.allow_plan_only:
        raise PermissionError("Plan-only arm requests are disabled by hardware_mode.yaml.")


def attempt_real_arm_execution(mode: HardwareMode, confirmed: bool = False) -> None:
    """Central fail-closed hook for any accidental execution attempt."""
    if not confirmed:
        raise PermissionError("Real arm execution was not confirmed.")
    assert_arm_execution_allowed(mode)
    raise PermissionError("HW4 never executes real arm trajectories.")


def describe_grasp_frame_to_ee_transform(config: dict | None = None) -> str:
    config = config or {}
    if not config.get("enabled", False):
        return "disabled; using GraspGenX grasp frame directly as MoveIt EE target"
    return (
        "enabled; T_base_ee = T_base_grasp * T_grasp_ee, "
        f"translation_m={config.get('translation_m', [0.0, 0.0, 0.0])}, "
        f"rotation_rpy_rad={config.get('rotation_rpy_rad', [0.0, 0.0, 0.0])}"
    )


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
        if stage.get("attempts"):
            print(
                f"{stage['stage']} selected planning attempt: "
                f"{stage.get('attempt_index')} / {len(stage.get('attempts', []))}"
            )
        print(f"{stage['stage']} trajectory points: {summary.get('num_trajectory_points')}")
        diagnostics = (summary.get("trajectory_preview", {}) or {}).get(
            "moveit_diagnostics"
        )
        if diagnostics:
            print(f"{stage['stage']} MoveIt diagnostics:")
            pprint.pprint(diagnostics, sort_dicts=False)
        joint_deltas = _format_joint_deltas(summary)
        if joint_deltas:
            print(f"{stage['stage']} joint deltas: {joint_deltas}")
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
        joint_deltas = _format_joint_deltas(summary)
        if joint_deltas:
            print(f"  joint deltas: {joint_deltas}")
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
        if item.get("failed_stage"):
            details.append(f"failed_stage={item.get('failed_stage')}")
        joint_deltas = _format_joint_deltas(item.get("summary", {}))
        if joint_deltas:
            details.append(f"joint_deltas={joint_deltas}")
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


def _format_joint_deltas(summary: dict) -> str:
    preview = summary.get("trajectory_preview", {}) or {}
    first = preview.get("first_point_positions") or []
    last = preview.get("last_point_positions") or []
    joint_names = summary.get("joint_names") or []
    if not first or not last:
        return ""
    count = min(len(first), len(last), len(joint_names) or len(first))
    items = []
    for index in range(count):
        name = joint_names[index] if index < len(joint_names) else f"joint_{index}"
        delta = float(last[index]) - float(first[index])
        items.append(f"{name}={delta:+.3f}")
    return ", ".join(items)


ROBOTIQ_2F140_TCP_OFFSET_M = [0.0, 0.0, 0.195]


def _pose_point_with_local_offset(pose: dict, offset_m: list[float]) -> dict:
    quaternion = _normalize_quaternion_dict(pose["orientation"])
    rotated = _rotate_vector_by_quaternion(offset_m, quaternion)
    return {
        "frame": pose["frame"],
        "position": {
            axis: float(pose["position"][axis]) + rotated[index]
            for index, axis in enumerate(("x", "y", "z"))
        },
    }


def _vector_between_points(start: dict, end: dict) -> dict:
    return {
        axis: float(end["position"][axis]) - float(start["position"][axis])
        for axis in ("x", "y", "z")
    }


def _print_selected_grasp_debug(
    selected: Any,
    point_cloud: dict,
    preview_config: dict,
    target_frame: str,
    transformer: TFPoseTransformer | None,
    final_poses: dict[str, dict],
) -> None:
    """Print frame/offset diagnostics for the selected GraspGenX candidate."""
    print("\nSelected GraspGenX frame debug:")
    print("object point cloud:")
    pprint.pprint(
        {
            "frame": point_cloud.get("frame"),
            "centroid_m": point_cloud.get("centroid_m"),
            "num_points": point_cloud.get("num_points"),
            "path": point_cloud.get("point_cloud_path"),
        },
        sort_dicts=False,
    )

    localized = localize_graspgenx_candidate(selected, point_cloud)
    grasp_base_poses = create_preview_poses(
        localized,
        preview_config,
        fallback_frame=point_cloud.get("frame") or target_frame or "base_link",
    )
    if target_frame:
        transformed = transform_pose_sequence_to_target(
            grasp_base_poses, transformer, target_frame
        )
        if transformed is None:
            print(
                "Could not transform selected GraspGenX base poses to "
                f"{target_frame} for debug."
            )
            return
        grasp_base_poses = transformed

    grasp_base_pose = grasp_base_poses["grasp"]
    tcp_point = _pose_point_with_local_offset(
        grasp_base_pose, ROBOTIQ_2F140_TCP_OFFSET_M
    )
    print("GraspGenX gripper-base grasp pose before tool0 transform:")
    pprint.pprint(grasp_base_pose, sort_dicts=False)
    print("Estimated Robotiq TCP point from GraspGenX base pose:")
    pprint.pprint(
        {
            **tcp_point,
            "local_offset_m": ROBOTIQ_2F140_TCP_OFFSET_M,
            "note": "For robotiq_2f_140, GraspGenX base -> TCP/fingertip is about +Z 0.195 m.",
        },
        sort_dicts=False,
    )
    print("Pregrasp/grasp/retreat before tool0 transform:")
    pprint.pprint(grasp_base_poses, sort_dicts=False)
    print("Final MoveIt tool0 poses after grasp_frame_to_ee_frame:")
    pprint.pprint(final_poses, sort_dicts=False)
    print("Final grasp tool0 minus estimated TCP point:")
    pprint.pprint(
        _vector_between_points(tcp_point, final_poses["grasp"]),
        sort_dicts=False,
    )
    print()


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
        target_frame = perception_config.get("perception", {}).get("target_frame", "base_link")
        transformer = getattr(camera_backend, "pose_transformer", None)
        if target_frame and (transformer is None or getattr(transformer, "target_frame", None) != target_frame):
            transformer = TFPoseTransformer(node, target_frame=target_frame)
        generation = generate_candidates_with_context(planner, point_cloud, transformer)
        print(f"GraspGenX status: {generation.get('status')}")
        if generation.get("message"):
            print(f"GraspGenX message: {generation.get('message')}")
        candidates = generation.get("candidates", [])
        print(f"GraspGenX candidates returned: {len(candidates)}")
        best_candidate = planner.select_best_candidate(candidates)
        if best_candidate is None:
            print("No valid GraspGenX candidates.")
            return

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
        _print_selected_grasp_debug(
            selected,
            point_cloud,
            preview_config,
            target_frame,
            transformer,
            poses,
        )
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
            stage_plan_kwargs={
                stage: moveit_stage_planning_kwargs(
                    preview_config,
                    stage,
                    selected_plan_kwargs,
                )
                for stage in ("pregrasp", "grasp", "retreat")
            },
            max_replanning_attempts=int(
                preview_config.get("sequence_replanning_attempts", 1)
            ),
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
