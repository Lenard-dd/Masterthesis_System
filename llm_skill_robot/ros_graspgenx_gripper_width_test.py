"""Test only a GraspGenX-recommended width on the real Robotiq gripper."""

from __future__ import annotations

import pprint
import sys
from typing import Any

import yaml

from llm_skill_robot.grasping.graspgenx_grasp_planner import GraspGenXGraspPlanner
from llm_skill_robot.grasping.graspgenx_subprocess_adapter import GraspGenXSubprocessAdapter
from llm_skill_robot.perception.object_point_cloud_builder import ObjectPointCloudBuilder
from llm_skill_robot.perception.perception_factory import create_perception_pipeline
from llm_skill_robot.robot.robotiq_2f140_real_backend import Robotiq2F140RealBackend
from llm_skill_robot.ros_real_gripper_test import confirmation_is_yes
from llm_skill_robot.safety.hardware_mode import (
    HardwareMode,
    clamp_or_reject_gripper_command,
    load_hardware_mode,
)
from llm_skill_robot.utils import get_config_dir


WIDTH_MISSING_MESSAGE = "Selected candidate does not provide gripper_width_m."


def _candidate_value(candidate: Any, name: str, default: Any = None) -> Any:
    return candidate.get(name, default) if isinstance(candidate, dict) else getattr(candidate, name, default)


def extract_candidate_width(candidate: Any) -> float:
    value = _candidate_value(candidate, "gripper_width_m")
    if value is None:
        raise ValueError(WIDTH_MISSING_MESSAGE)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(WIDTH_MISSING_MESSAGE) from exc


def validate_candidate_width(
    candidate: Any,
    hardware_mode: HardwareMode,
    speed_mps: float,
    force_N: float,
) -> dict[str, float]:
    return clamp_or_reject_gripper_command(
        width_m=extract_candidate_width(candidate),
        speed_mps=speed_mps,
        force_N=force_N,
        mode=hardware_mode,
    )


def command_candidate_width_if_confirmed(
    candidate: Any,
    backend: Robotiq2F140RealBackend,
    hardware_mode: HardwareMode,
    confirmation: str,
    speed_mps: float,
    force_N: float,
) -> dict:
    """Validate and send only a gripper-width command; no arm API is referenced."""
    try:
        command = validate_candidate_width(
            candidate, hardware_mode, speed_mps, force_N
        )
    except (ValueError, PermissionError) as exc:
        return {"success": False, "status": "REJECTED", "message": str(exc)}
    if not confirmation_is_yes(confirmation):
        return {
            "success": False,
            "status": "CANCELLED",
            "message": "Confirmation was not YES. No real gripper command was sent.",
        }
    return backend.set_gripper_width(**command, confirmed=True)


def _load_yaml(filename: str) -> dict:
    path = get_config_dir() / filename
    if not path.is_file():
        raise FileNotFoundError(f"Required config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _print_hardware_summary(mode: HardwareMode) -> None:
    print("Real camera: enabled" if mode.camera.enabled else "Real camera: disabled")
    print(
        "Real gripper: enabled"
        if mode.gripper.real_gripper_enabled and mode.gripper.allow_gripper_execution
        else "Real gripper: disabled"
    )
    arm_disabled = not mode.arm.real_arm_enabled and not mode.arm.allow_arm_execution
    print(
        "Real arm execution: disabled"
        if arm_disabled
        else "Real arm execution: ENABLED"
    )


def _select_highest_confidence(objects: list[Any]) -> Any | None:
    if not objects:
        return None
    return max(objects, key=lambda obj: float(_candidate_value(obj, "confidence", 0.0)))


def _pose_frame(candidate: Any, fallback: str | None = None) -> str | None:
    direct = _candidate_value(candidate, "pose_frame")
    if direct:
        return str(direct)
    pose = _candidate_value(candidate, "pose", {}) or {}
    return _candidate_value(pose, "frame", fallback)


def _print_selection(
    selected_object: Any,
    point_cloud: dict,
    candidates: list[Any],
    selected_candidate: Any,
    width_m: float,
) -> None:
    print(f"object_id: {_candidate_value(selected_object, 'object_id')}")
    print(f"confidence: {_candidate_value(selected_object, 'confidence')}")
    print(f"point_cloud_path: {point_cloud.get('point_cloud_path')}")
    print(f"num_points: {point_cloud.get('num_points')}")
    print(f"number of candidates: {len(candidates)}")
    print(
        "selected candidate id: "
        f"{_candidate_value(selected_candidate, 'candidate_id', _candidate_value(selected_candidate, 'id'))}"
    )
    print(f"score: {_candidate_value(selected_candidate, 'score')}")
    print(f"gripper_width_m: {width_m}")
    print(f"pose frame: {_pose_frame(selected_candidate, point_cloud.get('frame'))}")


def main() -> None:
    try:
        import rclpy
    except ImportError:
        print("Error: rclpy not available. Please source ROS 2 setup.", file=sys.stderr)
        return

    hardware_mode = load_hardware_mode()
    perception_config = _load_yaml("perception.yaml")
    grasping_config = _load_yaml("grasping.yaml")
    gripper_document = _load_yaml("gripper.yaml")
    gripper_config = gripper_document.get("gripper", {})
    _print_hardware_summary(hardware_mode)

    if hardware_mode.arm.real_arm_enabled or hardware_mode.arm.allow_arm_execution:
        print("Refusing to start because real arm execution is not disabled.")
        return

    node = None
    try:
        rclpy.init()
        node = rclpy.create_node("ros_graspgenx_gripper_width_test")
        perception_pipeline = create_perception_pipeline(
            node=node, config=perception_config
        )
        point_cloud_builder = ObjectPointCloudBuilder(
            grasping_config.get("grasping", {}).get("point_cloud", {})
        )
        adapter = GraspGenXSubprocessAdapter(
            grasping_config.get("grasping", {}).get("graspgenx_subprocess", {})
        )
        planner = GraspGenXGraspPlanner(adapter)
        gripper_backend = Robotiq2F140RealBackend(
            node=node,
            config=gripper_document,
            hardware_mode=hardware_mode,
        )

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
        depth_frame = depth_getter() if callable(depth_getter) else None
        camera_info = getattr(camera_backend, "latest_camera_info_msg", None)
        point_cloud = point_cloud_builder.build(
            selected_object,
            depth_frame,
            camera_info,
            frame=getattr(camera_backend, "config", {}).get("depth_frame"),
        )
        if not point_cloud.get("success", False) or not point_cloud.get("point_cloud_path"):
            print(f"No point cloud. {point_cloud.get('message', '')}".strip())
            return

        generation = planner.generate_candidates(point_cloud["point_cloud_path"])
        candidates = generation.get("candidates", [])
        if not candidates:
            print(f"No GraspGenX candidates. {generation.get('message', '')}".strip())
            return
        selected_candidate = planner.select_best_candidate(candidates)
        if selected_candidate is None:
            print("No valid GraspGenX candidates.")
            return
        try:
            width_m = extract_candidate_width(selected_candidate)
        except ValueError as exc:
            print(str(exc))
            return

        speed = float(gripper_config.get("default_speed_mps", 0.02))
        force = float(gripper_config.get("default_force_N", 10.0))
        try:
            validated = validate_candidate_width(
                selected_candidate, hardware_mode, speed, force
            )
        except (ValueError, PermissionError) as exc:
            print(f"Candidate gripper width rejected: {exc}")
            return

        _print_selection(
            selected_object, point_cloud, candidates, selected_candidate, width_m
        )
        print("Real arm execution is disabled. Only gripper width will be tested.")
        confirmation = input(
            f"Set REAL gripper to candidate width {width_m} m? Type YES to continue:"
        )
        result = command_candidate_width_if_confirmed(
            selected_candidate,
            gripper_backend,
            hardware_mode,
            confirmation,
            validated["speed_mps"],
            validated["force_N"],
        )
        if result.get("status") == "CANCELLED":
            print(result["message"])
            return
        print("Result:")
        pprint.pprint(result, sort_dicts=False)
    except KeyboardInterrupt:
        print("\nInterrupted. No further commands will be sent.")
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
