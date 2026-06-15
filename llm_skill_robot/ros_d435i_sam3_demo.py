"""Interactive ROS 2 demo for D435i RGB frames plus SAM3 subprocess detection."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import yaml

from llm_skill_robot.perception.d435i_backend import D435iBackend
from llm_skill_robot.perception.sam3_subprocess_adapter import SAM3SubprocessAdapter


def _load_perception_config() -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    config_path = repo_root / "config" / "perception.yaml"
    if not config_path.exists():
        return {}

    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _wait_for_camera_messages(backend: D435iBackend, node, rclpy_module) -> None:
    timeout_sec = float(backend.config.get("wait_timeout_sec", 5.0))
    deadline = time.monotonic() + timeout_sec

    while time.monotonic() < deadline:
        if (
            backend.latest_rgb_msg is not None
            and backend.latest_depth_msg is not None
            and backend.latest_camera_info_msg is not None
        ):
            return
        rclpy_module.spin_once(node, timeout_sec=0.1)


def _print_camera_status(backend: D435iBackend) -> None:
    print("\nCamera status:")
    print(f"RGB received: {backend.get_rgb_frame().get('received')}")
    print(f"Depth received: {backend.get_depth_frame().get('received')}")
    print(f"CameraInfo received: {backend.get_camera_info().get('received')}")


def _print_detection_result(result) -> None:
    scene = result.scene

    print("\nDetection result:")
    print(f"status: {result.status}")
    print(f"message: {result.message}")
    print(f"number of objects: {len(result.objects)}")

    for scene_object in result.objects:
        metadata = scene_object.metadata or {}
        print("\nObject:")
        print(f"object_id: {scene_object.object_id}")
        print(f"label: {scene_object.label}")
        print(f"confidence: {scene_object.confidence}")
        print(f"bbox: {metadata.get('bbox_xyxy')}")
        print(f"mask_available: {scene_object.mask_available}")
        print(f"mask_path: {metadata.get('mask_path')}")
        print(f"overlay_path: {metadata.get('overlay_path')}")
        print(f"pose_available: {scene_object.pose_available}")
        if scene_object.pose_available and scene_object.pose is not None:
            print("pose:")
            print(f"  frame: {scene_object.pose.frame}")
            print(f"  x: {scene_object.pose.x}")
            print(f"  y: {scene_object.pose.y}")
            print(f"  z: {scene_object.pose.z}")

    print("\nScene summary:")
    print(f"frame: {scene.frame if scene else None}")
    print(f"source: {scene.source if scene else None}")
    print(f"object_count: {len(scene.objects) if scene else 0}")


def main() -> None:
    try:
        import rclpy
    except ImportError:
        print("Error: rclpy not available. Please source ROS 2 setup.", file=sys.stderr)
        sys.exit(1)

    config = _load_perception_config()
    node = None

    try:
        rclpy.init()
        node = rclpy.create_node("d435i_sam3_demo")
        detector = SAM3SubprocessAdapter(config.get("sam3_subprocess", {}))
        backend = D435iBackend(
            node=node,
            config=config,
            detector_adapter=detector,
        )

        query = input("\nEnter object query:\n> ").strip()
        if not query:
            print("No object query entered.", file=sys.stderr)
            return

        print("\nWaiting for D435i RGB/depth/camera_info messages...")
        _wait_for_camera_messages(backend, node, rclpy)
        _print_camera_status(backend)

        result = backend.detect_object(query=query)
        _print_detection_result(result)

    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as exc:  # noqa: BLE001 - demo should fail with a clear message.
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
