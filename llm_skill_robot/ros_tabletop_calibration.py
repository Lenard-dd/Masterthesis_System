"""Read-only D435i tabletop-plane calibration utility."""

from __future__ import annotations

import pprint
import sys
import time
from typing import Any

import yaml

from llm_skill_robot.calibration.table_plane_calibrator import TablePlaneCalibrator
from llm_skill_robot.perception.perception_factory import create_perception_pipeline
from llm_skill_robot.perception.tf_pose_transformer import TFPoseTransformer
from llm_skill_robot.utils import get_config_dir


def _load_yaml(name: str) -> dict:
    with (get_config_dir() / name).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _matrix_from_transform(transform: Any) -> list[list[float]]:
    translation = transform.transform.translation
    rotation = transform.transform.rotation
    x, y, z, w = (float(rotation.x), float(rotation.y), float(rotation.z), float(rotation.w))
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), float(translation.x)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), float(translation.y)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), float(translation.z)],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _wait_for_depth_inputs(backend, node, rclpy_module, timeout_sec: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if getattr(backend, "latest_depth_msg", None) is not None and getattr(backend, "latest_camera_info_msg", None) is not None:
            getter = getattr(backend, "_get_latest_depth_image", None)
            if callable(getter) and getter() is not None:
                return True
        rclpy_module.spin_once(node, timeout_sec=0.1)
    return False


def main() -> None:
    try:
        import rclpy
    except ImportError:
        print("Error: source the ROS 2 environment first.", file=sys.stderr)
        return
    perception_config = _load_yaml("perception.yaml")
    calibration_config = _load_yaml("tabletop_calibration.yaml")
    node = None
    try:
        rclpy.init()
        node = rclpy.create_node("ros_tabletop_calibration")
        pipeline = create_perception_pipeline(node=node, config=perception_config)
        backend = pipeline.backend
        print("TABLETOP CALIBRATION — read-only: no arm or gripper command will be sent.")
        print("Clear the configured ROI so it contains only the tabletop, then wait for depth input.")
        if not _wait_for_depth_inputs(backend, node, rclpy):
            print("Error: no usable aligned depth image and camera_info were received.", file=sys.stderr)
            return
        source_frame = str(getattr(backend, "config", {}).get("depth_frame", "camera_color_optical_frame"))
        transformer = TFPoseTransformer(node, target_frame="base_link")
        transform = transformer._lookup_transform(source_frame, timeout_sec=2.0)
        if transform is None:
            print(f"Error: TF from {source_frame} to base_link is unavailable.", file=sys.stderr)
            return
        result = TablePlaneCalibrator(calibration_config).calibrate(
            backend._get_latest_depth_image(),
            backend.latest_camera_info_msg,
            _matrix_from_transform(transform),
        )
        pprint.pprint(result, sort_dicts=False)
        if result.get("success"):
            print("\nCopy suggested_config into config/tabletop_safety.yaml after physical verification.")
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
