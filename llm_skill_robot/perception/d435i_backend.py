"""ROS 2 D435i camera backend for RGB-D perception."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import yaml

from llm_skill_robot.perception.perception_backend_base import BasePerceptionBackend
from llm_skill_robot.perception.object_pose_estimator import ObjectPoseEstimator
from llm_skill_robot.perception.scene_model import (
    DetectionResult,
    SceneModel,
    SceneObject,
)
from llm_skill_robot.perception.tf_pose_transformer import TFPoseTransformer

try:  # ROS imports are optional so normal pytest does not need a ROS graph.
    import rclpy
except ImportError:  # pragma: no cover - depends on local ROS install.
    rclpy = None

try:
    from sensor_msgs.msg import CameraInfo, Image
except ImportError:  # pragma: no cover - depends on local ROS install.
    CameraInfo = None
    Image = None


class D435iBackend(BasePerceptionBackend):
    """Subscribe to D435i RGB, aligned depth, and camera info topics."""

    DEFAULT_CONFIG: dict[str, Any] = {
        "rgb_topic": "/camera/camera/color/image_raw",
        "depth_topic": "/camera/camera/aligned_depth_to_color/image_raw",
        "camera_info_topic": "/camera/camera/color/camera_info",
        "rgb_frame": "camera_color_optical_frame",
        "depth_frame": "camera_color_optical_frame",
        "wait_timeout_sec": 5.0,
        "runtime_dir": "data/perception_runtime",
    }

    def __init__(
        self,
        node=None,
        config: dict | None = None,
        detector_adapter=None,
        pose_transformer=None,
    ):
        self.node = node
        self.full_config = self._load_full_config(config)
        self.config = self._load_config(config)
        self.perception_config = self._load_perception_config(self.full_config)
        self.detector_adapter = detector_adapter or self._make_default_detector(config)
        self.latest_rgb_msg = None
        self.latest_depth_msg = None
        self.latest_camera_info_msg = None
        self.latest_scene = SceneModel(source="d435i_sam3")
        self.pose_estimator = ObjectPoseEstimator()
        self.pose_transformer = pose_transformer or self._make_pose_transformer()
        self._bridge = None
        self._subscriptions = []

        self._create_subscriptions()

    def get_rgb_frame(self) -> dict:
        return self._frame_result(
            name="RGB",
            key="image",
            value=self.latest_rgb_msg,
        )

    def get_depth_frame(self) -> dict:
        return self._frame_result(
            name="depth",
            key="image",
            value=self.latest_depth_msg,
        )

    def get_camera_info(self) -> dict:
        return self._frame_result(
            name="camera_info",
            key="camera_info",
            value=self.latest_camera_info_msg,
        )

    def detect_object(
        self,
        query: str,
        min_confidence: float = 0.5,
        require_pose: bool = True,
    ) -> DetectionResult:
        if not self._wait_for_rgb_frame():
            scene = self.get_scene()
            return DetectionResult(
                query=query,
                status="NOT_AVAILABLE",
                objects=[],
                message=f"No RGB frame received within {self.config['wait_timeout_sec']} seconds. {self._debug_message('NOT_AVAILABLE')}",
                scene=scene,
            )

        save_result = self._save_latest_rgb_frame()
        if save_result["status"] != "SUCCESS":
            return DetectionResult(
                query=query,
                status=save_result["status"],
                objects=[],
                message=f"{save_result['message']} {self._debug_message(save_result['status'])}",
                scene=self.get_scene(),
            )

        if self.detector_adapter is None:
            return DetectionResult(
                query=query,
                status="NOT_AVAILABLE",
                objects=[],
                message=f"No detector adapter configured. {self._debug_message('NOT_AVAILABLE')}",
                scene=self.get_scene(),
            )

        detector_result = self.detector_adapter.detect(
            save_result["path"],
            query,
            min_confidence,
        )
        detector_status = detector_result.get("status", "FAILED")
        detections = detector_result.get("detections", [])
        objects = self._detections_to_scene_objects(detections)
        self.latest_scene = SceneModel(
            objects=objects,
            frame=self._scene_frame(objects),
            source="d435i_sam3",
        )

        return DetectionResult(
            query=query,
            status=detector_status,
            objects=objects,
            message=f"{detector_result.get('message', '')} {self._debug_message(detector_status)}".strip(),
            scene=self.latest_scene,
        )

    def get_scene(self) -> SceneModel:
        return self.latest_scene

    @classmethod
    def _load_full_config(cls, config: dict | None) -> dict[str, Any]:
        if config is None:
            config_path = Path(__file__).resolve().parents[2] / "config" / "perception.yaml"
            if config_path.exists():
                with config_path.open("r", encoding="utf-8") as handle:
                    return yaml.safe_load(handle) or {}
            return {}
        return config

    @classmethod
    def _load_config(cls, config: dict | None) -> dict[str, Any]:
        loaded = cls._load_full_config(config)

        d435i_config = loaded.get("d435i", loaded) if isinstance(loaded, dict) else {}
        return {**cls.DEFAULT_CONFIG, **(d435i_config or {})}

    @staticmethod
    def _load_perception_config(config: dict | None) -> dict[str, Any]:
        if isinstance(config, dict):
            return config.get("perception", {}) or {}
        return {}

    @staticmethod
    def _make_default_detector(config: dict | None):
        try:
            from llm_skill_robot.perception.sam3_subprocess_adapter import (
                SAM3SubprocessAdapter,
            )
        except ImportError:
            return None

        return SAM3SubprocessAdapter(config)

    def _make_pose_transformer(self):
        target_frame = self.perception_config.get("target_frame")
        camera_frame = self.config.get("depth_frame") or self.config.get("rgb_frame")
        if self.node is None or not target_frame or target_frame == camera_frame:
            return None
        return TFPoseTransformer(self.node, target_frame=target_frame)

    def _create_subscriptions(self) -> None:
        if self.node is None or Image is None or CameraInfo is None:
            return

        self._subscriptions = [
            self.node.create_subscription(
                Image,
                self.config["rgb_topic"],
                self._on_rgb_image,
                10,
            ),
            self.node.create_subscription(
                Image,
                self.config["depth_topic"],
                self._on_depth_image,
                10,
            ),
            self.node.create_subscription(
                CameraInfo,
                self.config["camera_info_topic"],
                self._on_camera_info,
                10,
            ),
        ]

    def _on_rgb_image(self, msg) -> None:
        self.latest_rgb_msg = msg

    def _on_depth_image(self, msg) -> None:
        self.latest_depth_msg = msg

    def _on_camera_info(self, msg) -> None:
        self.latest_camera_info_msg = msg

    def _wait_for_rgb_frame(self) -> bool:
        deadline = time.monotonic() + float(self.config["wait_timeout_sec"])
        while self.latest_rgb_msg is None and time.monotonic() < deadline:
            if self.node is not None and rclpy is not None:
                try:
                    rclpy.spin_once(self.node, timeout_sec=0.05)
                except Exception:
                    time.sleep(0.05)
            else:
                time.sleep(0.05)
        return self.latest_rgb_msg is not None

    def _save_latest_rgb_frame(self) -> dict:
        if self.latest_rgb_msg is None:
            return {
                "status": "NOT_AVAILABLE",
                "message": "No RGB message is available to save.",
            }

        try:
            from cv_bridge import CvBridge
        except ImportError:
            return {
                "status": "NOT_AVAILABLE",
                "message": "cv_bridge is not available; cannot convert D435i RGB image.",
            }

        try:
            import cv2
        except ImportError:
            return {
                "status": "NOT_AVAILABLE",
                "message": "opencv-python is not available; cannot save D435i RGB image.",
            }

        if self._bridge is None:
            self._bridge = CvBridge()

        runtime_dir = Path(str(self.config["runtime_dir"]))
        latest_rgb_path = runtime_dir / "latest_rgb.png"
        try:
            runtime_dir.mkdir(parents=True, exist_ok=True)
            image = self._bridge.imgmsg_to_cv2(
                self.latest_rgb_msg,
                desired_encoding="bgr8",
            )
            if not cv2.imwrite(str(latest_rgb_path), image):
                return {
                    "status": "FAILED",
                    "message": f"Failed to write latest RGB frame to {latest_rgb_path}.",
                }
        except Exception as exc:
            return {
                "status": "FAILED",
                "message": f"Failed to convert or save latest RGB frame: {exc}",
            }

        return {
            "status": "SUCCESS",
            "message": f"Saved latest RGB frame to {latest_rgb_path}.",
            "path": str(latest_rgb_path),
        }

    def _detections_to_scene_objects(self, detections: Any) -> list[SceneObject]:
        if not isinstance(detections, list):
            return []

        depth_image = self._get_latest_depth_image()
        objects: list[SceneObject] = []
        for index, detection in enumerate(detections, start=1):
            if not isinstance(detection, dict):
                continue

            label = str(detection.get("label") or f"object_{index}")
            confidence = self._clamp_confidence(detection.get("confidence", 0.0))
            mask_path = detection.get("mask_path")
            overlay_path = detection.get("overlay_path")
            metadata = dict(detection.get("metadata") or {})
            metadata.update(
                {
                    "bbox_xyxy": detection.get("bbox_xyxy"),
                    "mask_path": mask_path,
                    "overlay_path": overlay_path,
                }
            )
            pose = self.pose_estimator.estimate_pose_from_detection(
                detection=detection,
                depth_image=depth_image,
                camera_info=self.latest_camera_info_msg,
                target_frame=self.config.get(
                    "depth_frame",
                    self.config.get("rgb_frame", "camera_color_optical_frame"),
                ),
            )
            pose = self._transform_pose_if_needed(pose, metadata)

            objects.append(
                SceneObject(
                    object_id=f"{self._object_id_label(label)}_{index}",
                    label=label,
                    confidence=confidence,
                    mask_available=bool(mask_path and Path(str(mask_path)).exists()),
                    pose_available=pose is not None,
                    pose=pose,
                    graspable=False,
                    source="d435i_sam3",
                    metadata=metadata,
                )
            )

        return objects

    def _transform_pose_if_needed(self, pose, metadata: dict):
        if pose is None:
            return None

        target_frame = self.perception_config.get("target_frame")
        if not target_frame or target_frame == pose.frame:
            return pose

        transformed_pose = None
        if self.pose_transformer is not None:
            transformed_pose = self.pose_transformer.transform_pose(pose)

        if transformed_pose is not None:
            return transformed_pose

        warnings = metadata.setdefault("warnings", [])
        warning = f"TF transform to {target_frame} unavailable."
        if warning not in warnings:
            warnings.append(warning)
        return pose

    def _scene_frame(self, objects: list[SceneObject]) -> str:
        if objects and objects[0].pose_available and objects[0].pose is not None:
            return objects[0].pose.frame
        return self.config.get("rgb_frame", "camera_color_optical_frame")

    @staticmethod
    def _object_id_label(label: str) -> str:
        normalized = " ".join(str(label).strip().split())
        return normalized or "object"

    def _get_latest_depth_image(self):
        if self.latest_depth_msg is None:
            return None

        if hasattr(self.latest_depth_msg, "image"):
            return self.latest_depth_msg

        if not hasattr(self.latest_depth_msg, "encoding"):
            return self.latest_depth_msg

        try:
            from cv_bridge import CvBridge
        except ImportError:
            return None

        if self._bridge is None:
            self._bridge = CvBridge()

        try:
            image = self._bridge.imgmsg_to_cv2(
                self.latest_depth_msg,
                desired_encoding="passthrough",
            )
        except Exception:
            return None

        return {
            "image": image,
            "encoding": getattr(self.latest_depth_msg, "encoding", ""),
        }

    @staticmethod
    def _clamp_confidence(value: Any) -> float:
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, confidence))

    def _debug_message(self, detector_status: str) -> str:
        return (
            f"Debug: RGB received {'yes' if self.latest_rgb_msg is not None else 'no'}; "
            f"depth received {'yes' if self.latest_depth_msg is not None else 'no'}; "
            f"camera_info received {'yes' if self.latest_camera_info_msg is not None else 'no'}; "
            f"detector status {detector_status}."
        )

    @staticmethod
    def _frame_result(name: str, key: str, value) -> dict:
        if value is None:
            return {
                "status": "NOT_AVAILABLE",
                "message": f"{name} frame has not been received.",
                key: None,
                "received": False,
            }
        return {
            "status": "SUCCESS",
            "message": f"{name} frame received.",
            key: value,
            "received": True,
        }
