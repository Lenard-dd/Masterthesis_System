"""ROS 2 D435i camera backend for RGB-D perception."""

from __future__ import annotations

import copy
import re
import threading
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
from llm_skill_robot.ros_executor_utils import node_has_executor

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
        self._sensor_lock = threading.Lock()
        self._snapshot_lock = threading.Lock()
        self._rgbd_snapshots: dict[str, dict[str, Any]] = {}
        self._rgb_frame_event = threading.Event()
        self._depth_frame_event = threading.Event()
        self._camera_info_event = threading.Event()
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

    def capture_rgb_image(self) -> dict:
        """Save a current RGB snapshot for a read-only vision-language skill."""
        if not self._wait_for_rgb_frame():
            return {
                "status": "NOT_AVAILABLE",
                "message": (
                    f"No RGB frame received within {self.config['wait_timeout_sec']} seconds. "
                    f"{self._debug_message('NOT_AVAILABLE')}"
                ),
            }
        return self._save_latest_rgb_frame()

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
        raw_detections = detector_result.get("detections", [])
        detections, applied_threshold, filtered_count = self._filter_detections_for_query(
            query, raw_detections, min_confidence,
        )
        if detector_status == "SUCCESS" and raw_detections and not detections:
            detector_status = "NOT_FOUND"
        capture_id = str(save_result.get("capture_id", ""))
        objects = self._detections_to_scene_objects(
            detections,
            snapshot=self.get_rgbd_snapshot(capture_id),
            capture_id=capture_id,
            rgb_image_path=str(save_result.get("path", "")),
        )
        self.latest_scene = SceneModel(
            objects=objects,
            frame=self._scene_frame(objects),
            source="d435i_sam3",
        )

        return DetectionResult(
            query=query,
            status=detector_status,
            objects=objects,
            message=(
                f"{detector_result.get('message', '')} "
                f"Applied confidence threshold {applied_threshold:.2f}; filtered {filtered_count}. "
                f"{self._debug_message(detector_status)}"
            ).strip(),
            scene=self.latest_scene,
        )

    def detect_objects(
        self,
        queries: list[str],
        min_confidence: float = 0.5,
        require_pose: bool = True,
    ) -> dict:
        """Localize several requested objects from one saved RGB-D frame."""
        normalized_queries = list(dict.fromkeys(
            query.strip() for query in queries if isinstance(query, str) and query.strip()
        ))
        if not normalized_queries:
            return self._multi_detection_failure("FAILED", "detect_objects requires at least one non-empty query.")
        if not self._wait_for_rgb_frame():
            return self._multi_detection_failure(
                "NOT_AVAILABLE",
                f"No RGB frame received within {self.config['wait_timeout_sec']} seconds. "
                f"{self._debug_message('NOT_AVAILABLE')}",
            )
        save_result = self._save_latest_rgb_frame()
        if save_result["status"] != "SUCCESS":
            return self._multi_detection_failure(save_result["status"], save_result["message"])
        if self.detector_adapter is None:
            return self._multi_detection_failure("NOT_AVAILABLE", "No detector adapter configured.")

        objects: list[SceneObject] = []
        query_results: list[dict[str, Any]] = []
        multi_status: str | None = None
        detect_many = getattr(type(self.detector_adapter), "detect_many", None)
        if callable(detect_many):
            multi_result = self.detector_adapter.detect_many(
                save_result["path"], normalized_queries, min_confidence,
            )
            multi_status = str(multi_result.get("status", "FAILED"))
            raw_query_results = multi_result.get("query_results", [])
        else:
            # Compatibility path for custom detectors that only implement the
            # original one-query protocol.
            raw_query_results = [
                {"query": query, **self.detector_adapter.detect(save_result["path"], query, min_confidence)}
                for query in normalized_queries
            ]
        capture_id = str(save_result.get("capture_id", ""))
        snapshot = self.get_rgbd_snapshot(capture_id)
        for raw_result in raw_query_results:
            query = str(raw_result.get("query", ""))
            detector_status = str(raw_result.get("status", "FAILED"))
            raw_detections = raw_result.get("detections", [])
            detections, applied_threshold, filtered_count = self._filter_detections_for_query(
                query, raw_detections, min_confidence,
            )
            if detector_status == "SUCCESS" and raw_detections and not detections:
                detector_status = "NOT_FOUND"
            query_objects = self._detections_to_scene_objects(
                detections,
                snapshot=snapshot,
                capture_id=capture_id,
                rgb_image_path=str(save_result.get("path", "")),
            )
            for item in query_objects:
                objects.append(item.model_copy(update={
                    "object_id": f"{self._object_id_label(item.label)}_{len(objects) + 1}",
                    "metadata": {**item.metadata, "query": query},
                }))
            query_results.append({
                "query": query,
                "status": detector_status,
                "message": str(raw_result.get("message", "")),
                "object_count": len(query_objects),
                "min_confidence_applied": applied_threshold,
                "filtered_detection_count": filtered_count,
            })

        self.latest_scene = SceneModel(
            objects=objects, frame=self._scene_frame(objects), source="d435i_sam3",
        )
        if objects:
            status = "SUCCESS"
        elif any(item["status"] == "NOT_AVAILABLE" for item in query_results):
            status = "NOT_AVAILABLE"
        elif any(item["status"] == "FAILED" for item in query_results):
            status = "FAILED"
        elif multi_status in {"NOT_AVAILABLE", "FAILED"}:
            # Preserve a worker-level failure even when it could not return
            # individual query results.
            status = multi_status
        else:
            status = "NOT_FOUND"
        return {
            "status": status,
            "message": f"Localized {len(objects)} object(s) from one RGB-D frame for {len(normalized_queries)} query(ies).",
            "objects": [item.model_dump() for item in objects],
            "scene": self.latest_scene.model_dump(),
            "query_results": query_results,
            "image_path": save_result["path"],
        }

    def _multi_detection_failure(self, status: str, message: str) -> dict:
        return {
            "status": status,
            "message": message,
            "objects": [],
            "scene": self.get_scene().model_dump(),
            "query_results": [],
        }

    def get_scene(self) -> SceneModel:
        return self.latest_scene

    def _filter_detections_for_query(
        self, query: str, detections: Any, requested_threshold: float,
    ) -> tuple[list[dict[str, Any]], float, int]:
        """Apply a configurable class threshold without rerunning SAM3."""
        threshold = self._class_confidence_threshold(query, requested_threshold)
        if not isinstance(detections, list):
            return [], threshold, 0
        kept = [
            detection for detection in detections
            if isinstance(detection, dict)
            and self._clamp_confidence(detection.get("confidence", 0.0)) >= threshold
        ]
        return kept, threshold, len(detections) - len(kept)

    def _class_confidence_threshold(self, query: str, requested_threshold: float) -> float:
        try:
            threshold = self._clamp_confidence(requested_threshold)
        except (TypeError, ValueError):
            threshold = 0.0
        configured = self.perception_config.get("class_confidence_thresholds", {})
        if not isinstance(configured, dict):
            return threshold
        query_tokens = self._confidence_class_tokens(query)
        for class_name, raw_threshold in configured.items():
            class_tokens = self._confidence_class_tokens(str(class_name))
            if not class_tokens or not class_tokens.issubset(query_tokens):
                continue
            try:
                candidate = float(raw_threshold)
            except (TypeError, ValueError):
                continue
            if 0.0 <= candidate <= 1.0:
                threshold = max(threshold, candidate)
        return threshold

    @staticmethod
    def _confidence_class_tokens(value: str) -> set[str]:
        tokens = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", str(value).casefold())
        return {
            token[:-1] if token.endswith("s") and len(token) > 3 and not token.endswith("ss") else token
            for token in tokens
        }

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
        with self._sensor_lock:
            self.latest_rgb_msg = msg
        self._rgb_frame_event.set()

    def _on_depth_image(self, msg) -> None:
        with self._sensor_lock:
            self.latest_depth_msg = msg
        self._depth_frame_event.set()

    def _on_camera_info(self, msg) -> None:
        with self._sensor_lock:
            self.latest_camera_info_msg = msg
        self._camera_info_event.set()

    def _wait_for_rgb_frame(self) -> bool:
        timeout = max(0.0, float(self.config["wait_timeout_sec"]))
        if self.latest_rgb_msg is not None:
            return True

        # RosWorker owns the GUI node's executor.  Waiting on an event lets its
        # callback thread deliver the frame without moving the node to rclpy's
        # global executor.  Standalone demos retain their legacy self-spin path.
        if node_has_executor(self.node):
            self._rgb_frame_event.wait(timeout=timeout)
            return self.latest_rgb_msg is not None

        deadline = time.monotonic() + timeout
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
        # Freeze the latest RGB-D inputs before SAM3 starts.  Detector inference
        # can take several seconds, during which ROS callbacks continue updating
        # ``latest_depth_msg``.  Keeping this small in-memory snapshot prevents a
        # mask from the saved RGB image being projected through a later depth
        # frame.  It adds no detector call and therefore negligible latency.
        with self._sensor_lock:
            rgb_msg = self.latest_rgb_msg
            depth_msg = self.latest_depth_msg
            # ROS messages are replaced by callbacks rather than mutated, so
            # retaining this reference safely binds the matching intrinsics
            # without relying on message deepcopy support.
            camera_info = self.latest_camera_info_msg
        if rgb_msg is None:
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
                rgb_msg,
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

        capture_id = f"rgbd-{time.time_ns()}"
        self._store_rgbd_snapshot(
            capture_id,
            depth_frame=self._depth_image_from_message(depth_msg),
            camera_info=camera_info,
        )
        return {
            "status": "SUCCESS",
            "message": f"Saved latest RGB frame to {latest_rgb_path}.",
            "path": str(latest_rgb_path),
            "capture_id": capture_id,
        }

    def _detections_to_scene_objects(
        self,
        detections: Any,
        *,
        snapshot: dict[str, Any] | None = None,
        capture_id: str = "",
        rgb_image_path: str = "",
    ) -> list[SceneObject]:
        if not isinstance(detections, list):
            return []

        snapshot_depth = snapshot.get("depth_frame") if snapshot else None
        snapshot_camera_info = snapshot.get("camera_info") if snapshot else None
        depth_image = (
            snapshot_depth if snapshot_depth is not None else self._get_latest_depth_image()
        )
        camera_info = (
            snapshot_camera_info
            if snapshot_camera_info is not None
            else self.latest_camera_info_msg
        )
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
            if capture_id:
                metadata["rgbd_capture_id"] = capture_id
            if rgb_image_path:
                metadata["rgb_image_path"] = rgb_image_path
            pose = self.pose_estimator.estimate_pose_from_detection(
                detection=detection,
                depth_image=depth_image,
                camera_info=camera_info,
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
        with self._sensor_lock:
            depth_msg = self.latest_depth_msg
        return self._depth_image_from_message(depth_msg)

    def _depth_image_from_message(self, depth_msg):
        if depth_msg is None:
            return None

        if hasattr(depth_msg, "image"):
            return copy.deepcopy(depth_msg)

        if not hasattr(depth_msg, "encoding"):
            return copy.deepcopy(depth_msg)

        try:
            from cv_bridge import CvBridge
        except ImportError:
            return None

        if self._bridge is None:
            self._bridge = CvBridge()

        try:
            image = self._bridge.imgmsg_to_cv2(
                depth_msg,
                desired_encoding="passthrough",
            )
        except Exception:
            return None

        return {
            "image": image.copy() if hasattr(image, "copy") else image,
            "encoding": getattr(depth_msg, "encoding", ""),
        }

    def _store_rgbd_snapshot(
        self, capture_id: str, *, depth_frame: Any, camera_info: Any,
    ) -> None:
        with self._snapshot_lock:
            self._rgbd_snapshots[capture_id] = {
                "depth_frame": depth_frame,
                "camera_info": camera_info,
            }
            # A few immutable frames cover sequential batches without allowing
            # long-running GUI sessions to accumulate camera-sized arrays.
            while len(self._rgbd_snapshots) > 8:
                self._rgbd_snapshots.pop(next(iter(self._rgbd_snapshots)))

    def get_rgbd_snapshot(self, capture_id: str) -> dict[str, Any] | None:
        """Return the RGB-associated depth inputs for a localized object."""
        if not capture_id:
            return None
        with self._snapshot_lock:
            return self._rgbd_snapshots.get(capture_id)

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
