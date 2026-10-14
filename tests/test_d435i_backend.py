from unittest.mock import Mock
import threading
import time
from types import SimpleNamespace

import llm_skill_robot.perception.d435i_backend as d435i_module
import numpy as np
from llm_skill_robot.perception.d435i_backend import D435iBackend
from llm_skill_robot.perception.scene_model import DetectionResult, SceneModel


def test_d435i_backend_imports_without_ros_graph():
    backend = D435iBackend(config={"wait_timeout_sec": 0.0}, detector_adapter=None)

    assert isinstance(backend, D435iBackend)
    assert backend.get_rgb_frame()["status"] == "NOT_AVAILABLE"
    assert backend.get_depth_frame()["status"] == "NOT_AVAILABLE"
    assert backend.get_camera_info()["status"] == "NOT_AVAILABLE"


def test_d435i_backend_loads_config_defaults():
    backend = D435iBackend(config={}, detector_adapter=None)

    assert backend.config["rgb_topic"] == "/camera/camera/color/image_raw"
    assert backend.config["depth_topic"] == "/camera/camera/aligned_depth_to_color/image_raw"
    assert backend.config["camera_info_topic"] == "/camera/camera/color/camera_info"
    assert backend.config["runtime_dir"] == "data/perception_runtime"


def test_d435i_backend_uses_nested_d435i_config():
    backend = D435iBackend(
        config={
            "d435i": {
                "rgb_topic": "/rgb",
                "depth_topic": "/depth",
                "camera_info_topic": "/info",
                "wait_timeout_sec": 0.0,
            }
        },
        detector_adapter=None,
    )

    assert backend.config["rgb_topic"] == "/rgb"
    assert backend.config["depth_topic"] == "/depth"
    assert backend.config["camera_info_topic"] == "/info"


def test_d435i_backend_creates_subscriptions_with_mocked_ros_types(monkeypatch):
    image_type = object()
    camera_info_type = object()
    monkeypatch.setattr(d435i_module, "Image", image_type)
    monkeypatch.setattr(d435i_module, "CameraInfo", camera_info_type)
    node = Mock()
    node.create_subscription.side_effect = ["rgb_sub", "depth_sub", "info_sub"]

    backend = D435iBackend(
        node=node,
        config={
            "rgb_topic": "/rgb",
            "depth_topic": "/depth",
            "camera_info_topic": "/info",
        },
        detector_adapter=None,
    )

    assert backend._subscriptions == ["rgb_sub", "depth_sub", "info_sub"]
    assert node.create_subscription.call_count == 3
    assert node.create_subscription.call_args_list[0].args[0] is image_type
    assert node.create_subscription.call_args_list[0].args[1] == "/rgb"
    assert node.create_subscription.call_args_list[1].args[1] == "/depth"
    assert node.create_subscription.call_args_list[2].args[0] is camera_info_type
    assert node.create_subscription.call_args_list[2].args[1] == "/info"


def test_detect_object_without_rgb_returns_not_available():
    backend = D435iBackend(config={"wait_timeout_sec": 0.0}, detector_adapter=None)

    result = backend.detect_object("red cube")

    assert isinstance(result, DetectionResult)
    assert result.status == "NOT_AVAILABLE"
    assert "No RGB frame received" in result.message
    assert "RGB received no" in result.message
    assert isinstance(result.scene, SceneModel)


def test_wait_for_rgb_does_not_spin_node_owned_by_external_executor(monkeypatch):
    class FakeRclpy:
        @staticmethod
        def spin_once(*_args, **_kwargs):
            raise AssertionError("shared ROS node must not be spun by perception")

    monkeypatch.setattr(d435i_module, "rclpy", FakeRclpy())
    backend = D435iBackend(
        config={"wait_timeout_sec": 0.2}, detector_adapter=None,
    )
    backend.node = SimpleNamespace(executor=object())

    producer = threading.Thread(
        target=lambda: (time.sleep(0.01), backend._on_rgb_image(object()))
    )
    producer.start()
    try:
        assert backend._wait_for_rgb_frame() is True
    finally:
        producer.join(timeout=1.0)


def test_detect_object_converts_detector_detections(tmp_path):
    mask_path = tmp_path / "mask.png"
    mask_path.write_text("mask", encoding="utf-8")
    detector = Mock()
    detector.detect.return_value = {
        "status": "SUCCESS",
        "message": "detected",
        "detections": [
            {
                "label": "red cube",
                "confidence": 0.85,
                "bbox_xyxy": [1, 2, 3, 4],
                "mask_path": str(mask_path),
                "overlay_path": "overlay.png",
                "metadata": {"source_id": "sam3"},
            }
        ],
    }
    backend = D435iBackend(config={"wait_timeout_sec": 0.0}, detector_adapter=detector)
    backend.latest_rgb_msg = object()
    backend._save_latest_rgb_frame = Mock(
        return_value={
            "status": "SUCCESS",
            "message": "saved",
            "path": "data/perception_runtime/latest_rgb.png",
        }
    )

    result = backend.detect_object("red cube", min_confidence=0.6)

    detector.detect.assert_called_once_with(
        "data/perception_runtime/latest_rgb.png",
        "red cube",
        0.6,
    )
    assert result.status == "SUCCESS"
    assert result.objects[0].object_id == "red cube_1"
    assert result.objects[0].label == "red cube"
    assert result.objects[0].source == "d435i_sam3"
    assert result.objects[0].mask_available is True
    assert result.objects[0].pose_available is False
    assert result.objects[0].metadata["bbox_xyxy"] == [1, 2, 3, 4]
    assert "detector status SUCCESS" in result.message


def test_detect_objects_uses_one_saved_rgb_frame_for_all_queries():
    detector = Mock()
    detector.detect.side_effect = [
        {
            "status": "SUCCESS", "message": "black found",
            "detections": [{"label": "black cube", "confidence": 0.9, "metadata": {}}],
        },
        {
            "status": "SUCCESS", "message": "red found",
            "detections": [{"label": "red cube", "confidence": 0.8, "metadata": {}}],
        },
    ]
    backend = D435iBackend(config={"wait_timeout_sec": 0.0}, detector_adapter=detector)
    backend.latest_rgb_msg = object()
    backend._save_latest_rgb_frame = Mock(return_value={
        "status": "SUCCESS", "message": "saved", "path": "data/perception_runtime/latest_rgb.png",
    })

    result = backend.detect_objects(["black cube", "red cube"])

    backend._save_latest_rgb_frame.assert_called_once()
    assert detector.detect.call_count == 2
    assert result["status"] == "SUCCESS"
    assert [item["metadata"]["query"] for item in result["objects"]] == ["black cube", "red cube"]
    assert [item["pose_available"] for item in result["objects"]] == [False, False]


def test_detect_objects_uses_detector_multi_query_api_when_available():
    class MultiQueryDetector:
        def __init__(self):
            self.calls = []

        def detect_many(self, image_path, queries, min_confidence):
            self.calls.append((image_path, queries, min_confidence))
            return {
                "status": "SUCCESS",
                "message": "shared image encoding",
                "query_results": [
                    {
                        "query": "black cube",
                        "status": "SUCCESS",
                        "message": "found",
                        "detections": [
                            {"label": "black cube", "confidence": 0.9, "metadata": {}}
                        ],
                    },
                    {
                        "query": "red cube",
                        "status": "NOT_FOUND",
                        "message": "none",
                        "detections": [],
                    },
                ],
            }

    detector = MultiQueryDetector()
    backend = D435iBackend(config={"wait_timeout_sec": 0.0}, detector_adapter=detector)
    backend.latest_rgb_msg = object()
    backend._save_latest_rgb_frame = Mock(return_value={
        "status": "SUCCESS", "message": "saved", "path": "data/perception_runtime/latest_rgb.png",
    })

    result = backend.detect_objects(["black cube", "red cube"], min_confidence=0.6)

    assert detector.calls == [
        ("data/perception_runtime/latest_rgb.png", ["black cube", "red cube"], 0.6)
    ]
    assert result["status"] == "SUCCESS"
    assert result["query_results"][1]["status"] == "NOT_FOUND"


def test_detect_objects_applies_nut_threshold_without_raising_other_classes():
    class MultiQueryDetector:
        def detect_many(self, _image_path, _queries, _min_confidence):
            return {
                "status": "SUCCESS",
                "query_results": [
                    {
                        "query": "metal nuts", "status": "SUCCESS", "message": "found",
                        "detections": [
                            {"label": "metal nuts", "confidence": 0.63, "metadata": {}},
                            {"label": "metal nuts", "confidence": 0.93, "metadata": {}},
                        ],
                    },
                    {
                        "query": "screw", "status": "SUCCESS", "message": "found",
                        "detections": [
                            {"label": "screw", "confidence": 0.55, "metadata": {}},
                        ],
                    },
                ],
            }

    backend = D435iBackend(
        config={
            "perception": {"class_confidence_thresholds": {"nut": 0.70}},
            "d435i": {"wait_timeout_sec": 0.0},
        },
        detector_adapter=MultiQueryDetector(),
    )
    backend.latest_rgb_msg = object()
    backend._save_latest_rgb_frame = Mock(return_value={
        "status": "SUCCESS", "message": "saved",
        "path": "data/perception_runtime/latest_rgb.png",
    })

    result = backend.detect_objects(["metal nuts", "screw"], min_confidence=0.5)

    assert [item["confidence"] for item in result["objects"]] == [0.93, 0.55]
    assert result["query_results"][0]["min_confidence_applied"] == 0.70
    assert result["query_results"][0]["filtered_detection_count"] == 1
    assert result["query_results"][1]["min_confidence_applied"] == 0.5
    assert result["query_results"][1]["filtered_detection_count"] == 0


def test_detect_object_reports_not_found_when_class_threshold_filters_every_candidate():
    detector = Mock()
    detector.detect.return_value = {
        "status": "SUCCESS", "message": "detected",
        "detections": [{"label": "nut", "confidence": 0.63, "metadata": {}}],
    }
    backend = D435iBackend(
        config={
            "perception": {"class_confidence_thresholds": {"nut": 0.70}},
            "d435i": {"wait_timeout_sec": 0.0},
        },
        detector_adapter=detector,
    )
    backend.latest_rgb_msg = object()
    backend._save_latest_rgb_frame = Mock(return_value={
        "status": "SUCCESS", "message": "saved",
        "path": "data/perception_runtime/latest_rgb.png",
    })

    result = backend.detect_object("nut", min_confidence=0.5)

    assert result.status == "NOT_FOUND"
    assert result.objects == []
    assert "threshold 0.70" in result.message


def test_detect_objects_preserves_multi_query_worker_failure():
    class FailedMultiQueryDetector:
        def detect_many(self, image_path, queries, min_confidence):
            return {"status": "FAILED", "message": "worker stopped", "query_results": []}

    backend = D435iBackend(
        config={"wait_timeout_sec": 0.0}, detector_adapter=FailedMultiQueryDetector(),
    )
    backend.latest_rgb_msg = object()
    backend._save_latest_rgb_frame = Mock(return_value={
        "status": "SUCCESS", "message": "saved", "path": "data/perception_runtime/latest_rgb.png",
    })

    result = backend.detect_objects(["black cube"])

    assert result["status"] == "FAILED"


def test_detect_object_estimates_pose_when_depth_and_camera_info_available():
    detector = Mock()
    detector.detect.return_value = {
        "status": "SUCCESS",
        "message": "detected",
        "detections": [
            {
                "label": "red cube",
                "confidence": 0.85,
                "bbox_xyxy": [2, 2, 4, 4],
                "metadata": {},
            }
        ],
    }
    backend = D435iBackend(config={"wait_timeout_sec": 0.0}, detector_adapter=detector)
    backend.latest_rgb_msg = object()
    backend.latest_depth_msg = {
        "image": np.full((5, 5), 2.0, dtype=np.float32),
        "encoding": "32FC1",
    }
    backend.latest_camera_info_msg = Mock(
        k=[100.0, 0.0, 2.0, 0.0, 100.0, 2.0, 0.0, 0.0, 1.0]
    )
    backend._save_latest_rgb_frame = Mock(
        return_value={
            "status": "SUCCESS",
            "message": "saved",
            "path": "data/perception_runtime/latest_rgb.png",
        }
    )

    result = backend.detect_object("red cube")

    assert result.objects[0].pose_available is True
    assert result.objects[0].pose is not None
    assert result.objects[0].pose.x == 0.02
    assert result.objects[0].pose.y == 0.02
    assert result.objects[0].pose.z == 2.0


def test_detect_object_uses_rgb_associated_depth_snapshot_after_detector_delay():
    backend = None

    class Detector:
        def detect(self, *_args):
            # Simulate live depth callbacks continuing while SAM3 is running.
            backend.latest_depth_msg = {
                "image": np.full((5, 5), 3.0, dtype=np.float32),
                "encoding": "32FC1",
            }
            return {
                "status": "SUCCESS",
                "message": "detected",
                "detections": [{
                    "label": "red cube",
                    "confidence": 0.9,
                    "bbox_xyxy": [2, 2, 4, 4],
                    "metadata": {},
                }],
            }

    backend = D435iBackend(
        config={"wait_timeout_sec": 0.0}, detector_adapter=Detector(),
    )
    backend.latest_rgb_msg = object()
    backend.latest_depth_msg = {
        "image": np.full((5, 5), 2.0, dtype=np.float32),
        "encoding": "32FC1",
    }
    backend.latest_camera_info_msg = Mock(
        k=[100.0, 0.0, 2.0, 0.0, 100.0, 2.0, 0.0, 0.0, 1.0]
    )
    backend._store_rgbd_snapshot(
        "capture-1",
        depth_frame=backend._get_latest_depth_image(),
        camera_info=backend.latest_camera_info_msg,
    )
    backend._save_latest_rgb_frame = Mock(return_value={
        "status": "SUCCESS",
        "message": "saved",
        "path": "data/perception_runtime/latest_rgb.png",
        "capture_id": "capture-1",
    })

    result = backend.detect_object("red cube")

    assert result.objects[0].pose.z == 2.0
    assert result.objects[0].metadata["rgbd_capture_id"] == "capture-1"


def test_detect_object_transforms_pose_to_target_frame_when_tf_available():
    detector = Mock()
    detector.detect.return_value = {
        "status": "SUCCESS",
        "message": "detected",
        "detections": [
            {
                "label": "red cube",
                "confidence": 0.85,
                "bbox_xyxy": [2, 2, 4, 4],
                "metadata": {},
            }
        ],
    }
    pose_transformer = Mock()
    pose_transformer.transform_pose.side_effect = lambda pose: pose.model_copy(
        update={"frame": "base_link", "x": pose.x + 1.0}
    )
    backend = D435iBackend(
        config={
            "perception": {"target_frame": "base_link"},
            "d435i": {"wait_timeout_sec": 0.0},
        },
        detector_adapter=detector,
        pose_transformer=pose_transformer,
    )
    backend.latest_rgb_msg = object()
    backend.latest_depth_msg = {
        "image": np.full((5, 5), 2.0, dtype=np.float32),
        "encoding": "32FC1",
    }
    backend.latest_camera_info_msg = Mock(
        k=[100.0, 0.0, 2.0, 0.0, 100.0, 2.0, 0.0, 0.0, 1.0]
    )
    backend._save_latest_rgb_frame = Mock(
        return_value={
            "status": "SUCCESS",
            "message": "saved",
            "path": "data/perception_runtime/latest_rgb.png",
        }
    )

    result = backend.detect_object("red cube")

    assert result.objects[0].pose_available is True
    assert result.objects[0].pose.frame == "base_link"
    assert result.objects[0].pose.x == 1.02
    assert result.objects[0].metadata.get("warnings") is None
    assert result.scene.frame == "base_link"


def test_detect_object_keeps_camera_pose_and_warns_when_tf_unavailable():
    detector = Mock()
    detector.detect.return_value = {
        "status": "SUCCESS",
        "message": "detected",
        "detections": [
            {
                "label": "red cube",
                "confidence": 0.85,
                "bbox_xyxy": [2, 2, 4, 4],
                "metadata": {},
            }
        ],
    }
    pose_transformer = Mock()
    pose_transformer.transform_pose.return_value = None
    backend = D435iBackend(
        config={
            "perception": {"target_frame": "base_link"},
            "d435i": {"wait_timeout_sec": 0.0},
        },
        detector_adapter=detector,
        pose_transformer=pose_transformer,
    )
    backend.latest_rgb_msg = object()
    backend.latest_depth_msg = {
        "image": np.full((5, 5), 2.0, dtype=np.float32),
        "encoding": "32FC1",
    }
    backend.latest_camera_info_msg = Mock(
        k=[100.0, 0.0, 2.0, 0.0, 100.0, 2.0, 0.0, 0.0, 1.0]
    )
    backend._save_latest_rgb_frame = Mock(
        return_value={
            "status": "SUCCESS",
            "message": "saved",
            "path": "data/perception_runtime/latest_rgb.png",
        }
    )

    result = backend.detect_object("red cube")

    assert result.objects[0].pose_available is True
    assert result.objects[0].pose.frame == "camera_color_optical_frame"
    assert result.objects[0].metadata["warnings"] == [
        "TF transform to base_link unavailable."
    ]


def test_detect_object_uses_label_based_object_id():
    detector = Mock()
    detector.detect.return_value = {
        "status": "SUCCESS",
        "message": "detected",
        "detections": [
            {
                "label": "red cubes",
                "confidence": 0.95,
                "bbox_xyxy": [1, 2, 3, 4],
                "metadata": {},
            }
        ],
    }
    backend = D435iBackend(config={"wait_timeout_sec": 0.0}, detector_adapter=detector)
    backend.latest_rgb_msg = object()
    backend._save_latest_rgb_frame = Mock(
        return_value={
            "status": "SUCCESS",
            "message": "saved",
            "path": "data/perception_runtime/latest_rgb.png",
        }
    )

    result = backend.detect_object("red cubes")

    assert result.objects[0].object_id == "red cubes_1"
