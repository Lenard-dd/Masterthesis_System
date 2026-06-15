from unittest.mock import Mock

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
    assert result.objects[0].label == "red cube"
    assert result.objects[0].source == "d435i_sam3"
    assert result.objects[0].mask_available is True
    assert result.objects[0].pose_available is False
    assert result.objects[0].metadata["bbox_xyxy"] == [1, 2, 3, 4]
    assert "detector status SUCCESS" in result.message


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
