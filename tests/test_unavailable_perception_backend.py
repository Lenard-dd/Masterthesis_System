import sys

from llm_skill_robot.perception.d435i_backend import D435iBackend
from llm_skill_robot.perception.object_pose_estimator import ObjectPoseEstimator
from llm_skill_robot.perception.perception_backend_base import BasePerceptionBackend
from llm_skill_robot.perception.sam3_adapter import SAM3Adapter
from llm_skill_robot.perception.scene_model import SceneModel
from llm_skill_robot.perception.unavailable_perception_backend import (
    NOT_AVAILABLE_MESSAGE,
    UnavailablePerceptionBackend,
)


def test_detect_object_returns_not_available():
    backend = UnavailablePerceptionBackend()

    result = backend.detect_object("cup")

    assert isinstance(backend, BasePerceptionBackend)
    assert result.query == "cup"
    assert result.status == "NOT_AVAILABLE"
    assert result.objects == []
    assert result.message == NOT_AVAILABLE_MESSAGE
    assert isinstance(result.scene, SceneModel)
    assert result.scene.objects == []


def test_get_scene_returns_empty_scene_model():
    backend = UnavailablePerceptionBackend()

    scene = backend.get_scene()

    assert isinstance(scene, SceneModel)
    assert scene.objects == []
    assert scene.source == "unavailable_perception_backend"


def test_frame_and_camera_methods_return_not_available():
    backend = UnavailablePerceptionBackend()

    assert backend.get_rgb_frame()["status"] == "NOT_AVAILABLE"
    assert backend.get_depth_frame()["status"] == "NOT_AVAILABLE"
    assert backend.get_camera_info()["status"] == "NOT_AVAILABLE"


def test_future_backend_modules_import_without_real_hardware_dependencies():
    assert D435iBackend is not None
    assert SAM3Adapter is not None
    assert ObjectPoseEstimator is not None
    assert "pyrealsense2" not in sys.modules
    assert "sam3" not in sys.modules
