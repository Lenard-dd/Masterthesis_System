from types import SimpleNamespace

import pytest

from llm_skill_robot.perception.scene_model import ObjectPose
from llm_skill_robot.perception.tf_pose_transformer import TFPoseTransformer


def test_tf_pose_transformer_imports_without_ros_dependencies():
    assert TFPoseTransformer is not None


def test_apply_transform_translates_pose_position():
    transformer = TFPoseTransformer.__new__(TFPoseTransformer)
    transformer.target_frame = "base_link"
    transform = SimpleNamespace(
        transform=SimpleNamespace(
            translation=SimpleNamespace(x=1.0, y=2.0, z=3.0),
            rotation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
        )
    )
    pose = ObjectPose(frame="camera_color_optical_frame", x=0.1, y=0.2, z=0.3)

    transformed = transformer._apply_transform(pose, transform)

    assert transformed.frame == "base_link"
    assert transformed.x == 1.1
    assert transformed.y == 2.2
    assert transformed.z == 3.3
    assert transformed.qw == 1.0


def test_apply_transform_to_pose_dict_preserves_full_pose_orientation():
    transformer = TFPoseTransformer.__new__(TFPoseTransformer)
    transformer.target_frame = "base_link"
    transform = SimpleNamespace(
        transform=SimpleNamespace(
            translation=SimpleNamespace(x=1.0, y=2.0, z=3.0),
            rotation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
        )
    )
    pose = {
        "frame": "camera_color_optical_frame",
        "position": {"x": 0.1, "y": 0.2, "z": 0.3},
        "orientation": {"x": 0.0, "y": 0.0, "z": 0.70710678, "w": 0.70710678},
    }

    transformed = transformer._apply_transform_to_pose_dict(pose, transform)

    assert transformed["frame"] == "base_link"
    assert transformed["position"] == {"x": 1.1, "y": 2.2, "z": 3.3}
    assert transformed["orientation"]["z"] == pytest.approx(0.70710678)


def test_lookup_transform_spins_until_tf_buffer_is_populated():
    transform = SimpleNamespace(
        transform=SimpleNamespace(
            translation=SimpleNamespace(x=0.0, y=0.0, z=0.0),
            rotation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
        )
    )

    class FakeBuffer:
        def __init__(self):
            self.calls = 0

        def lookup_transform(self, target_frame, source_frame, time_value):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("frame not ready")
            return transform

    class FakeRclpy:
        class time:
            class Time:
                pass

        def __init__(self):
            self.spin_calls = 0

        def spin_once(self, node, timeout_sec=0.0):
            self.spin_calls += 1

    fake_rclpy = FakeRclpy()
    transformer = TFPoseTransformer.__new__(TFPoseTransformer)
    transformer.node = object()
    transformer.target_frame = "base_link"
    transformer._buffer = FakeBuffer()
    transformer._rclpy = fake_rclpy

    result = transformer._lookup_transform("camera_color_optical_frame", 0.2)

    assert result is transform
    assert fake_rclpy.spin_calls == 1


def test_lookup_transform_does_not_spin_node_owned_by_external_executor():
    transform = SimpleNamespace(
        transform=SimpleNamespace(
            translation=SimpleNamespace(x=0.0, y=0.0, z=0.0),
            rotation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
        )
    )

    class FakeBuffer:
        def __init__(self):
            self.calls = 0

        def lookup_transform(self, *_args):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("frame not ready")
            return transform

    class FakeRclpy:
        class time:
            class Time:
                pass

        @staticmethod
        def spin_once(*_args, **_kwargs):
            raise AssertionError("shared ROS node must not be spun by TF lookup")

    transformer = TFPoseTransformer.__new__(TFPoseTransformer)
    transformer.node = SimpleNamespace(executor=object())
    transformer.target_frame = "base_link"
    transformer._buffer = FakeBuffer()
    transformer._rclpy = FakeRclpy()

    assert transformer._lookup_transform("camera_color_optical_frame", 0.2) is transform
