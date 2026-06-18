from types import SimpleNamespace

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
