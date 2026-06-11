import pytest
from pydantic import ValidationError

from llm_skill_robot.perception.scene_model import (
    DetectionResult,
    ObjectPose,
    SceneModel,
    SceneObject,
)


def test_scene_object_with_pose_validates_correctly():
    pose = ObjectPose(frame="base_link", x=0.1, y=0.2, z=0.3)

    scene_object = SceneObject(
        object_id="object_1",
        label="cup",
        confidence=0.92,
        mask_available=True,
        pose_available=True,
        pose=pose,
        graspable=True,
        source="d435i_sam3_future",
        metadata={"color": "red"},
    )

    assert scene_object.pose == pose
    assert scene_object.pose.qw == 1.0
    assert scene_object.graspable is True


def test_scene_model_find_by_label_works():
    cup = SceneObject(object_id="cup_1", label="cup", confidence=0.9)
    box = SceneObject(object_id="box_1", label="box", confidence=0.8)
    scene = SceneModel(objects=[cup, box])

    assert scene.find_by_label("cup") == [cup]
    assert scene.find_by_label("missing") == []
    assert scene.has_object("cup") is True


def test_scene_model_find_by_id_works():
    cup = SceneObject(object_id="cup_1", label="cup", confidence=0.9)
    scene = SceneModel(objects=[cup])

    assert scene.find_by_id("cup_1") == cup
    assert scene.find_by_id("missing") is None
    assert scene.has_object("cup_1") is True


def test_detection_result_not_available_works():
    result = DetectionResult(
        query="cup",
        status="NOT_AVAILABLE",
        message="Perception pipeline is not connected.",
    )

    assert result.status == "NOT_AVAILABLE"
    assert result.objects == []
    assert result.scene is None


@pytest.mark.parametrize("confidence", [-0.01, 1.01])
def test_scene_object_confidence_outside_zero_to_one_is_rejected(confidence):
    with pytest.raises(ValidationError):
        SceneObject(object_id="bad_confidence", label="cup", confidence=confidence)
