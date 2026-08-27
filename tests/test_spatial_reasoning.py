import pytest

from llm_skill_robot.perception.scene_model import ObjectPose, SceneModel, SceneObject
from llm_skill_robot.perception.spatial_reasoning import (
    RelativePlacePosePlanner,
    SpatialReasoningError,
)


def _object(object_id, x, y, z, *, dimensions=(0.04, 0.04, 0.04)):
    return SceneObject(
        object_id=object_id,
        label=object_id.replace("_", " "),
        confidence=0.9,
        pose_available=True,
        pose=ObjectPose(frame="base_link", x=x, y=y, z=z),
        metadata={"dimensions_m": list(dimensions)},
    )


def _scene():
    return SceneModel(objects=[
        _object("black_cube_1", 0.10, 0.10, 0.02),
        _object("red_cube_1", 0.30, 0.10, 0.02),
        _object("white_cube_1", 0.50, 0.10, 0.02),
    ])


def test_between_uses_reference_midpoint_and_source_table_height():
    result = RelativePlacePosePlanner().compute(
        _scene(),
        source_id="black_cube_1",
        relation="between",
        reference_ids=["red_cube_1", "white_cube_1"],
    )

    pose = result["target_pose"]
    assert pose["frame"] == "base_link"
    assert pose["x"] == pytest.approx(0.40)
    assert pose["y"] == pytest.approx(0.10)
    assert pose["z"] == pytest.approx(0.023)
    assert result["requires_confirmation"] is True
    assert result["source_pose"]["x"] == pytest.approx(0.10)
    assert len(result["reference_poses"]) == 2


def test_right_of_uses_object_extents_and_clearance():
    result = RelativePlacePosePlanner().compute(
        _scene(),
        source_id="white_cube_1",
        relation="right_of",
        reference_ids=["red_cube_1"],
    )

    assert result["target_pose"]["x"] == pytest.approx(0.37)
    assert result["target_pose"]["y"] == pytest.approx(0.10)


def test_on_top_of_uses_both_object_heights():
    scene = SceneModel(objects=[
        _object("black_cube_1", 0.10, 0.10, 0.02, dimensions=(0.04, 0.04, 0.06)),
        _object("white_cube_1", 0.30, 0.10, 0.02, dimensions=(0.04, 0.04, 0.04)),
    ])

    result = RelativePlacePosePlanner().compute(
        scene,
        source_id="black_cube_1",
        relation="on_top_of",
        reference_ids=["white_cube_1"],
    )

    assert result["target_pose"]["x"] == pytest.approx(0.30)
    assert result["target_pose"]["z"] == pytest.approx(0.073)


def test_between_requires_two_reference_objects():
    with pytest.raises(SpatialReasoningError, match="exactly 2"):
        RelativePlacePosePlanner().compute(
            _scene(),
            source_id="black_cube_1",
            relation="between",
            reference_ids=["red_cube_1"],
        )
