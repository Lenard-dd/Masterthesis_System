import pytest

from llm_skill_robot.perception.scene_model import ObjectPose, SceneModel, SceneObject
from llm_skill_robot.perception.spatial_reasoning import (
    RelativePlacePosePlanner,
    SpatialReasoningError,
)
from llm_skill_robot.perception.spatial_constraints import (
    AxisAssignmentSolver,
    ConstraintConflictError,
    SpatialConstraintCompiler,
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


def test_right_of_uses_measured_point_cloud_footprints_with_conservative_height():
    scene = SceneModel(objects=[
        SceneObject(
            object_id="black_cube_1", label="black cube", confidence=0.9,
            pose_available=True,
            pose=ObjectPose(frame="base_link", x=0.10, y=0.10, z=0.02),
            metadata={"footprint_dimensions_m": {"x": 0.10, "y": 0.06}},
        ),
        SceneObject(
            object_id="white_cube_1", label="white cube", confidence=0.9,
            pose_available=True,
            pose=ObjectPose(frame="base_link", x=0.30, y=0.10, z=0.02),
            metadata={"footprint_dimensions_m": {"x": 0.20, "y": 0.08}},
        ),
    ])

    result = RelativePlacePosePlanner().compute(
        scene,
        source_id="black_cube_1",
        relation="right_of",
        reference_ids=["white_cube_1"],
    )

    assert result["target_pose"]["x"] == pytest.approx(0.48)
    assert result["assumptions"]["object_dimensions_estimated"] is True


def test_between_requires_two_reference_objects():
    with pytest.raises(SpatialReasoningError, match="exactly 2"):
        RelativePlacePosePlanner().compute(
            _scene(),
            source_id="black_cube_1",
            relation="between",
            reference_ids=["red_cube_1"],
        )


def test_relation_is_exposed_as_auditable_base_constraints():
    result = RelativePlacePosePlanner().compute(
        _scene(), source_id="white_cube_1", relation="right_of",
        reference_ids=["red_cube_1"],
    )

    constraints = result["geometry_constraints"]
    assert constraints["relation"] == "right_of"
    assert [item["kind"] for item in constraints["constraints"]] == [
        "axis_separation", "table_support",
    ]
    assert constraints["constraints"][0]["axes"] == ["x"]


def test_composite_solver_combines_side_and_alignment_constraints_into_one_pose():
    result = RelativePlacePosePlanner().compute_composite(
        _scene(), source_id="black_cube_1",
        relation_specs=[
            {"relation": "right_of", "reference_ids": ["red_cube_1"]},
            {"relation": "aligned_y_with", "reference_ids": ["white_cube_1"]},
        ],
    )

    assert result["relation"] == "composite"
    assert result["target_pose"]["x"] == pytest.approx(0.37)
    assert result["target_pose"]["y"] == pytest.approx(0.10)
    assert len(result["geometry_constraints"]) == 2


def test_constraint_solver_rejects_conflicting_axis_assignments():
    with pytest.raises(ConstraintConflictError, match="conflict on x"):
        AxisAssignmentSolver(tolerance_m=0.001).merge(
            {"x": 0.0, "y": 0.0, "z": 0.0},
            [{"x": 0.1}, {"x": 0.2}],
        )


def test_constraint_compiler_rejects_unknown_relation():
    with pytest.raises(ValueError, match="Unsupported constraint relation"):
        SpatialConstraintCompiler().compile("inside", ["red_box_1"])
