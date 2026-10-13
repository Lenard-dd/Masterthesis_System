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


def test_side_placement_flat_fallback_preserves_source_height_not_reference_bottom():
    scene = SceneModel(objects=[
        _object("nut_1", 0.40, -0.10, 0.012, dimensions=(0.02, 0.02, 0.01)),
        # Deliberately inconsistent container bottom; it must not drag the
        # tabletop placement height below the picked object's support level.
        _object("box_1", 0.50, -0.10, 0.01, dimensions=(0.12, 0.10, 0.10)),
    ])

    result = RelativePlacePosePlanner().compute(
        scene, source_id="nut_1", relation="left_of", reference_ids=["box_1"],
    )

    assert result["target_pose"]["z"] == pytest.approx(0.015)
    assert result["assumptions"]["support_height_strategy"] == "source_height_flat_table_fallback"


def test_side_placement_uses_calibrated_tilted_plane_at_target_xy():
    scene = SceneModel(objects=[
        _object("nut_1", 0.0, 0.0, 0.01, dimensions=(0.02, 0.02, 0.02)),
        _object("box_1", 0.30, 0.0, 0.05, dimensions=(0.10, 0.10, 0.10)),
    ])
    planner = RelativePlacePosePlanner(support_plane={
        "frame": "base_link", "normal": {"x": -0.10, "y": 0.0, "z": 1.0}, "d": 0.0,
    })

    result = planner.compute(
        scene, source_id="nut_1", relation="right_of", reference_ids=["box_1"],
    )

    target = result["target_pose"]
    normal_length = (1.0 + 0.10 ** 2) ** 0.5
    expected_plane_z = -(
        (-0.10 / normal_length) * target["x"]
    ) / (1.0 / normal_length)
    assert target["z"] == pytest.approx(expected_plane_z + 0.01 + 0.003)
    assert result["assumptions"]["support_height_strategy"] == "calibrated_plane"


def test_calibrated_plane_rejects_inconsistent_source_support_height():
    scene = SceneModel(objects=[
        _object("nut_1", 0.0, 0.0, 0.10, dimensions=(0.02, 0.02, 0.02)),
        _object("box_1", 0.30, 0.0, 0.05, dimensions=(0.10, 0.10, 0.10)),
    ])
    planner = RelativePlacePosePlanner(
        support_plane={
            "frame": "base_link", "normal": {"x": 0.0, "y": 0.0, "z": 1.0}, "d": 0.0,
        },
        max_source_support_error_m=0.02,
    )

    with pytest.raises(SpatialReasoningError, match="inconsistent"):
        planner.compute(
            scene, source_id="nut_1", relation="right_of", reference_ids=["box_1"],
        )


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
    assert result["placement_mode"] == "supported_surface"
    assert result["placement_safety"]["source_support_z_m"] == pytest.approx(-0.01)
    assert result["placement_safety"]["minimum_object_bottom_z_m"] == pytest.approx(0.043)


def test_on_top_of_uses_calibrated_source_support_for_release_height_invariant():
    scene = SceneModel(objects=[
        _object("apple_1", 0.40, 0.0, 0.065, dimensions=(0.08, 0.08, 0.10)),
        _object("paper_1", 0.60, 0.0, 0.002, dimensions=(0.10, 0.10, 0.002)),
    ])
    result = RelativePlacePosePlanner(support_plane={
        "frame": "base_link", "normal": {"x": 0.0, "y": 0.0, "z": 1.0}, "d": -0.004,
    }).compute(
        scene, source_id="apple_1", relation="on_top_of", reference_ids=["paper_1"],
    )

    assert result["placement_safety"]["source_support_z_m"] == pytest.approx(0.004)
    assert result["placement_safety"]["minimum_object_bottom_z_m"] == pytest.approx(0.006)
    assert result["placement_safety"]["support_height_strategy"] == "calibrated_plane"


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
        SpatialConstraintCompiler().compile("teleport_near", ["red_box_1"])


def test_inside_uses_safe_above_opening_release_without_gripper_insertion():
    scene = SceneModel(objects=[
        _object("nut_1", 0.10, 0.10, 0.01, dimensions=(0.02, 0.02, 0.01)),
        SceneObject(
            object_id="red_box_1", label="red box", confidence=0.9, pose_available=True,
            pose=ObjectPose(frame="base_link", x=0.40, y=0.20, z=0.04),
            metadata={
                "dimensions_m": [0.12, 0.10, 0.08],
                "container_opening_dimensions_m": {"x": 0.10, "y": 0.08},
            },
        ),
    ])

    result = RelativePlacePosePlanner().compute(
        scene, source_id="nut_1", relation="inside", reference_ids=["red_box_1"],
    )

    assert result["placement_mode"] == "above_opening_drop"
    assert result["inside_safety"]["gripper_enters_container"] is False
    assert result["inside_safety"]["opening_dimensions_measured"] is True
    assert result["target_pose"]["x"] == pytest.approx(0.40)
    assert result["target_pose"]["y"] == pytest.approx(0.20)
    assert result["target_pose"]["z"] == pytest.approx(0.125)


def test_inside_rejects_object_that_does_not_fit_opening_with_clearance():
    scene = SceneModel(objects=[
        _object("large_object_1", 0.10, 0.10, 0.02, dimensions=(0.08, 0.07, 0.04)),
        SceneObject(
            object_id="small_box_1", label="small box", confidence=0.9, pose_available=True,
            pose=ObjectPose(frame="base_link", x=0.40, y=0.20, z=0.03),
            metadata={
                "dimensions_m": [0.08, 0.07, 0.06],
                "container_opening_dimensions_m": {"x": 0.075, "y": 0.065},
            },
        ),
    ])

    with pytest.raises(SpatialReasoningError, match="does not fit"):
        RelativePlacePosePlanner().compute(
            scene, source_id="large_object_1", relation="inside",
            reference_ids=["small_box_1"],
        )
