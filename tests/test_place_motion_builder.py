import pytest

from llm_skill_robot.perception.place_motion_builder import PlaceMotionBuilder, PlaceMotionError


def test_place_motion_preserves_reviewed_grasp_offset_and_orientation():
    result = PlaceMotionBuilder(approach_height_m=0.05).build(
        source_pose={"frame": "base_link", "x": 0.4, "y": 0.1, "z": 0.05},
        target_pose={"frame": "base_link", "x": 0.6, "y": -0.2, "z": 0.05},
        grasp_pose={
            "frame": "base_link",
            "position": {"x": 0.41, "y": 0.12, "z": 0.10},
            "orientation": {"x": 0.0, "y": 1.0, "z": 0.0, "w": 0.0},
        },
    )

    assert result["place_contact_pose"]["position"] == pytest.approx({"x": 0.61, "y": -0.18, "z": 0.10})
    assert result["place_approach_pose"]["position"]["z"] == pytest.approx(0.15)
    assert result["place_contact_pose"]["orientation"]["y"] == 1.0


def test_place_motion_rejects_an_implausibly_large_grasp_offset():
    with pytest.raises(PlaceMotionError, match="offset"):
        PlaceMotionBuilder(max_grasp_offset_m=0.10).build(
            source_pose={"frame": "base_link", "x": 0.0, "y": 0.0, "z": 0.0},
            target_pose={"frame": "base_link", "x": 0.2, "y": 0.0, "z": 0.0},
            grasp_pose={
                "frame": "base_link",
                "position": {"x": 0.2, "y": 0.0, "z": 0.0},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            },
        )


def test_place_motion_accepts_a_reviewed_tool_offset_within_configured_envelope():
    result = PlaceMotionBuilder(max_grasp_offset_m=0.30).build(
        source_pose={"frame": "base_link", "x": 0.4, "y": 0.0, "z": 0.05},
        target_pose={"frame": "base_link", "x": 0.6, "y": 0.0, "z": 0.05},
        grasp_pose={
            "frame": "base_link",
            "position": {"x": 0.629, "y": 0.0, "z": 0.05},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        },
    )

    assert result["source_to_gripper_offset_norm_m"] == pytest.approx(0.229)
    assert result["place_contact_pose"]["position"]["x"] == pytest.approx(0.829)
