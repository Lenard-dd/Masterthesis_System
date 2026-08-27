from llm_skill_robot.perception.relative_place_visualizer import RelativePlaceVisualizer


def test_preview_requires_ros_node_without_importing_ros_messages():
    result = RelativePlaceVisualizer().publish(
        {
            "source_pose": {"frame": "base_link", "x": 0.1, "y": 0.0, "z": 0.02},
            "reference_poses": [
                {"frame": "base_link", "x": 0.3, "y": 0.0, "z": 0.02},
            ],
            "target_pose": {"frame": "base_link", "x": 0.37, "y": 0.0, "z": 0.02},
        }
    )

    assert result["success"] is False
    assert "ROS node is required" in result["message"]
    assert result["topic"] == "/relative_place_preview"
