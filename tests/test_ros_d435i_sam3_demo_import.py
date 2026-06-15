def test_ros_d435i_sam3_demo_imports_without_ros_initialization():
    """Importing the D435i/SAM3 demo must not initialize ROS or ask for input."""
    import llm_skill_robot.ros_d435i_sam3_demo as demo

    assert callable(demo.main)
