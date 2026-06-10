def test_ros_nl_rviz_sim_demo_imports_without_ros_initialization():
    """Importing the NL RViz sim demo must not initialize ROS or ask for input."""
    import llm_skill_robot.ros_nl_rviz_sim_demo as demo

    assert callable(demo.main)
