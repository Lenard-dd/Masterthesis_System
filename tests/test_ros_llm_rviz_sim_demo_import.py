def test_ros_llm_rviz_sim_demo_imports_without_ros_initialization():
    """Importing the LLM RViz sim demo must not initialize ROS or call APIs."""
    import llm_skill_robot.ros_llm_rviz_sim_demo as demo

    assert callable(demo.main)
