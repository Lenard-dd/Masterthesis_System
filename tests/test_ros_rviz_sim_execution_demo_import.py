def test_ros_rviz_sim_execution_demo_imports_without_ros_initialization():
    """Importing the demo module must not initialize rclpy or ask for input."""
    import llm_skill_robot.ros_rviz_sim_execution_demo as demo

    assert callable(demo.main)
