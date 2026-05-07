def test_ros_moveit_plan_only_demo_imports_without_ros_initialization():
    """Importing the ROS demo module must not initialize rclpy or MoveIt."""
    import llm_skill_robot.ros_moveit_plan_only_demo as demo

    assert callable(demo.main)
