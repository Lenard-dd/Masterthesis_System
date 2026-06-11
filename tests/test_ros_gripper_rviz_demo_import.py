def test_ros_gripper_rviz_demo_imports_without_ros_initialization():
    import llm_skill_robot.ros_gripper_rviz_demo as module

    assert callable(module.main)
