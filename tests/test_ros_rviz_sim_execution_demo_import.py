def test_ros_rviz_sim_execution_demo_imports_without_ros_initialization():
    """Importing the demo module must not initialize rclpy or ask for input."""
    import llm_skill_robot.ros_rviz_sim_execution_demo as demo

    assert callable(demo.main)


def test_ros_rviz_sim_demo_hardcoded_plan_is_stable():
    import llm_skill_robot.ros_rviz_sim_execution_demo as demo

    plan = demo._create_plan()

    assert demo.RVIZ_SIM_MODE_WARNING == (
        "RVIZ_SIM MODE ONLY. This demo is not intended for real robot execution."
    )
    assert len(plan.steps) == 1
    step = plan.steps[0]
    assert step.skill_id == "move_to_named_target"
    assert step.parameters["target"] == "home"
    assert step.parameters["velocity_scale"] == 0.03
    assert step.parameters["acceleration_scale"] == 0.03
