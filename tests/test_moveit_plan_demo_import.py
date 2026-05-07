def test_moveit_plan_demo_imports_without_user_input():
    """Importing the demo module must not start interactive validation."""
    import llm_skill_robot.moveit_plan_demo as moveit_plan_demo

    assert callable(moveit_plan_demo.main)
