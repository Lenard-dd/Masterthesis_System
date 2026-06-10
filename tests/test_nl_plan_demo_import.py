def test_nl_plan_demo_imports_without_user_input():
    """Importing the NL plan demo must not ask for interactive input."""
    import llm_skill_robot.nl_plan_demo as demo

    assert callable(demo.main)
