def test_llm_plan_demo_imports_without_user_input():
    """Importing the LLM plan demo must not ask for input or call APIs."""
    import llm_skill_robot.llm_plan_demo as demo

    assert callable(demo.main)
