def test_perception_demo_imports_without_hardware_dependencies():
    from llm_skill_robot import perception_demo

    assert hasattr(perception_demo, "main")
