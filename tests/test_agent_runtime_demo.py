from llm_skill_robot.agent.agent_state import AgentState, AgentStatus
from llm_skill_robot.agent_runtime_demo import (
    ScriptedDemoLLMClient,
    create_mock_session,
)


def test_scripted_mock_demo_runs_the_composite_skill_without_hardware():
    state = AgentState(user_goal="pick the red cube", safety_mode="mock")
    result = create_mock_session(
        ScriptedDemoLLMClient(), auto_approve=True
    ).run(state)

    assert result.status == AgentStatus.COMPLETED
    assert result.composite_results[0].status.value == "completed"
    assert state.selected_object_id == "mock_cube_1"
    assert state.selected_grasp_candidate_id == "mock_grasp_1"
    assert any(item.tool_name == "verify_grasp" for item in state.tool_results)


def test_demo_module_imports_without_prompting_or_calling_an_api():
    import llm_skill_robot.agent_runtime_demo as demo

    assert callable(demo.main)
