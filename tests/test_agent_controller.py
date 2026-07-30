import json

import pytest

from llm_skill_robot.agent.agent_controller import (
    AgentController,
    AgentDecisionError,
)
from llm_skill_robot.agent.agent_state import AgentState, AgentStatus


class FakeLLMClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.messages = []

    def generate_text(self, messages):
        self.messages.append(messages)
        return self.responses.pop(0)


def test_agent_proposes_one_valid_observation_tool_and_records_feedback():
    client = FakeLLMClient([
        json.dumps({
            "kind": "tool_call",
            "message": "Observe the requested object.",
            "tool_call": {"tool_name": "detect_object", "arguments": {"query": "red cube"}},
        }),
        json.dumps({"kind": "complete", "message": "The object was found."}),
    ])
    controller = AgentController(client)
    state = AgentState(user_goal="find the red cube", safety_mode="rviz_sim")

    proposal = controller.propose_next_action(state)

    assert proposal.plan_step.skill_id == "detect_object"
    assert proposal.plan_step.step_id == "1"
    assert proposal.requires_human_gate is False
    assert state.proposed_calls[-1].tool_name == "detect_object"
    assert "tool_catalog" in client.messages[0][1]["content"]

    result = controller.record_tool_result(
        state,
        {"success": True, "message": "Found cube.", "output": {"object_id": "cube_1"}},
    )
    assert result.output["object_id"] == "cube_1"
    assert state.context["last_tool_result"]["tool_name"] == "detect_object"

    complete = controller.propose_next_action(state)
    assert complete.plan_step is None
    assert state.status == AgentStatus.COMPLETED


def test_high_risk_tool_is_marked_for_external_human_gate():
    client = FakeLLMClient([json.dumps({
        "kind": "tool_call",
        "message": "Open the gripper.",
        "tool_call": {"tool_name": "open_gripper", "arguments": {}},
    })])
    proposal = AgentController(client).propose_next_action(AgentState(user_goal="open gripper"))

    assert proposal.plan_step.skill_id == "open_gripper"
    assert proposal.requires_human_gate is True
    assert proposal.safety_tier.value == "human_confirm"


def test_agent_can_start_a_composite_safe_pick_without_exposing_its_stages_to_llm():
    client = FakeLLMClient([json.dumps({
        "kind": "composite_skill_call",
        "message": "Use the supervised grasp workflow.",
        "composite_skill_call": {
            "skill_name": "safe_pick_object",
            "arguments": {"object_query": "red cube"},
        },
    })])
    state = AgentState(user_goal="pick the red cube")

    proposal = AgentController(client).propose_next_action(state)

    assert proposal.plan_step is None
    assert proposal.composite_skill.skill_name == "safe_pick_object"
    assert proposal.requires_human_gate is True
    assert proposal.composite_skill.next_tool_call().tool_name == "move_to_named_target"
    assert state.context["active_composite_skill"]["arguments"] == {
        "object_query": "red cube"
    }
    prompt = client.messages[0][1]["content"]
    assert "composite_skill_catalog" in prompt
    assert "safe_pick_object" in prompt


def test_agent_normalizes_common_flattened_composite_llm_response_and_prompts_nesting():
    client = FakeLLMClient([json.dumps({
        "kind": "composite_skill_call",
        "message": "Use the reviewed grasp workflow.",
        "skill_name": "safe_pick_object",
        "arguments": {"object_query": "red cube"},
    })])
    proposal = AgentController(client).propose_next_action(
        AgentState(user_goal="pick the red cube")
    )

    assert proposal.composite_skill.skill_name == "safe_pick_object"
    prompt = client.messages[0][1]["content"]
    assert "required_output_examples" in prompt
    assert "composite_skill_call" in prompt


def test_agent_rejects_unknown_or_invalid_tool_calls_without_recording_them():
    client = FakeLLMClient([json.dumps({
        "kind": "tool_call",
        "message": "Bypass safety.",
        "tool_call": {"tool_name": "send_ros_command", "arguments": {}},
    })])
    state = AgentState(user_goal="do something")

    with pytest.raises(AgentDecisionError, match="unknown Agent tool"):
        AgentController(client).propose_next_action(state)
    assert state.proposed_calls == []


def test_agent_can_pause_for_human_and_be_resumed():
    client = FakeLLMClient([json.dumps({
        "kind": "request_human",
        "message": "Which blue object should be selected?",
    })])
    state = AgentState(user_goal="pick the blue object")
    proposal = AgentController(client).propose_next_action(state)

    assert proposal.requires_human_gate is True
    assert state.status == AgentStatus.WAITING_FOR_HUMAN
    state.resume_after_human_input("Select the blue cup.")
    assert state.status == AgentStatus.ACTIVE


def test_invalid_decision_json_is_rejected():
    with pytest.raises(AgentDecisionError, match="valid AgentDecision JSON"):
        AgentController(FakeLLMClient(["not json"])).propose_next_action(
            AgentState(user_goal="inspect scene")
        )
