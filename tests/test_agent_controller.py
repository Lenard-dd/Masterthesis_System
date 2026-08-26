import json

import pytest

from llm_skill_robot.agent.agent_controller import (
    ApprovalStage,
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
    assert proposal.approval_stages == []
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


def test_multilingual_gripper_aliases_route_to_the_specific_safe_action():
    controller = AgentController(FakeLLMClient([]))

    assert controller._allowed_actions_for_goal(
        AgentState(user_goal="闭合夹具")
    ) == {"close_gripper"}
    assert controller._allowed_actions_for_goal(
        AgentState(user_goal="张开夹爪")
    ) == {"open_gripper"}
    assert controller._allowed_actions_for_goal(
        AgentState(user_goal="グリッパーを閉じる")
    ) == {"close_gripper"}


def test_scene_description_goal_routes_only_to_read_only_scene_tool():
    controller = AgentController(FakeLLMClient([]))

    assert controller._allowed_actions_for_goal(
        AgentState(user_goal="扫描场景并描述桌面上有什么")
    ) == {"describe_scene"}


def test_scene_description_proposal_skips_human_approval_gate():
    client = FakeLLMClient([json.dumps({
        "kind": "tool_call",
        "message": "Describe the scene.",
        "tool_call": {"tool_name": "describe_scene", "arguments": {}},
    })])

    proposal = AgentController(client).propose_next_action(
        AgentState(user_goal="describe the scene")
    )

    assert proposal.plan_step.skill_id == "describe_scene"
    assert proposal.requires_human_gate is False
    assert proposal.approval_stages == []


def test_localization_goal_allows_single_and_multi_object_localization_tools():
    controller = AgentController(FakeLLMClient([]))

    assert controller._allowed_actions_for_goal(
        AgentState(user_goal="localize pen and black cube")
    ) == {"detect_object", "detect_objects", "move_to_named_target"}


def test_agent_accepts_one_multi_object_localization_call():
    client = FakeLLMClient([json.dumps({
        "kind": "tool_call",
        "message": "Localize both requested objects from one RGB-D frame.",
        "tool_call": {
            "tool_name": "detect_objects",
            "arguments": {"queries": ["pen", "black cube"]},
        },
    })])

    proposal = AgentController(client).propose_next_action(
        AgentState(user_goal="localize pen and black cube")
    )

    assert proposal.plan_step.skill_id == "detect_objects"
    assert proposal.plan_step.parameters["queries"] == ["pen", "black cube"]
    assert proposal.requires_human_gate is False
    assert proposal.approval_stages == []
    prompt = client.messages[0][1]["content"]
    assert "multi-object localization" in prompt


def test_scene_scan_wording_and_common_typo_route_to_scene_description_without_llm():
    client = FakeLLMClient([])

    proposal = AgentController(client).propose_next_action(
        AgentState(user_goal="scan the senario")
    )

    assert proposal.plan_step.skill_id == "describe_scene"
    assert proposal.requires_human_gate is False
    assert "interpret" in proposal.decision.message
    assert client.messages == []


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
    assert "allowed_next_actions" in prompt
    assert "target_grounding" in prompt
    assert "required_output_examples" not in prompt
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


def test_pick_policy_rejects_a_default_example_target_not_in_the_user_goal():
    client = FakeLLMClient([json.dumps({
        "kind": "composite_skill_call",
        "message": "Use the reviewed grasp workflow.",
        "composite_skill_call": {
            "skill_name": "safe_pick_object",
            "arguments": {"object_query": "red cube"},
        },
    })])

    proposal = AgentController(client).propose_next_action(
        AgentState(user_goal="pick the blue bottle")
    )

    assert proposal.decision.kind.value == "request_human"
    assert proposal.composite_skill is None
    assert "default example" in proposal.decision.message


def test_mixed_pick_and_return_home_goal_keeps_composite_and_follow_up_primitive_available():
    controller = AgentController(FakeLLMClient([]))
    allowed = controller._allowed_actions_for_goal(
        AgentState(user_goal="pick the red cube and then return home")
    )

    assert {"safe_pick_object", "move_to_named_target"}.issubset(allowed)


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
            AgentState(user_goal="pick the blue object")
        )
