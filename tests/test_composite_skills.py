import pytest

from llm_skill_robot.agent.composite_skills import (
    CompositeSkillCall,
    CompositeSkillRegistry,
    CompositeSkillStatus,
    SafePickObjectSkill,
)
from llm_skill_robot.agent.tool_registry import AgentToolResult


def _success(tool_name, output=None):
    return AgentToolResult(
        tool_name=tool_name,
        success=True,
        message=f"{tool_name} succeeded",
        output=output or {},
    )


def test_safe_pick_emits_the_fixed_supervised_primitive_sequence():
    skill = SafePickObjectSkill()
    skill.start({"object_query": "red cube"})

    observed = []
    while skill.status == CompositeSkillStatus.RUNNING:
        call = skill.next_tool_call()
        assert call is not None
        observed.append(call)
        output = {}
        if call.tool_name == "detect_object":
            output = {"object_id": "cube_1"}
        elif call.tool_name == "generate_grasp_pose":
            output = {"candidate_id": "grasp_1"}
        skill.observe_result(_success(call.tool_name, output))

    assert [call.tool_name for call in observed] == [
        "move_to_named_target", "detect_object", "build_object_point_cloud",
        "generate_grasp_pose", "review_grasp_candidate", "open_gripper",
        "move_to_pregrasp", "approach_grasp", "close_gripper", "retreat_grasp",
        "verify_grasp",
    ]
    assert observed[0].arguments == {"target_name": "observe"}
    assert observed[1].arguments == {"query": "red cube"}
    assert observed[8].arguments == {"object_id": "cube_1", "during_contact": True}
    assert skill.result().status == CompositeSkillStatus.COMPLETED
    assert skill.result().output["candidate_id"] == "grasp_1"


def test_safe_pick_stops_on_failed_or_unresolved_detection():
    skill = SafePickObjectSkill()
    skill.start({"object_query": "red cube"})
    skill.observe_result(_success(skill.next_tool_call().tool_name))
    detect = skill.next_tool_call()
    skill.observe_result(AgentToolResult(
        tool_name=detect.tool_name, success=False, message="No object found."
    ))

    assert skill.status == CompositeSkillStatus.FAILED
    assert skill.next_tool_call() is None
    assert "detect_object failed" in skill.result().message


def test_safe_pick_refuses_out_of_order_or_mismatched_results():
    skill = SafePickObjectSkill()
    skill.start({"object_query": "red cube"})
    with pytest.raises(ValueError, match="No primitive tool call"):
        skill.observe_result(_success("detect_object"))

    call = skill.next_tool_call()
    with pytest.raises(ValueError, match="expected"):
        skill.observe_result(_success("detect_object"))
    assert call.tool_name == "move_to_named_target"


def test_composite_registry_creates_safe_pick_and_rejects_unknown_skills():
    registry = CompositeSkillRegistry()
    skill = registry.create(CompositeSkillCall(
        skill_name="safe_pick_object", arguments={"object_query": "red cube"}
    ))
    assert skill.next_tool_call().tool_name == "move_to_named_target"
    with pytest.raises(ValueError, match="Unknown composite skill"):
        registry.create(CompositeSkillCall(skill_name="unsafe_pick", arguments={}))
