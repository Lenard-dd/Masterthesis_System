import pytest
from pydantic import ValidationError

from llm_skill_robot.agent.tool_registry import AgentToolCall, AgentToolRegistry, ToolSafetyTier


def test_registry_derives_tools_and_openai_schema_from_skill_library():
    registry = AgentToolRegistry()

    detect = registry.get("detect_object")
    assert detect is not None
    assert detect.safety_tier == ToolSafetyTier.OBSERVE
    schema = detect.openai_function_schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "detect_object"
    assert schema["function"]["parameters"]["properties"]["query"]["type"] == "string"
    assert "query" in schema["function"]["parameters"]["required"]


def test_registry_validates_types_bounds_enums_and_extra_arguments():
    registry = AgentToolRegistry()
    move = registry.get("move_to_named_target")
    assert move is not None
    assert move.validate_arguments({"target_name": "observe"}) == {
        "target_name": "observe",
        "velocity_scale": 0.03,
        "acceleration_scale": 0.03,
    }

    with pytest.raises(ValueError, match="allowed values"):
        move.validate_arguments({"target_name": "table"})
    with pytest.raises(ValidationError):
        move.validate_arguments({"target_name": "observe", "velocity_scale": 0.2})
    with pytest.raises(ValidationError):
        move.validate_arguments({"target_name": "observe", "ros_command": "unsafe"})


def test_validated_tool_call_converts_only_to_a_plan_step():
    registry = AgentToolRegistry()
    step = registry.call_to_plan_step(
        AgentToolCall(tool_name="detect_object", arguments={"query": "red cube"}),
        step_id="3",
    )

    assert step.step_id == "3"
    assert step.skill_id == "detect_object"
    assert step.parameters["query"] == "red cube"
    assert step.parameters["min_confidence"] == 0.5


def test_tool_normalizes_backend_feedback_for_the_next_agent_turn():
    registry = AgentToolRegistry()
    result = registry.get("detect_object").normalize_result(
        {
            "success": True,
            "message": "Detected one object.",
            "output": {"object_id": "cube_1", "confidence": 0.94},
        }
    )

    assert result.success is True
    assert result.tool_name == "detect_object"
    assert result.output["object_id"] == "cube_1"


def test_tool_call_rejects_unknown_tool():
    registry = AgentToolRegistry()
    with pytest.raises(ValueError, match="Unknown Agent tool"):
        registry.call_to_plan_step(
            AgentToolCall(tool_name="send_ros_command", arguments={}), step_id="1"
        )
