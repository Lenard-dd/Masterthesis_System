"""Schema-first Agent tools built from the project's reviewed skill library.

The module is deliberately an *intent* layer.  It validates an LLM tool call
and converts it to one ``PlanStep``; it never imports ROS, MoveIt, or a robot
backend.  Execution must still pass through ``PlanVerifier`` and
``ExecutionManager``.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, create_model

from llm_skill_robot.core.plan import PlanStep
from llm_skill_robot.core.skill import Parameter, ParameterType, Skill
from llm_skill_robot.core.skill_library import SkillLibrary


class ToolSafetyTier(str, Enum):
    """Risk category used by a later Agent policy/HITL layer."""

    OBSERVE = "observe"
    PLAN = "plan"
    HUMAN_REVIEW = "human_review"
    HUMAN_CONFIRM = "human_confirm"


class AgentToolCall(BaseModel):
    """Provider-neutral structured action proposed by an LLM."""

    model_config = ConfigDict(extra="forbid")

    tool_name: str = Field(..., min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)


class AgentToolResult(BaseModel):
    """Normalized observation returned after a tool has been executed."""

    model_config = ConfigDict(extra="forbid")

    tool_name: str
    success: bool
    message: str
    output: dict[str, Any] = Field(default_factory=dict)


class ToolArguments(BaseModel):
    """Base for dynamically created per-skill argument models."""

    model_config = ConfigDict(extra="forbid")


class AgentTool:
    """A safe, schema-described view of one existing project skill."""

    def __init__(self, skill: Skill, safety_tier: ToolSafetyTier | None = None):
        self.skill = skill
        self.name = skill.skill_id
        self.safety_tier = safety_tier or self._default_safety_tier(skill.skill_id)
        self.arguments_model = self._build_arguments_model(skill)

    @staticmethod
    def _default_safety_tier(skill_id: str) -> ToolSafetyTier:
        if skill_id in {
            "describe_scene", "detect_object", "build_object_point_cloud", "generate_grasp_pose",
            "get_gripper_state", "verify_grasp",
        }:
            return ToolSafetyTier.OBSERVE
        if skill_id in {
            "review_grasp_candidate", "move_to_named_target", "move_to_pregrasp",
            "approach_grasp", "retreat_grasp", "preview_grasp_motion",
        }:
            return ToolSafetyTier.HUMAN_REVIEW
        if skill_id in {
            "open_gripper", "close_gripper", "set_gripper_width",
            "test_gripper_width", "pick_object", "place_object",
        }:
            return ToolSafetyTier.HUMAN_CONFIRM
        return ToolSafetyTier.PLAN

    @classmethod
    def _build_arguments_model(cls, skill: Skill) -> type[ToolArguments]:
        fields: dict[str, tuple[Any, Any]] = {}
        for parameter in skill.parameters:
            annotation = cls._python_type(parameter)
            default = ... if parameter.required else parameter.default
            field_kwargs: dict[str, Any] = {}
            if parameter.min_value is not None:
                field_kwargs["ge"] = parameter.min_value
            if parameter.max_value is not None:
                field_kwargs["le"] = parameter.max_value
            if parameter.allowed_values:
                # Keep the JSON Schema useful for tool-calling providers.  The
                # value is also enforced by ``validate_arguments`` below.
                field_kwargs["json_schema_extra"] = {
                    "enum": parameter.allowed_values,
                }
            fields[parameter.name] = (annotation, Field(default, **field_kwargs))
        return create_model(
            f"{''.join(part.title() for part in skill.skill_id.split('_'))}Arguments",
            __base__=ToolArguments,
            **fields,
        )

    @staticmethod
    def _python_type(parameter: Parameter) -> type:
        return {
            ParameterType.STRING: str,
            ParameterType.FLOAT: float,
            ParameterType.INT: int,
            ParameterType.BOOL: bool,
        }[parameter.type]

    def validate_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Validate and normalize arguments without executing a robot action."""
        validated = self.arguments_model.model_validate(arguments).model_dump(
            exclude_none=True
        )
        for parameter in self.skill.parameters:
            if parameter.allowed_values and parameter.name in validated:
                value = validated[parameter.name]
                if value not in parameter.allowed_values:
                    raise ValueError(
                        f"Invalid value {value!r} for {self.name}.{parameter.name}; "
                        f"allowed values: {parameter.allowed_values}."
                    )
        return validated

    def to_plan_step(self, call: AgentToolCall, step_id: str) -> PlanStep:
        """Turn a validated tool intent into a plan-only step.

        This bridge intentionally does not bind to a backend.  The caller must
        verify the resulting plan against current scene, hardware and HITL
        context before any execution is attempted.
        """
        if call.tool_name != self.name:
            raise ValueError(f"Tool call is for {call.tool_name!r}, not {self.name!r}.")
        return PlanStep(
            step_id=str(step_id),
            skill_id=self.name,
            description=self.skill.description,
            parameters=self.validate_arguments(call.arguments),
        )

    def normalize_result(self, result: Any) -> AgentToolResult:
        """Normalize backend observations for the future Agent feedback loop.

        Backends currently return dictionaries with slightly different shapes.
        This preserves their data under ``output`` while giving the Agent one
        reliable success/message contract.
        """
        if isinstance(result, AgentToolResult):
            if result.tool_name != self.name:
                raise ValueError(f"Result is for {result.tool_name!r}, not {self.name!r}.")
            return result
        if not isinstance(result, dict):
            return AgentToolResult(
                tool_name=self.name,
                success=False,
                message="Tool backend returned a non-dictionary result.",
                output={"raw_result": repr(result)},
            )
        output = result.get("output", result)
        return AgentToolResult(
            tool_name=self.name,
            success=bool(result.get("success", False)),
            message=str(result.get("message", "Tool execution completed.")),
            output=output if isinstance(output, dict) else {"value": output},
        )

    def openai_function_schema(self) -> dict[str, Any]:
        """Return an OpenAI-compatible function schema without SDK coupling."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.skill.description,
                "parameters": self.arguments_model.model_json_schema(),
            },
        }

    def model_context(self) -> dict[str, Any]:
        """Compact, provider-neutral metadata for prompts or graph nodes."""
        return {
            "name": self.name,
            "description": self.skill.description,
            "safety_tier": self.safety_tier.value,
            "preconditions": self.skill.preconditions,
            "effects": self.skill.effects,
            "input_schema": self.arguments_model.model_json_schema(),
            "output_schema": AgentToolResult.model_json_schema(),
        }


class AgentToolRegistry:
    """Build Agent tools from reviewed YAML skills, with no duplicate catalog."""

    def __init__(self, skill_library: SkillLibrary | None = None):
        self.skill_library = skill_library or SkillLibrary()
        self._tools = {
            skill_id: AgentTool(skill)
            for skill_id in self.skill_library.list_skills()
            if (skill := self.skill_library.get_skill(skill_id)) is not None
        }

    def get(self, tool_name: str) -> AgentTool | None:
        return self._tools.get(tool_name)

    def list_tools(self) -> list[AgentTool]:
        return [self._tools[name] for name in sorted(self._tools)]

    def validate_call(self, call: AgentToolCall) -> dict[str, Any]:
        tool = self.get(call.tool_name)
        if tool is None:
            raise ValueError(f"Unknown Agent tool: {call.tool_name}")
        return tool.validate_arguments(call.arguments)

    def call_to_plan_step(self, call: AgentToolCall, step_id: str) -> PlanStep:
        tool = self.get(call.tool_name)
        if tool is None:
            raise ValueError(f"Unknown Agent tool: {call.tool_name}")
        return tool.to_plan_step(call, step_id)

    def openai_tools(self) -> list[dict[str, Any]]:
        return [tool.openai_function_schema() for tool in self.list_tools()]
