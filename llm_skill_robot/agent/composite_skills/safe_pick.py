"""State-machine constrained composite skill for a supervised object grasp."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from llm_skill_robot.agent.composite_skills.base import (
    BaseCompositeSkill,
    CompositeSkillResult,
    CompositeSkillStatus,
)
from llm_skill_robot.agent.tool_registry import AgentToolCall, AgentToolResult


class SafePickObjectArguments(BaseModel):
    """Task-level arguments; motion and gripper details stay outside the LLM."""

    model_config = ConfigDict(extra="forbid")

    object_query: str = Field(..., min_length=1)


class SafePickObjectSkill(BaseCompositeSkill):
    """Emit the reviewed full-grasp primitive sequence one step at a time.

    This workflow mirrors the existing supervised full-grasp order.  It never
    accepts an LLM-provided pose, trajectory, object ID, or gripper command.
    Object IDs are learned only from the trusted ``detect_object`` result.
    """

    skill_name = "safe_pick_object"
    description = (
        "Safely pick one queried object through the fixed supervised full-grasp "
        "workflow with per-tool verification and human gates."
    )

    _STAGES = (
        "move_to_named_target",
        "detect_object",
        "build_object_point_cloud",
        "generate_grasp_pose",
        "review_grasp_candidate",
        "open_gripper",
        "move_to_pregrasp",
        "approach_grasp",
        "close_gripper",
        "retreat_grasp",
        "verify_grasp",
    )

    def __init__(self) -> None:
        super().__init__()
        self._query: str | None = None
        self._object_id: str | None = None
        self._candidate_id: str | None = None
        self._stage_index = 0
        self._pending_call: AgentToolCall | None = None
        self._history: list[AgentToolResult] = []

    def start(self, arguments: dict[str, Any]) -> None:
        if self.status != CompositeSkillStatus.READY:
            raise ValueError("A composite skill instance may only be started once.")
        parsed = SafePickObjectArguments.model_validate(arguments)
        self._query = parsed.object_query
        self._status = CompositeSkillStatus.RUNNING
        self._message = f"Starting supervised pick workflow for {self._query!r}."

    def next_tool_call(self) -> AgentToolCall | None:
        if self.status != CompositeSkillStatus.RUNNING:
            return None
        if self._pending_call is not None:
            raise ValueError(
                "The previous primitive tool is still pending; consume its result first."
            )
        if self._stage_index >= len(self._STAGES):
            self._status = CompositeSkillStatus.COMPLETED
            self._message = "Supervised pick workflow completed."
            return None
        call = self._call_for_stage(self._STAGES[self._stage_index])
        self._pending_call = call
        return call

    def observe_result(self, result: AgentToolResult) -> None:
        if self.status != CompositeSkillStatus.RUNNING:
            raise ValueError(f"Cannot consume a result while workflow is {self.status.value}.")
        if self._pending_call is None:
            raise ValueError("No primitive tool call is awaiting a result.")
        if result.tool_name != self._pending_call.tool_name:
            raise ValueError(
                f"Result is for {result.tool_name!r}; expected {self._pending_call.tool_name!r}."
            )
        self._history.append(result)
        stage = self._STAGES[self._stage_index]
        self._pending_call = None
        if not result.success:
            self._status = CompositeSkillStatus.FAILED
            self._message = f"{stage} failed: {result.message}"
            return

        if stage == "detect_object":
            self._object_id = self._resolved_object_id(result.output)
            if not self._object_id:
                self._status = CompositeSkillStatus.FAILED
                self._message = "detect_object succeeded without a trusted object_id."
                return
        elif stage == "generate_grasp_pose":
            candidate_id = result.output.get("candidate_id")
            if candidate_id is not None:
                self._candidate_id = str(candidate_id)

        self._stage_index += 1
        if self._stage_index == len(self._STAGES):
            self._status = CompositeSkillStatus.COMPLETED
            self._message = "Supervised pick workflow completed."

    def result(self) -> CompositeSkillResult:
        return CompositeSkillResult(
            skill_name=self.skill_name,
            status=self.status,
            message=self._message,
            output={
                "object_query": self._query,
                "object_id": self._object_id,
                "candidate_id": self._candidate_id,
                "completed_stages": [item.tool_name for item in self._history],
                "next_stage": (
                    self._STAGES[self._stage_index]
                    if self.status == CompositeSkillStatus.RUNNING
                    else None
                ),
            },
        )

    def _call_for_stage(self, stage: str) -> AgentToolCall:
        if stage == "move_to_named_target":
            return AgentToolCall(
                tool_name=stage,
                arguments={"target_name": "observe"},
            )
        if stage == "detect_object":
            return AgentToolCall(tool_name=stage, arguments={"query": self._query})
        if stage in {
            "build_object_point_cloud", "generate_grasp_pose", "review_grasp_candidate",
            "move_to_pregrasp", "approach_grasp", "retreat_grasp", "verify_grasp",
        }:
            return AgentToolCall(
                tool_name=stage,
                arguments=self._object_arguments(stage),
            )
        if stage == "open_gripper":
            return AgentToolCall(tool_name=stage, arguments={})
        if stage == "close_gripper":
            return AgentToolCall(
                tool_name=stage,
                arguments={"object_id": self._required_object_id(), "during_contact": True},
            )
        raise ValueError(f"Unsupported supervised pick stage: {stage}")

    def _object_arguments(self, stage: str) -> dict[str, Any]:
        arguments: dict[str, Any] = {"object_id": self._required_object_id()}
        if stage == "review_grasp_candidate" and self._candidate_id:
            arguments["candidate_id"] = self._candidate_id
        return arguments

    def _required_object_id(self) -> str:
        if not self._object_id:
            raise ValueError("The workflow has no trusted object_id from detect_object.")
        return self._object_id

    @staticmethod
    def _resolved_object_id(output: dict[str, Any]) -> str | None:
        direct = output.get("object_id")
        if direct:
            return str(direct)
        objects = output.get("objects")
        if isinstance(objects, list) and objects and isinstance(objects[0], dict):
            value = objects[0].get("object_id")
            return str(value) if value else None
        return None
