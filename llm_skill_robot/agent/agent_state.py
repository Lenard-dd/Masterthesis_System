"""Persistent, serializable state for the constrained robot Agent loop."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from llm_skill_robot.agent.tool_registry import AgentToolCall, AgentToolResult


class AgentStatus(str, Enum):
    ACTIVE = "active"
    WAITING_FOR_HUMAN = "waiting_for_human"
    COMPLETED = "completed"
    BLOCKED = "blocked"


class AgentState(BaseModel):
    """State shared across observe-decide-verify-execute Agent turns.

    ``context`` contains only structured observations supplied by trusted
    perception/execution components.  It is never populated from free-form
    model output.
    """

    model_config = ConfigDict(extra="forbid")

    user_goal: str = Field(..., min_length=1)
    status: AgentStatus = AgentStatus.ACTIVE
    safety_mode: str = "plan_only"
    selected_object_id: str | None = None
    selected_grasp_candidate_id: str | None = None
    active_composite_skill: str | None = None
    completed_composite_skills: list[dict[str, Any]] = Field(default_factory=list)
    consecutive_failures: int = 0
    total_failures: int = 0
    termination_reason: str | None = None
    context: dict[str, Any] = Field(default_factory=dict)
    proposed_calls: list[AgentToolCall] = Field(default_factory=list)
    tool_results: list[AgentToolResult] = Field(default_factory=list)
    human_messages: list[str] = Field(default_factory=list)

    @property
    def next_step_id(self) -> str:
        """Return the next stable plan-step identifier."""
        return str(len(self.proposed_calls) + 1)

    def record_proposal(self, call: AgentToolCall) -> None:
        if self.status != AgentStatus.ACTIVE:
            raise ValueError(f"Cannot propose a tool call while agent is {self.status.value}.")
        self.proposed_calls.append(call)

    def record_tool_result(self, result: AgentToolResult) -> None:
        """Append a trusted tool observation for the next LLM decision."""
        if not self.proposed_calls:
            raise ValueError("Cannot record a result before a tool has been proposed.")
        expected = self.proposed_calls[-1].tool_name
        if result.tool_name != expected:
            raise ValueError(
                f"Result is for {result.tool_name!r}; expected latest tool {expected!r}."
            )
        self.tool_results.append(result)
        self.context["last_tool_result"] = result.model_dump()
        if result.success:
            self.consecutive_failures = 0
        else:
            self.consecutive_failures += 1
            self.total_failures += 1
        self._project_trusted_result(result)

    def record_workflow_failure(self, reason: str) -> None:
        """Record a non-tool workflow/HITL failure for session stop policies."""
        self.consecutive_failures += 1
        self.total_failures += 1
        self.context["last_workflow_failure"] = reason

    def record_completed_composite_skill(
        self, skill_name: str, arguments: dict[str, Any], output: dict[str, Any]
    ) -> None:
        """Persist a completed workflow as trusted state for later LLM turns."""
        entry = {
            "skill_name": str(skill_name),
            "arguments": dict(arguments),
            "output": dict(output),
        }
        self.completed_composite_skills.append(entry)
        self.context["completed_composite_skills"] = list(
            self.completed_composite_skills
        )

    def has_completed_composite_skill(
        self, skill_name: str, arguments: dict[str, Any]
    ) -> bool:
        """Return whether this exact task-level workflow already succeeded."""
        return any(
            entry.get("skill_name") == skill_name
            and entry.get("arguments") == arguments
            for entry in self.completed_composite_skills
        )

    def abort(self, reason: str) -> None:
        if not reason.strip():
            raise ValueError("A non-empty agent abort reason is required.")
        self.status = AgentStatus.BLOCKED
        self.termination_reason = reason

    def request_human_input(self, message: str) -> None:
        self.status = AgentStatus.WAITING_FOR_HUMAN
        self.human_messages.append(message)

    def resume_after_human_input(self, message: str) -> None:
        if self.status != AgentStatus.WAITING_FOR_HUMAN:
            raise ValueError("Agent is not waiting for human input.")
        self.human_messages.append(message)
        self.status = AgentStatus.ACTIVE

    def _project_trusted_result(self, result: AgentToolResult) -> None:
        """Expose selected perception outputs as explicit, prompt-friendly state."""
        if not result.success:
            return
        if result.tool_name in {"detect_object", "detect_objects"}:
            objects = result.output.get("objects", [])
            scene = result.output.get("scene")
            if isinstance(objects, list):
                self.context["localized_objects"] = objects
            if isinstance(scene, dict):
                # This is the persistent, trusted object table for later
                # relative-place computations. It originates only from the
                # RGB-D localization backend, never free-form LLM output.
                self.context["localized_object_registry"] = scene
                self.context["scene"] = scene
        if result.tool_name == "detect_object":
            object_id = result.output.get("object_id")
            if object_id:
                self.selected_object_id = str(object_id)
        elif result.tool_name == "compute_place_pose":
            self.context["place_pose_plan"] = dict(result.output)
        elif result.tool_name == "generate_grasp_pose":
            candidate_id = result.output.get("candidate_id")
            if candidate_id:
                self.selected_grasp_candidate_id = str(candidate_id)
            candidate = result.output.get("candidate")
            if isinstance(candidate, dict):
                self.context["selected_grasp_candidate"] = candidate
