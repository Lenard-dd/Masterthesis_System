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

    def request_human_input(self, message: str) -> None:
        self.status = AgentStatus.WAITING_FOR_HUMAN
        self.human_messages.append(message)

    def resume_after_human_input(self, message: str) -> None:
        if self.status != AgentStatus.WAITING_FOR_HUMAN:
            raise ValueError("Agent is not waiting for human input.")
        self.human_messages.append(message)
        self.status = AgentStatus.ACTIVE
