"""Base interfaces for stateful, task-level Agent skills."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from llm_skill_robot.agent.tool_registry import AgentToolCall, AgentToolResult


class CompositeSkillStatus(str, Enum):
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


class CompositeSkillCall(BaseModel):
    """Task-level intent selected by the LLM Agent."""

    model_config = ConfigDict(extra="forbid")

    skill_name: str = Field(..., min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)


class CompositeSkillResult(BaseModel):
    """Structured outcome available to the Agent after a workflow ends."""

    model_config = ConfigDict(extra="forbid")

    skill_name: str
    status: CompositeSkillStatus
    message: str
    output: dict[str, Any] = Field(default_factory=dict)


class BaseCompositeSkill(ABC):
    """A safe workflow that emits one primitive tool call at a time.

    A composite skill is intentionally not a robot backend method.  The
    caller must run each returned call through ``AgentRuntime`` and pass the
    resulting trusted ``AgentToolResult`` back to ``observe_result``.
    """

    skill_name: str
    description: str
    arguments_model: type[BaseModel]

    def __init__(self) -> None:
        self._status = CompositeSkillStatus.READY
        self._message = "Composite skill has not started."

    @property
    def status(self) -> CompositeSkillStatus:
        return self._status

    @abstractmethod
    def start(self, arguments: dict[str, Any]) -> None:
        """Validate arguments and initialize the workflow."""

    @abstractmethod
    def next_tool_call(self) -> AgentToolCall | None:
        """Return the one primitive tool currently permitted by the workflow."""

    @abstractmethod
    def observe_result(self, result: AgentToolResult) -> None:
        """Consume a trusted result for the previously returned tool call."""

    @abstractmethod
    def result(self) -> CompositeSkillResult:
        """Return the current structured workflow outcome."""

    def abort(self, reason: str) -> None:
        if not reason.strip():
            raise ValueError("A non-empty composite skill abort reason is required.")
        self._status = CompositeSkillStatus.ABORTED
        self._message = reason

    @classmethod
    def model_context(cls) -> dict[str, Any]:
        """Provider-neutral metadata that may be shown to an LLM Agent."""
        return {
            "skill_name": cls.skill_name,
            "description": cls.description,
            "input_schema": cls.arguments_model.model_json_schema(),
        }
