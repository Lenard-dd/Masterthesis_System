"""Single-step, constrained LLM Agent controller.

The controller proposes one reviewed tool intent at a time.  It does not
execute it: callers must route its ``PlanStep`` through the existing verifier,
HITL gate, state machine, and execution manager.
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from llm_skill_robot.agent.agent_state import AgentState, AgentStatus
from llm_skill_robot.agent.tool_registry import (
    AgentToolCall,
    AgentToolRegistry,
    AgentToolResult,
    ToolSafetyTier,
)
from llm_skill_robot.core.plan import PlanStep


class AgentDecisionError(RuntimeError):
    """Raised when the LLM response is not a valid constrained Agent decision."""


class AgentDecisionKind(str, Enum):
    TOOL_CALL = "tool_call"
    REQUEST_HUMAN = "request_human"
    COMPLETE = "complete"
    BLOCKED = "blocked"


class AgentDecision(BaseModel):
    """The only response format accepted from an Agent LLM turn."""

    model_config = ConfigDict(extra="forbid")

    kind: AgentDecisionKind
    message: str = Field(..., min_length=1)
    tool_call: AgentToolCall | None = None

    @model_validator(mode="after")
    def _validate_shape(self):
        if self.kind == AgentDecisionKind.TOOL_CALL and self.tool_call is None:
            raise ValueError("tool_call is required when kind is tool_call.")
        if self.kind != AgentDecisionKind.TOOL_CALL and self.tool_call is not None:
            raise ValueError("tool_call is allowed only when kind is tool_call.")
        return self


class AgentProposal(BaseModel):
    """A validated next action ready for verification, not execution."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    decision: AgentDecision
    plan_step: PlanStep | None = None
    safety_tier: ToolSafetyTier | None = None
    requires_human_gate: bool = False


class AgentController:
    """Build prompts and validate one next action against the Tool Registry."""

    SYSTEM_PROMPT = (
        "You are a constrained robot task agent. Decide exactly one next action. "
        "You may only use tools supplied in the tool catalog. Never output ROS, "
        "shell, driver, trajectory, or joint commands. Return only JSON matching "
        "the AgentDecision schema. A tool proposal is not permission to execute; "
        "verification and human safety gates run outside the model."
    )

    def __init__(self, llm_client, tool_registry: AgentToolRegistry | None = None):
        self.llm_client = llm_client
        self.tool_registry = tool_registry or AgentToolRegistry()

    def propose_next_action(self, state: AgentState) -> AgentProposal:
        """Ask the model for one action and return a validated plan-only proposal."""
        if state.status != AgentStatus.ACTIVE:
            raise AgentDecisionError(
                f"Cannot decide next action while agent is {state.status.value}."
            )
        response = self.llm_client.generate_text(self._build_messages(state))
        decision = self._parse_decision(response)

        if decision.kind == AgentDecisionKind.REQUEST_HUMAN:
            state.request_human_input(decision.message)
            return AgentProposal(decision=decision, requires_human_gate=True)
        if decision.kind == AgentDecisionKind.COMPLETE:
            state.status = AgentStatus.COMPLETED
            return AgentProposal(decision=decision)
        if decision.kind == AgentDecisionKind.BLOCKED:
            state.status = AgentStatus.BLOCKED
            return AgentProposal(decision=decision)

        assert decision.tool_call is not None
        return self.prepare_tool_call(state, decision.tool_call, decision.message)

    def prepare_tool_call(
        self,
        state: AgentState,
        call: AgentToolCall,
        message: str = "Tool call prepared by a constrained workflow.",
    ) -> AgentProposal:
        """Validate one non-LLM tool intent for a Composite Skill runner.

        Composite skills may choose only their fixed, reviewed primitive
        sequence.  They still use this method so their calls receive the same
        schema validation and risk classification as LLM-proposed calls.
        """
        if state.status != AgentStatus.ACTIVE:
            raise AgentDecisionError(
                f"Cannot prepare a tool call while agent is {state.status.value}."
            )
        decision = AgentDecision(
            kind=AgentDecisionKind.TOOL_CALL,
            message=message,
            tool_call=call,
        )
        tool = self.tool_registry.get(call.tool_name)
        if tool is None:
            raise AgentDecisionError(f"Requested unknown Agent tool: {call.tool_name}")
        try:
            plan_step = tool.to_plan_step(call, state.next_step_id)
        except (TypeError, ValueError) as exc:
            raise AgentDecisionError(f"Invalid arguments for {tool.name}: {exc}") from exc
        state.record_proposal(call)
        requires_human_gate = tool.safety_tier in {
            ToolSafetyTier.HUMAN_REVIEW,
            ToolSafetyTier.HUMAN_CONFIRM,
        }
        return AgentProposal(
            decision=decision,
            plan_step=plan_step,
            safety_tier=tool.safety_tier,
            requires_human_gate=requires_human_gate,
        )

    def record_tool_result(self, state: AgentState, result: Any) -> AgentToolResult:
        """Normalize trusted backend feedback and make it available next turn."""
        if not state.proposed_calls:
            raise AgentDecisionError("No proposed tool call is awaiting a result.")
        tool = self.tool_registry.get(state.proposed_calls[-1].tool_name)
        assert tool is not None  # Calls are validated before being recorded.
        normalized = tool.normalize_result(result)
        state.record_tool_result(normalized)
        return normalized

    def _build_messages(self, state: AgentState) -> list[dict[str, str]]:
        prompt = {
            "agent_state": state.model_dump(),
            "tool_catalog": [tool.model_context() for tool in self.tool_registry.list_tools()],
            "decision_schema": AgentDecision.model_json_schema(),
            "policy": {
                "one_tool_per_turn": True,
                "human_review_tools_require_external_gate": [
                    ToolSafetyTier.HUMAN_REVIEW.value,
                    ToolSafetyTier.HUMAN_CONFIRM.value,
                ],
                "on_uncertainty": "request_human or choose a safe observation tool",
            },
        }
        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(prompt, sort_keys=True)},
        ]

    @staticmethod
    def _parse_decision(response: str) -> AgentDecision:
        text = response.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(lines[1:-1]).strip() if len(lines) >= 2 else text
        try:
            return AgentDecision.model_validate(json.loads(text))
        except (json.JSONDecodeError, ValueError) as exc:
            raise AgentDecisionError(f"LLM did not return valid AgentDecision JSON: {exc}") from exc
