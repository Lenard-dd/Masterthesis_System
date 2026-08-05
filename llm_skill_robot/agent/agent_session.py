"""Multi-turn orchestrator for mixed Primitive Tool and Composite Skill plans."""

from __future__ import annotations

from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field

from llm_skill_robot.agent.agent_controller import AgentDecisionError, ApprovalStage
from llm_skill_robot.agent.agent_runtime import AgentRuntime
from llm_skill_robot.agent.agent_state import AgentState, AgentStatus
from llm_skill_robot.agent.composite_skills.base import (
    BaseCompositeSkill,
    CompositeSkillResult,
)
from llm_skill_robot.agent.composite_skills.runner import CompositeSkillRunner
from llm_skill_robot.core.schemas import ValidationDecision, ValidationResult


CompositeHumanGate = Callable[[BaseCompositeSkill, AgentState], Any]


class AgentSessionResult(BaseModel):
    """Final, serializable trace summary for one autonomous Agent session."""

    model_config = ConfigDict(extra="forbid")

    status: AgentStatus
    turns: int
    termination_reason: str | None = None
    composite_results: list[CompositeSkillResult] = Field(default_factory=list)


class AgentSessionRunner:
    """Run a bounded Agent session without bypassing existing safety gates.

    Primitive tools are delegated to ``AgentRuntime``.  Composite skills first
    require a task-level human gate, then use ``CompositeSkillRunner`` so every
    internal primitive tool is still individually verified and gated.
    """

    def __init__(
        self,
        runtime: AgentRuntime,
        composite_human_gate: CompositeHumanGate | None = None,
        audit_logger: Any = None,
    ):
        self.runtime = runtime
        self.composite_runner = CompositeSkillRunner(runtime)
        self.composite_human_gate = composite_human_gate
        self.audit_logger = audit_logger

    def run(
        self,
        state: AgentState,
        *,
        max_turns: int = 20,
        max_consecutive_failures: int = 3,
        max_composite_steps: int = 32,
    ) -> AgentSessionResult:
        if max_turns <= 0 or max_consecutive_failures <= 0 or max_composite_steps <= 0:
            raise ValueError("Session limits must be positive.")
        self._audit("agent_session_started", user_goal=state.user_goal)
        composite_results: list[CompositeSkillResult] = []
        turns = 0

        while state.status == AgentStatus.ACTIVE and turns < max_turns:
            turns += 1
            try:
                turn = self.runtime.run_turn(state)
            except AgentDecisionError as exc:
                state.abort(f"Invalid Agent decision: {exc}")
                self._audit("agent_session_blocked", reason=state.termination_reason)
                break

            proposal = turn.proposal
            self._audit(
                "agent_decision",
                turn=turns,
                decision=proposal.decision.model_dump(),
            )
            if proposal.composite_skill is not None:
                call = proposal.decision.composite_skill_call
                assert call is not None
                if state.has_completed_composite_skill(call.skill_name, call.arguments):
                    state.status = AgentStatus.COMPLETED
                    state.context["duplicate_completed_composite_skill_ignored"] = {
                        "skill_name": call.skill_name,
                        "arguments": dict(call.arguments),
                    }
                    self._audit(
                        "duplicate_completed_composite_skill_ignored",
                        skill_name=call.skill_name,
                        arguments=call.arguments,
                        message=(
                            "A completed Composite Skill was proposed again; no "
                            "robot action was executed and the session was completed."
                        ),
                    )
                    break
                result = self._run_composite_skill(proposal.composite_skill, state, max_composite_steps)
                composite_results.append(result)
                state.active_composite_skill = None
                state.context.pop("active_composite_skill", None)
                self._audit(
                    "composite_skill_finished",
                    skill_name=result.skill_name,
                    status=result.status.value,
                    result=result.model_dump(),
                )
                if result.status.value == "completed":
                    state.record_completed_composite_skill(
                        result.skill_name, call.arguments, result.output
                    )
                else:
                    state.record_workflow_failure(result.message)
            elif turn.tool_result is not None:
                self._audit(
                    "agent_tool_result",
                    tool_result=turn.tool_result.model_dump(),
                    verification=(turn.verification.model_dump() if turn.verification else None),
                    human_decision=(turn.human_decision.model_dump() if turn.human_decision else None),
                )

            if state.consecutive_failures >= max_consecutive_failures:
                state.abort(
                    "Maximum consecutive failures reached "
                    f"({max_consecutive_failures})."
                )
                self._audit("agent_session_blocked", reason=state.termination_reason)
                break

        if state.status == AgentStatus.ACTIVE and turns >= max_turns:
            state.abort(f"Maximum session turns reached ({max_turns}).")
            self._audit("agent_session_blocked", reason=state.termination_reason)

        result = AgentSessionResult(
            status=state.status,
            turns=turns,
            termination_reason=state.termination_reason,
            composite_results=composite_results,
        )
        self._audit("agent_session_finished", result=result.model_dump())
        return result

    def _run_composite_skill(
        self,
        skill: BaseCompositeSkill,
        state: AgentState,
        max_composite_steps: int,
    ) -> CompositeSkillResult:
        decision = self._composite_human_decision(skill, state)
        self._audit(
            "composite_skill_human_gate",
            skill_name=skill.skill_name,
            decision=decision.model_dump(),
        )
        if decision.decision != ValidationDecision.APPROVE:
            skill.abort("Task-level human gate did not approve the composite skill.")
            return skill.result()
        # The Composite Skill's task-level gate is approval point A. Its
        # reviewed primitive stages still request their own B/C/D gates.
        state.context["task_intent_approved"] = True
        return self.composite_runner.run_until_terminal(
            skill, state, max_steps=max_composite_steps
        )

    def _composite_human_decision(
        self, skill: BaseCompositeSkill, state: AgentState
    ) -> ValidationResult:
        if self.composite_human_gate is None:
            return ValidationResult(
                decision=ValidationDecision.REJECT,
                comments="A task-level human gate is required for composite skills.",
            )
        value = self.composite_human_gate(skill, state)
        if isinstance(value, ValidationResult):
            return value
        if value is True or value == "APPROVE":
            return ValidationResult(decision=ValidationDecision.APPROVE)
        if value == "MODIFY":
            return ValidationResult(decision=ValidationDecision.MODIFY)
        return ValidationResult(decision=ValidationDecision.REJECT)

    def _audit(self, event: str, **fields: Any) -> None:
        if self.audit_logger and hasattr(self.audit_logger, "log_agent_event"):
            self.audit_logger.log_agent_event(event, **fields)
