"""Bridge Composite Skills to the normal AgentRuntime safety pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from llm_skill_robot.agent.agent_runtime import AgentRuntime, AgentTurnResult
from llm_skill_robot.agent.agent_state import AgentState
from llm_skill_robot.agent.composite_skills.base import (
    BaseCompositeSkill,
    CompositeSkillResult,
    CompositeSkillStatus,
)
from llm_skill_robot.agent.tool_registry import AgentToolCall


@dataclass
class CompositeSkillTurnResult:
    """One primitive stage of a Composite Skill workflow."""

    tool_call: AgentToolCall | None
    runtime_turn: AgentTurnResult | None
    status: CompositeSkillStatus


class CompositeSkillRunner:
    """Execute a composite workflow one Primitive Tool at a time.

    No backend is called here directly.  Each stage always goes through
    ``AgentRuntime.run_tool_call``, which retains schema validation,
    ``PlanVerifier``, HITL, and ``ExecutionManager`` enforcement.
    """

    def __init__(self, runtime: AgentRuntime):
        self.runtime = runtime

    def run_next(
        self, skill: BaseCompositeSkill, state: AgentState
    ) -> CompositeSkillTurnResult:
        """Run exactly one workflow stage and feed back its trusted result."""
        if skill.status != CompositeSkillStatus.RUNNING:
            return CompositeSkillTurnResult(None, None, skill.status)
        call = skill.next_tool_call()
        if call is None:
            return CompositeSkillTurnResult(None, None, skill.status)
        runtime_turn = self.runtime.run_tool_call(
            state,
            call,
            message=f"Composite skill {skill.skill_name} requested {call.tool_name}.",
        )
        if runtime_turn.tool_result is None:
            raise RuntimeError(
                f"Runtime did not produce a result for composite tool {call.tool_name}."
            )
        skill.observe_result(runtime_turn.tool_result)
        return CompositeSkillTurnResult(call, runtime_turn, skill.status)

    def run_until_terminal(
        self,
        skill: BaseCompositeSkill,
        state: AgentState,
        max_steps: int = 32,
    ) -> CompositeSkillResult:
        """Convenience loop for mock/RViz demos; real UIs should call run_next."""
        if max_steps <= 0:
            raise ValueError("max_steps must be positive.")
        for _ in range(max_steps):
            if skill.status != CompositeSkillStatus.RUNNING:
                return skill.result()
            self.run_next(skill, state)
        skill.abort(f"Composite skill exceeded max_steps={max_steps}.")
        return skill.result()
