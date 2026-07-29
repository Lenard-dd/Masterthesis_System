"""Safe runtime bridge from one Agent decision to the existing executor.

This module owns orchestration only.  It does not relax any verifier,
hardware-mode, state-machine, or ``ExecutionManager`` safety rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from llm_skill_robot.agent.agent_controller import AgentController, AgentProposal
from llm_skill_robot.agent.agent_state import AgentState
from llm_skill_robot.agent.tool_registry import AgentToolResult
from llm_skill_robot.core.plan import RobotPlan
from llm_skill_robot.core.schemas import (
    ExecutionReport,
    ValidationDecision,
    ValidationResult,
    VerificationDecision,
    VerificationResult,
)


HumanGate = Callable[[RobotPlan, AgentProposal, AgentState], Any]


@dataclass
class AgentTurnResult:
    """Auditable result of one propose-verify-gate-execute Agent turn."""

    proposal: AgentProposal
    verification: VerificationResult | None = None
    human_decision: ValidationResult | None = None
    execution_report: ExecutionReport | None = None
    tool_result: AgentToolResult | None = None


class AgentRuntime:
    """Run exactly one Agent step through the project's existing safety chain.

    ``human_gate`` is mandatory for ``human_review`` and ``human_confirm``
    tools.  If it is absent, the runtime safely rejects those proposals.
    Observation tools may proceed under the explicit low-risk policy, but the
    executor remains free to apply stricter hardware-mode checks.
    """

    def __init__(
        self,
        controller: AgentController,
        plan_verifier,
        execution_manager,
        human_gate: HumanGate | None = None,
    ):
        self.controller = controller
        self.plan_verifier = plan_verifier
        self.execution_manager = execution_manager
        self.human_gate = human_gate

    def run_turn(self, state: AgentState) -> AgentTurnResult:
        proposal = self.controller.propose_next_action(state)
        if proposal.plan_step is None:
            return AgentTurnResult(proposal=proposal)

        plan = RobotPlan(task=state.user_goal, steps=[proposal.plan_step])
        verification = self.plan_verifier.verify_plan(plan, context=state.context)
        if verification.decision != VerificationDecision.ALLOW:
            result = self.controller.record_tool_result(
                state,
                {
                    "success": False,
                    "message": "Plan verification blocked the proposed tool.",
                    "output": {
                        "reasons": verification.reasons,
                        "warnings": verification.warnings,
                    },
                },
            )
            return AgentTurnResult(
                proposal=proposal, verification=verification, tool_result=result
            )

        human_decision = self._human_decision(plan, proposal, state)
        if human_decision.decision != ValidationDecision.APPROVE:
            result = self.controller.record_tool_result(
                state,
                {
                    "success": False,
                    "message": "Human/policy gate did not approve the proposed tool.",
                    "output": {"decision": human_decision.decision.value},
                },
            )
            return AgentTurnResult(
                proposal=proposal,
                verification=verification,
                human_decision=human_decision,
                tool_result=result,
            )

        report = self.execution_manager.execute_plan(
            plan,
            verification,
            human_decision,
            context=state.context,
        )
        result = self.controller.record_tool_result(
            state, self._tool_result_from_report(proposal.plan_step.skill_id, report)
        )
        return AgentTurnResult(
            proposal=proposal,
            verification=verification,
            human_decision=human_decision,
            execution_report=report,
            tool_result=result,
        )

    def _human_decision(
        self, plan: RobotPlan, proposal: AgentProposal, state: AgentState
    ) -> ValidationResult:
        if not proposal.requires_human_gate:
            return ValidationResult(
                decision=ValidationDecision.APPROVE,
                comments="Allowed by low-risk autonomous observation policy.",
            )
        if self.human_gate is None:
            return ValidationResult(
                decision=ValidationDecision.REJECT,
                comments="A required human gate was not configured.",
            )
        return self._normalize_human_decision(self.human_gate(plan, proposal, state))

    @staticmethod
    def _normalize_human_decision(value: Any) -> ValidationResult:
        if isinstance(value, ValidationResult):
            return value
        if value is True or value == "APPROVE":
            return ValidationResult(decision=ValidationDecision.APPROVE)
        if value == "MODIFY":
            return ValidationResult(decision=ValidationDecision.MODIFY)
        return ValidationResult(decision=ValidationDecision.REJECT)

    @staticmethod
    def _tool_result_from_report(tool_name: str, report: ExecutionReport) -> dict[str, Any]:
        if not report.step_results:
            return {
                "success": False,
                "message": report.summary,
                "output": {"execution_status": report.status},
            }
        step = report.step_results[-1]
        return {
            "success": report.status == "SUCCESS" and step.status == "SUCCESS",
            "message": step.message,
            "output": {
                **step.output,
                "execution_status": report.status,
                "step_status": step.status,
            },
        }
