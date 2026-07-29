"""Stateful adapter for executing one Agent tool step in RViz/fake hardware.

ROS-specific setup is intentionally supplied by the caller.  This keeps the
Agent layer importable without ROS while allowing one shared perception and
motion context to survive across a Composite Skill workflow.
"""

from __future__ import annotations

from typing import Any, Callable

from llm_skill_robot.core.plan import RobotPlan
from llm_skill_robot.core.schemas import (
    ExecutionReport,
    StepExecutionResult,
    ValidationDecision,
    VerificationDecision,
)


RvizStepExecutor = Callable[[Any, dict[str, Any]], dict[str, Any]]


class RvizSkillExecutor:
    """ExecutionManager-compatible RViz adapter with persistent context."""

    def __init__(self, step_executor: RvizStepExecutor, context: dict[str, Any] | None = None):
        self.step_executor = step_executor
        self.context = context if context is not None else {}

    def execute_plan(self, plan, verification_result, human_decision, context=None):
        if verification_result.decision != VerificationDecision.ALLOW:
            return ExecutionReport(
                task=plan.task, status="NOT_EXECUTED", step_results=[],
                summary="Plan verification did not allow RViz execution.",
            )
        if not self._approved(human_decision):
            return ExecutionReport(
                task=plan.task, status="NOT_EXECUTED", step_results=[],
                summary="Human decision did not approve RViz execution.",
            )
        if len(plan.steps) != 1:
            return ExecutionReport(
                task=plan.task, status="FAILED", step_results=[],
                summary="RvizSkillExecutor accepts exactly one Agent plan step.",
            )
        if isinstance(context, dict):
            self.context.update(context)
        step = plan.steps[0]
        result = self.step_executor(step, self.context)
        success = bool(result.get("success", False)) if isinstance(result, dict) else False
        message = result.get("message", "Invalid RViz step result.") if isinstance(result, dict) else "Invalid RViz step result."
        output = result.get("output", {}) if isinstance(result, dict) else {}
        return ExecutionReport(
            task=plan.task,
            status="SUCCESS" if success else "FAILED",
            summary=message,
            step_results=[StepExecutionResult(
                step_id=int(step.step_id), skill_id=step.skill_id,
                status="SUCCESS" if success else "FAILED",
                message=message, output=output if isinstance(output, dict) else {"value": output},
            )],
        )

    @staticmethod
    def _approved(decision: Any) -> bool:
        if isinstance(decision, str):
            return decision.lower() in {"approve", "approved"}
        return getattr(decision, "decision", None) == ValidationDecision.APPROVE
