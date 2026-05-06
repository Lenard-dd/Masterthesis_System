from typing import Any, Dict, Optional

from .plan import RobotPlan
from .skill_library import SkillLibrary
from .schemas import (
    ExecutionReport,
    StepExecutionResult,
    VerificationResult,
    ValidationResult,
    ValidationDecision,
    VerificationDecision,
)


class ExecutionManager:
    def __init__(self, skill_library: SkillLibrary, robot_backend: Any, audit_logger: Any = None):
        self.skill_library = skill_library
        self.robot_backend = robot_backend
        self.audit_logger = audit_logger

    def execute_plan(
        self,
        plan: RobotPlan,
        verification_result: VerificationResult,
        human_decision: Any,
        context: Optional[Dict[str, Any]] = None,
    ) -> ExecutionReport:
        if verification_result.decision != VerificationDecision.ALLOW:
            return ExecutionReport(
                task=plan.task,
                status="NOT_EXECUTED",
                step_results=[],
                summary="Plan verification did not allow execution."
            )

        if not self._is_human_approved(human_decision):
            return ExecutionReport(
                task=plan.task,
                status="NOT_EXECUTED",
                step_results=[],
                summary="Human decision did not approve execution."
            )

        step_results = []
        execution_failed = False

        for step in plan.steps:
            if execution_failed:
                step_results.append(
                    StepExecutionResult(
                        step_id=int(step.step_id),
                        skill_id=step.skill_id,
                        status="SKIPPED",
                        message="Step skipped after previous failure.",
                        output={}
                    )
                )
                continue

            skill = self.skill_library.get_skill(step.skill_id)
            if not skill:
                step_results.append(
                    StepExecutionResult(
                        step_id=int(step.step_id),
                        skill_id=step.skill_id,
                        status="FAILED",
                        message=f"Skill '{step.skill_id}' not found.",
                        output={}
                    )
                )
                execution_failed = True
                continue

            backend_method_name = skill.execution.backend_method
            backend_method = getattr(self.robot_backend, backend_method_name, None)
            if backend_method is None or not callable(backend_method):
                step_results.append(
                    StepExecutionResult(
                        step_id=int(step.step_id),
                        skill_id=step.skill_id,
                        status="FAILED",
                        message=f"Backend method '{backend_method_name}' not available.",
                        output={}
                    )
                )
                execution_failed = True
                continue

            backend_kwargs = dict(step.parameters)
            if backend_method_name == "move_to_named_target" and "target_name" in backend_kwargs and "target" not in backend_kwargs:
                backend_kwargs["target"] = backend_kwargs.pop("target_name")

            backend_result = backend_method(**backend_kwargs)
            step_result = self._convert_backend_result(step, backend_result)
            step_results.append(step_result)
            if step_result.status != "SUCCESS":
                execution_failed = True

        status = "SUCCESS" if not execution_failed else "FAILED"
        summary = "All steps executed successfully." if status == "SUCCESS" else "One or more steps failed."
        report = ExecutionReport(
            task=plan.task,
            status=status,
            step_results=step_results,
            summary=summary,
        )

        if self.audit_logger and hasattr(self.audit_logger, "log_execution_report"):
            self.audit_logger.log_execution_report(report)

        return report

    def _is_human_approved(self, human_decision: Any) -> bool:
        if isinstance(human_decision, str):
            return human_decision.strip().lower() in {"approved", "approve"}

        if isinstance(human_decision, ValidationResult):
            return human_decision.decision == ValidationDecision.APPROVE

        return False

    def _convert_backend_result(self, step: Any, backend_result: Any) -> StepExecutionResult:
        if isinstance(backend_result, dict):
            success = backend_result.get("success", False)
            message = backend_result.get("message", "No message provided.")
            output = backend_result.get("output", {})
        else:
            success = False
            message = "Backend method returned invalid result."
            output = {}

        status = "SUCCESS" if success else "FAILED"
        return StepExecutionResult(
            step_id=int(step.step_id),
            skill_id=step.skill_id,
            status=status,
            message=message,
            output=output if isinstance(output, dict) else {"value": output}
        )