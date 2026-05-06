from typing import Any, Dict

from .plan import RobotPlan
from .schemas import ExecutionReport


class ExecutionEvaluator:
    def evaluate(self, plan: RobotPlan, execution_report: ExecutionReport) -> Dict[str, Any]:
        num_steps = len(plan.steps)
        num_successful_steps = sum(1 for step in execution_report.step_results if step.status == "SUCCESS")
        failed_steps = [
            {
                "step_id": step.step_id,
                "skill_id": step.skill_id,
                "message": step.message,
            }
            for step in execution_report.step_results
            if step.status == "FAILED"
        ]

        summary = (
            f"Task '{plan.task}' succeeded with {num_successful_steps}/{num_steps} successful steps."
            if execution_report.status == "SUCCESS"
            else f"Task '{plan.task}' failed with {len(failed_steps)} failed steps out of {num_steps}."
        )

        return {
            "task_success": execution_report.status == "SUCCESS",
            "num_steps": num_steps,
            "num_successful_steps": num_successful_steps,
            "failed_steps": failed_steps,
            "summary": summary,
        }
