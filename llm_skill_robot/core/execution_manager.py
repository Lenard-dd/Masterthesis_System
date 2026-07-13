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
from llm_skill_robot.safety.hardware_mode import HardwareMode


class ExecutionManager:
    REAL_GRIPPER_SKILLS = {
        "open_gripper", "close_gripper", "set_gripper_width", "test_gripper_width"
    }
    CANDIDATE_GRIPPER_SKILLS = {"set_gripper_width", "test_gripper_width"}
    PLAN_ONLY_ARM_SKILLS = {
        "move_to_named_target",
        "move_to_pregrasp",
        "approach_grasp",
        "retreat_grasp",
        "preview_grasp_motion",
    }

    def __init__(
        self,
        skill_library: SkillLibrary,
        robot_backend: Any,
        audit_logger: Any = None,
        hardware_mode: HardwareMode | None = None,
    ):
        self.skill_library = skill_library
        self.robot_backend = robot_backend
        self.audit_logger = audit_logger
        self.hardware_mode = hardware_mode

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

        context = context or {}
        hardware_mode = context.get("hardware_mode_object") or self.hardware_mode
        restricted_real_motion = self._restricted_real_motion_active(
            hardware_mode, context
        )
        if restricted_real_motion:
            self._audit_hardware_event(
                "restricted_real_motion_mode_active",
                restricted_real_motion_mode_active=True,
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

            safety_failure = self._hardware_safety_failure(
                step, hardware_mode, context
            )
            if safety_failure is not None:
                step_results.append(safety_failure)
                execution_failed = True
                continue

            if step.skill_id == "review_grasp_candidate":
                review_result = self._review_grasp_candidate(step, context)
                step_results.append(review_result)
                if review_result.status != "SUCCESS":
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
            if hardware_mode is not None and step.skill_id in self.REAL_GRIPPER_SKILLS:
                backend_kwargs.pop("confirmed", None)
                backend_kwargs.pop("human_confirmed", None)
                backend_kwargs["confirmed"] = True
                self._audit_hardware_event(
                    "real_gripper_command_requested",
                    real_gripper_command_requested=True,
                    step_id=step.step_id,
                    skill_id=step.skill_id,
                )

            backend_result = backend_method(**backend_kwargs)
            step_result = self._convert_backend_result(step, backend_result)
            step_results.append(step_result)
            self._audit_step_hardware_result(
                step, backend_result, step_result, restricted_real_motion
            )
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

    def _hardware_safety_failure(
        self,
        step,
        mode: HardwareMode | None,
        context: Dict[str, Any],
    ) -> StepExecutionResult | None:
        if mode is None:
            return None
        restricted_real_motion = self._restricted_real_motion_active(mode, context)
        supervised_real_grasp = self._supervised_real_grasp_active(mode, context)
        if restricted_real_motion:
            if step.skill_id in {
                "pick_object",
                "approach_grasp",
                "lift_object",
                "execute_pick_sequence",
            }:
                return self._unsafe_llm_step_blocked(
                    step,
                    f"{step.skill_id} is blocked in restricted_real_motion mode.",
                )
            if step.skill_id == "close_gripper" and self._is_contact_close_step(step):
                return self._unsafe_llm_step_blocked(
                    step,
                    "close_gripper during contact or on an object is blocked in restricted_real_motion mode.",
                )

        if step.skill_id == "pick_object":
            self._audit_hardware_event(
                "real_arm_execution_blocked",
                real_arm_execution_blocked=True,
                step_id=step.step_id,
                skill_id=step.skill_id,
            )
            return self._failed_step(
                step, "pick_object direct execution is blocked in arm-safe hardware mode."
            )
        if step.skill_id in self.PLAN_ONLY_ARM_SKILLS:
            execution_mode = context.get(
                "arm_execution_mode", getattr(self.robot_backend, "execution_mode", None)
            )
            if supervised_real_grasp:
                allowed_modes = (
                    {"dry_run", "plan_only", "rviz_sim", "restricted_real_motion"}
                    if step.skill_id == "move_to_named_target"
                    else {"dry_run", "plan_only", "restricted_real_motion"}
                )
            else:
                allowed_modes = (
                    {"dry_run", "plan_only", "rviz_sim"}
                    if step.skill_id == "move_to_named_target"
                    else {"dry_run", "plan_only"}
                )
            if execution_mode not in allowed_modes:
                self._audit_hardware_event(
                    "real_arm_execution_blocked",
                    real_arm_execution_blocked=True,
                    step_id=step.step_id,
                    skill_id=step.skill_id,
                )
                return self._failed_step(
                    step, "Real arm execution is disabled by hardware_mode.yaml."
                )
            if (
                restricted_real_motion
                and execution_mode == "restricted_real_motion"
                and step.skill_id in {"move_to_named_target", "move_to_pregrasp"}
            ):
                if not self._per_step_confirmation(
                    context, "arm_confirmations", step.step_id, "EXECUTE"
                ):
                    self._audit_hardware_event(
                        "real_arm_step_requires_confirmation",
                        real_arm_step_requires_confirmation=True,
                        step_id=step.step_id,
                        skill_id=step.skill_id,
                    )
                    return self._failed_step(
                        step,
                        "Exact EXECUTE confirmation is required before each real arm motion.",
                    )
        if step.skill_id in self.REAL_GRIPPER_SKILLS:
            if supervised_real_grasp:
                confirmed = self._per_step_confirmation(
                    context, "gripper_confirmations", step.step_id, "YES"
                )
            else:
                confirmations = context.get("gripper_confirmations", {})
                confirmed = (
                    isinstance(confirmations, dict)
                    and confirmations.get(str(step.step_id)) == "YES"
                ) or context.get("gripper_confirmation") == "YES"
            if not confirmed:
                self._audit_hardware_event(
                    "real_gripper_step_requires_confirmation",
                    real_gripper_step_requires_confirmation=True,
                    step_id=step.step_id,
                    skill_id=step.skill_id,
                )
                return self._failed_step(
                    step,
                    "Exact YES confirmation is required before a real gripper command.",
                )
            if (
                step.skill_id in self.CANDIDATE_GRIPPER_SKILLS
                and not context.get("grasp_candidate_approved", False)
            ):
                return self._failed_step(
                    step, "A reviewed grasp candidate must be approved first."
                )
        if step.skill_id in {
            "move_to_pregrasp", "approach_grasp", "retreat_grasp", "preview_grasp_motion"
        } and not context.get("grasp_candidate_approved", False):
            return self._failed_step(
                step, "A reviewed grasp candidate must be approved first."
            )
        return None

    def _review_grasp_candidate(
        self, step, context: Dict[str, Any]
    ) -> StepExecutionResult:
        candidate = context.get("selected_grasp_candidate")
        if candidate is None:
            return self._failed_step(step, "No selected grasp candidate is available for review.")
        print("\nGrasp candidate review:")
        print(candidate)
        approved = context.get("grasp_candidate_confirmation") == "YES"
        context["grasp_candidate_approved"] = approved
        self._audit_hardware_event(
            "grasp_candidate_reviewed",
            grasp_candidate_reviewed=True,
            grasp_candidate_approved=approved,
            step_id=step.step_id,
        )
        return StepExecutionResult(
            step_id=int(step.step_id),
            skill_id=step.skill_id,
            status="SUCCESS" if approved else "FAILED",
            message=(
                "Grasp candidate reviewed and approved."
                if approved
                else "Grasp candidate review requires exact YES approval."
            ),
            output={"candidate": candidate, "approved": approved},
        )

    @staticmethod
    def _failed_step(step, message: str) -> StepExecutionResult:
        return StepExecutionResult(
            step_id=int(step.step_id),
            skill_id=step.skill_id,
            status="FAILED",
            message=message,
            output={},
        )

    def _audit_step_hardware_result(
        self,
        step,
        backend_result,
        step_result,
        restricted_real_motion: bool = False,
    ) -> None:
        if step.skill_id == "detect_object" and step_result.status == "SUCCESS":
            self._audit_hardware_event("real_camera_used", real_camera_used=True)
        if step.skill_id == "review_grasp_candidate" and step_result.status == "SUCCESS":
            self._audit_hardware_event(
                "grasp_candidate_reviewed", grasp_candidate_reviewed=True
            )
        if step.skill_id in self.REAL_GRIPPER_SKILLS:
            output = backend_result.get("output", {}) if isinstance(backend_result, dict) else {}
            sent = bool(output.get("command_sent", False))
            self._audit_hardware_event(
                "real_gripper_command_result",
                real_gripper_command_requested=True,
                real_gripper_command_sent=sent,
                step_id=step.step_id,
                skill_id=step.skill_id,
            )
            if sent:
                self._audit_hardware_event(
                    "real_gripper_command_sent",
                    real_gripper_command_sent=True,
                    step_id=step.step_id,
                    skill_id=step.skill_id,
                )
        if (
            restricted_real_motion
            and step.skill_id in {"move_to_named_target", "move_to_pregrasp"}
            and step_result.status == "SUCCESS"
        ):
            self._audit_hardware_event(
                "real_arm_motion_executed",
                real_arm_motion_executed=True,
                step_id=step.step_id,
                skill_id=step.skill_id,
            )

    def _audit_hardware_event(self, event: str, **fields: Any) -> None:
        if self.audit_logger and hasattr(self.audit_logger, "log_hardware_safety_event"):
            self.audit_logger.log_hardware_safety_event(event, **fields)

    def _unsafe_llm_step_blocked(self, step, message: str) -> StepExecutionResult:
        self._audit_hardware_event(
            "unsafe_llm_step_blocked",
            unsafe_llm_step_blocked=True,
            step_id=step.step_id,
            skill_id=step.skill_id,
            reason=message,
        )
        return self._failed_step(step, message)

    @staticmethod
    def _per_step_confirmation(
        context: Dict[str, Any],
        confirmation_key: str,
        step_id: Any,
        phrase: str,
    ) -> bool:
        confirmations = context.get(confirmation_key, {})
        return isinstance(confirmations, dict) and confirmations.get(str(step_id)) == phrase

    @staticmethod
    def _restricted_real_motion_active(
        mode: HardwareMode | None,
        context: Dict[str, Any],
    ) -> bool:
        if context.get("arm_execution_mode") == "restricted_real_motion":
            return True
        if mode is None:
            return False
        return (
            mode.mode == "real_camera_real_gripper_restricted_arm"
            or getattr(mode.arm, "execution_mode", None) == "restricted_real_motion"
        )

    @classmethod
    def _supervised_real_grasp_active(
        cls,
        mode: HardwareMode | None,
        context: Dict[str, Any],
    ) -> bool:
        """Both restricted and full-supervised hardware require per-step approval."""
        if cls._restricted_real_motion_active(mode, context):
            return True
        if context.get("arm_execution_mode") == "full_grasp_supervised":
            return True
        return mode is not None and (
            mode.mode == "real_full_grasp_supervised"
            or getattr(mode.arm, "execution_mode", None) == "full_grasp_supervised"
        )

    @staticmethod
    def _is_contact_close_step(step) -> bool:
        if step.parameters.get("during_contact") is True:
            return True
        if step.parameters.get("on_object") is True:
            return True
        text = f"{step.description} {step.parameters}".lower()
        return "during_contact" in text or "contact" in text or "on object" in text

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
