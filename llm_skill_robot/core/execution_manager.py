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
from llm_skill_robot.grasping.full_grasp_state_machine import FullGraspStateMachine


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
        full_grasp_state_machine: FullGraspStateMachine | None = None,
        confirmation_provider: Any = None,
    ):
        self.skill_library = skill_library
        self.robot_backend = robot_backend
        self.audit_logger = audit_logger
        self.hardware_mode = hardware_mode
        self.full_grasp_state_machine = full_grasp_state_machine
        self.confirmation_provider = confirmation_provider

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
        supervised_full_grasp = self._full_grasp_supervised_active(
            hardware_mode, context
        )
        state_managed_full_grasp = supervised_full_grasp and (
            self._is_full_grasp_plan(plan)
            or any(
                step.skill_id == "close_gripper" and self._is_contact_close_step(step)
                for step in plan.steps
            )
        )
        state_machine = None
        if state_managed_full_grasp:
            state_machine = context.get("full_grasp_state_machine")
            if not isinstance(state_machine, FullGraspStateMachine):
                state_machine = self.full_grasp_state_machine or FullGraspStateMachine()
            context["full_grasp_state_machine"] = state_machine
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

            if state_managed_full_grasp:
                state_failure = self._prepare_full_grasp_step(step, state_machine)
                if state_failure is not None:
                    step_results.append(state_failure)
                    execution_failed = True
                    continue

            safety_failure = self._hardware_safety_failure(
                step, hardware_mode, context
            )
            if safety_failure is not None:
                step_results.append(safety_failure)
                execution_failed = True
                if state_managed_full_grasp:
                    self._abort_full_grasp_state(
                        state_machine,
                        f"{step.skill_id} safety gate failed: {safety_failure.message}",
                    )
                continue

            if step.skill_id == "review_grasp_candidate":
                review_result = self._review_grasp_candidate(step, context)
                step_results.append(review_result)
                if review_result.status != "SUCCESS":
                    execution_failed = True
                    if state_managed_full_grasp:
                        self._abort_full_grasp_state(
                            state_machine,
                            f"review_grasp_candidate failed: {review_result.message}",
                        )
                continue

            backend_method_name = self._full_grasp_backend_method(
                step, skill.execution.backend_method, state_managed_full_grasp
            )
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
                if state_managed_full_grasp:
                    self._abort_full_grasp_state(
                        state_machine,
                        f"Backend method {backend_method_name} is unavailable.",
                    )
                continue

            backend_kwargs = dict(step.parameters)
            backend_kwargs.pop("during_contact", None)
            if (
                state_managed_full_grasp
                and backend_kwargs.get("object_id") == "<resolved>"
            ):
                resolved_object_id = self._resolved_object_id(context, state_machine)
                if not resolved_object_id:
                    failure = self._failed_step(
                        step,
                        "The <resolved> object_id is unavailable from perception/state context.",
                    )
                    step_results.append(failure)
                    execution_failed = True
                    self._abort_full_grasp_state(state_machine, failure.message)
                    continue
                backend_kwargs["object_id"] = resolved_object_id
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
            if state_managed_full_grasp:
                accepted = context.get("_accepted_step_confirmations", {}).pop(
                    str(step.step_id), None
                )
                if accepted is not None:
                    backend_kwargs["confirmation"] = accepted
                if step.skill_id in {
                    "move_to_pregrasp", "approach_grasp", "retreat_grasp"
                }:
                    grasp_plans = context.get("grasp_plans") or context.get("grasp_plan")
                    if grasp_plans is not None:
                        backend_kwargs.setdefault("grasp_plans", grasp_plans)

            backend_result = backend_method(**backend_kwargs)
            if state_managed_full_grasp:
                self._capture_full_grasp_context(
                    step, backend_result, context, state_machine
                )
            step_result = self._convert_backend_result(step, backend_result)
            step_results.append(step_result)
            self._audit_step_hardware_result(
                step, backend_result, step_result, restricted_real_motion
            )
            if step_result.status != "SUCCESS":
                execution_failed = True
                if state_managed_full_grasp:
                    self._abort_full_grasp_state(
                        state_machine,
                        f"{step.skill_id} execution failed: {step_result.message}",
                    )
            elif state_managed_full_grasp:
                transition_failure = self._complete_full_grasp_step(step, state_machine)
                if transition_failure is not None:
                    step_results[-1] = transition_failure
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
        full_grasp_supervised = self._full_grasp_supervised_active(mode, context)
        raw_keys = {
            "ros_command", "command_topic", "action_name", "service_name",
            "driver_command", "trajectory", "joint_trajectory", "controller_command",
        }.intersection(step.parameters)
        unsafe_text = f"{step.description} {step.parameters}".lower()
        if raw_keys or any(marker in unsafe_text for marker in (
            "ros2 topic pub", "ros2 action send_goal", "ros2 service call",
            "/_action/send_goal", "control_msgs/action/grippercommand",
        )):
            detail = ", ".join(sorted(raw_keys)) if raw_keys else "raw command text"
            return self._unsafe_llm_step_blocked(
                step, f"Raw ROS, trajectory, or gripper-driver commands are forbidden: {detail}."
            )
        if self._is_batch_request(step):
            return self._unsafe_llm_step_blocked(
                step,
                "Batch execution or one confirmation for multiple real steps is forbidden.",
            )
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
            return self._unsafe_llm_step_blocked(
                step, "pick_object direct execution is blocked in arm-safe hardware mode."
            )
        if step.skill_id in self.PLAN_ONLY_ARM_SKILLS:
            execution_mode = context.get(
                "arm_execution_mode", getattr(self.robot_backend, "execution_mode", None)
            )
            if supervised_real_grasp:
                allowed_modes = (
                    {"dry_run", "plan_only", "rviz_sim", "restricted_real_motion", "full_grasp_supervised"}
                    if step.skill_id == "move_to_named_target"
                    else {"dry_run", "plan_only", "restricted_real_motion", "full_grasp_supervised"}
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
            if (
                full_grasp_supervised
                and step.skill_id in {
                    "move_to_named_target", "move_to_pregrasp",
                    "approach_grasp", "retreat_grasp",
                }
            ):
                if not self._request_step_confirmation(
                    context, step, "EXECUTE", "arm"
                ):
                    return self._failed_step(
                        step,
                        "Exact EXECUTE confirmation is required before each real arm motion.",
                    )
        if step.skill_id in self.REAL_GRIPPER_SKILLS:
            if full_grasp_supervised:
                confirmed = self._request_step_confirmation(
                    context, step, "YES", "gripper"
                )
            elif supervised_real_grasp:
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

    def _prepare_full_grasp_step(
        self, step, machine: FullGraspStateMachine
    ) -> StepExecutionResult | None:
        required = {
            "detect_object": machine.INIT,
            "build_object_point_cloud": machine.OBJECT_DETECTED,
            "generate_grasp_pose": machine.POINT_CLOUD_BUILT,
            "review_grasp_candidate": machine.GRASP_GENERATED,
            "open_gripper": machine.GRASP_GENERATED,
            "move_to_pregrasp": machine.GRASP_GENERATED,
            "approach_grasp": machine.AT_PREGRASP,
            "close_gripper": machine.AT_GRASP_POSE,
            "retreat_grasp": machine.GRIPPER_CLOSED,
            "verify_grasp": machine.RETREATED,
        }.get(step.skill_id)
        if required is not None:
            state_check = machine.require_state(required)
            if not state_check["allowed"]:
                message = (
                    f"{step.skill_id} blocked by FullGraspStateMachine: "
                    f"{state_check['reason']}"
                )
                self._audit_hardware_event(
                    "llm_step_blocked", step_id=step.step_id,
                    skill_id=step.skill_id, reason=message,
                )
                return self._failed_step(step, message)
        if step.skill_id == "close_gripper" and not self._is_contact_close_step(step):
            return self._failed_step(
                step, "Full grasp close_gripper requires during_contact=true."
            )
        planning_state = {
            "move_to_pregrasp": machine.PREGRASP_PLANNED,
            "approach_grasp": machine.APPROACH_PLANNED,
            "retreat_grasp": machine.RETREAT_PLANNED,
        }.get(step.skill_id)
        if planning_state is not None:
            result = machine.transition(
                planning_state, f"{step.skill_id} approved for single-step planning."
            )
            if not result["allowed"]:
                return self._failed_step(step, result["reason"])
            self._audit_state_transition(step, result)
        self._audit_hardware_event(
            "llm_step_verified", step_id=step.step_id,
            skill_id=step.skill_id, state=machine.current_state(),
        )
        return None

    def _complete_full_grasp_step(
        self, step, machine: FullGraspStateMachine
    ) -> StepExecutionResult | None:
        targets = {
            "detect_object": [machine.OBJECT_DETECTED],
            "build_object_point_cloud": [machine.POINT_CLOUD_BUILT],
            "generate_grasp_pose": [machine.GRASP_GENERATED],
            "move_to_pregrasp": [machine.AT_PREGRASP],
            "approach_grasp": [machine.AT_GRASP_POSE],
            "close_gripper": [machine.GRIPPER_CLOSED],
            "retreat_grasp": [machine.RETREATED],
            "verify_grasp": [machine.GRASP_VERIFIED, machine.DONE],
        }.get(step.skill_id, [])
        for target in targets:
            result = machine.transition(target, f"{step.skill_id} completed.")
            if not result["allowed"]:
                self._abort_full_grasp_state(machine, result["reason"])
                return self._failed_step(step, result["reason"])
            self._audit_state_transition(step, result)
        return None

    def _abort_full_grasp_state(
        self, machine: FullGraspStateMachine, reason: str
    ) -> None:
        if machine.current_state() == machine.ABORTED:
            return
        result = machine.abort(reason)
        self._audit_hardware_event(
            "full_grasp_state_transition",
            previous_state=result.get("previous_state"),
            next_state=result.get("state"),
            reason=reason,
        )

    @staticmethod
    def _resolved_object_id(
        context: Dict[str, Any], machine: FullGraspStateMachine
    ) -> str | None:
        value = context.get("resolved_object_id") or machine.selected_object_id
        if value:
            return str(value)
        selected = context.get("selected_object")
        if isinstance(selected, dict) and selected.get("object_id"):
            return str(selected["object_id"])
        return None

    def _capture_full_grasp_context(
        self,
        step,
        backend_result: Any,
        context: Dict[str, Any],
        machine: FullGraspStateMachine,
    ) -> None:
        if not isinstance(backend_result, dict) or not backend_result.get("success", False):
            return
        output = backend_result.get("output", {})
        output = output if isinstance(output, dict) else {}
        if step.skill_id == "detect_object":
            object_id = output.get("object_id") or backend_result.get("object_id")
            objects = output.get("objects") or backend_result.get("objects")
            if not object_id and isinstance(objects, list) and objects:
                first = objects[0]
                object_id = first.get("object_id") if isinstance(first, dict) else None
            if object_id:
                context["resolved_object_id"] = str(object_id)
                machine.update_context(selected_object_id=str(object_id))
        elif step.skill_id == "build_object_point_cloud":
            path = output.get("point_cloud_path") or backend_result.get("point_cloud_path")
            if path:
                machine.update_context(point_cloud_path=str(path))
        elif step.skill_id == "generate_grasp_pose":
            candidate_id = output.get("candidate_id") or backend_result.get("candidate_id")
            score = output.get("grasp_score", output.get("score"))
            updates: dict[str, Any] = {}
            if candidate_id:
                updates["selected_grasp_candidate_id"] = str(candidate_id)
            if score is not None:
                updates["selected_grasp_score"] = float(score)
            if updates:
                machine.update_context(**updates)

    def _audit_state_transition(self, step, result: dict[str, Any]) -> None:
        self._audit_hardware_event(
            "full_grasp_state_transition",
            step_id=step.step_id,
            skill_id=step.skill_id,
            previous_state=result.get("previous_state"),
            next_state=result.get("state"),
            reason=result.get("reason"),
        )

    def _request_step_confirmation(
        self, context: Dict[str, Any], step, phrase: str, action_type: str
    ) -> bool:
        provider = context.get("confirmation_provider") or self.confirmation_provider
        if callable(provider):
            prompt = (
                "Type EXECUTE to move the REAL UR5:\n> "
                if phrase == "EXECUTE"
                else "Type YES to command this REAL gripper step:\n> "
            )
            try:
                value = provider(prompt)
            except TypeError:
                value = provider(step, phrase)
        else:
            key = "arm_confirmations" if phrase == "EXECUTE" else "gripper_confirmations"
            confirmations = context.get(key, {})
            value = (
                confirmations.pop(str(step.step_id), None)
                if isinstance(confirmations, dict)
                else None
            )
        accepted = value == phrase
        self._audit_hardware_event(
            "human_confirmation_result",
            step_id=step.step_id,
            skill_id=step.skill_id,
            action_type=action_type,
            required_phrase=phrase,
            accepted=accepted,
        )
        if accepted:
            context.setdefault("_accepted_step_confirmations", {})[
                str(step.step_id)
            ] = phrase
        return accepted

    @staticmethod
    def _full_grasp_backend_method(
        step, configured_method: str, supervised_full_grasp: bool
    ) -> str:
        if not supervised_full_grasp:
            return configured_method
        if step.skill_id == "close_gripper" and ExecutionManager._is_contact_close_step(step):
            return "close_for_grasp"
        return {
            "move_to_pregrasp": "move_to_pregrasp",
            "approach_grasp": "approach_grasp",
            "retreat_grasp": "retreat_grasp",
        }.get(step.skill_id, configured_method)

    @staticmethod
    def _is_batch_request(step) -> bool:
        if step.skill_id in {
            "execute_pick_sequence", "execute_full_grasp",
            "execute_full_grasp_sequence", "batch_execute_grasp",
        }:
            return True
        if any(step.parameters.get(key) is True for key in (
            "execute_all", "batch_execute", "confirm_all", "single_confirmation"
        )):
            return True
        text = f"{step.description} {step.parameters}".lower()
        return any(marker in text for marker in (
            "one confirmation", "single confirmation", "execute all steps"
        ))

    @staticmethod
    def _is_full_grasp_plan(plan: RobotPlan) -> bool:
        return any(
            step.skill_id in {
                "move_to_pregrasp", "approach_grasp", "retreat_grasp", "verify_grasp"
            }
            or (
                step.skill_id == "close_gripper"
                and step.parameters.get("during_contact") is True
            )
            for step in plan.steps
        )

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
        self._audit_hardware_event(
            "llm_step_blocked",
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

    @staticmethod
    def _full_grasp_supervised_active(
        mode: HardwareMode | None,
        context: Dict[str, Any],
    ) -> bool:
        requested = context.get("arm_execution_mode")
        if requested == "restricted_real_motion":
            return False
        if requested == "full_grasp_supervised":
            return True
        return mode is not None and (
            mode.mode == "real_full_grasp_supervised"
            or getattr(mode.arm, "execution_mode", None) == "full_grasp_supervised"
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
