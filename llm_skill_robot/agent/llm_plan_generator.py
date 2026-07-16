"""LLM-backed RobotPlan generator.

This module only converts natural language into structured RobotPlan JSON.
It does not execute robot motion, call MoveIt, or generate trajectories.
"""

import json
from dataclasses import asdict
from typing import Any

from pydantic import ValidationError

from llm_skill_robot.agent.base_plan_generator import BasePlanGenerator
from llm_skill_robot.agent.llm_client import ROBOT_PLAN_JSON_SYSTEM_PROMPT
from llm_skill_robot.agent.mock_plan_generator import PlanGenerationError
from llm_skill_robot.core.plan import PlanStep, RobotPlan
from llm_skill_robot.safety.hardware_mode import HardwareMode, load_hardware_mode


class LLMPlanGenerator(BasePlanGenerator):
    """Generate RobotPlan objects by prompting an LLM for JSON only."""

    def __init__(
        self,
        llm_client,
        default_velocity_scale: float = 0.03,
        default_acceleration_scale: float = 0.03,
        hardware_mode: HardwareMode | None = None,
        audit_logger: Any = None,
        require_observe_before_detection: bool = False,
    ):
        self.llm_client = llm_client
        self.default_velocity_scale = default_velocity_scale
        self.default_acceleration_scale = default_acceleration_scale
        self.hardware_mode = hardware_mode or load_hardware_mode()
        self.audit_logger = audit_logger
        self.require_observe_before_detection = require_observe_before_detection

    def generate_plan(
        self,
        user_instruction: str,
        available_skills: list,
        context: dict | None = None,
    ) -> RobotPlan:
        messages = self._build_messages(
            user_instruction=user_instruction,
            available_skills=available_skills,
            context=context,
        )
        response_text = self.llm_client.generate_text(messages)
        plan = self._parse_robot_plan(response_text)
        if self.require_observe_before_detection:
            plan = self._ensure_observe_before_detection(plan)
        self._validate_hardware_safe_plan(plan, available_skills)
        if self._full_grasp_supervised_active() and self._is_full_grasp_plan(plan):
            self._audit(
                "llm_full_grasp_plan_generated",
                task=plan.task,
                skill_ids=[step.skill_id for step in plan.steps],
            )
        return plan

    def repair_plan(
        self,
        user_instruction: str,
        available_skills: list,
        previous_plan_or_text,
        verification_result,
        context: dict | None = None,
    ) -> RobotPlan:
        messages = self._build_messages(
            user_instruction=user_instruction,
            available_skills=available_skills,
            context=context,
            previous_plan_or_text=previous_plan_or_text,
            verification_result=verification_result,
        )
        response_text = self.llm_client.generate_text(messages)
        plan = self._parse_robot_plan(response_text)
        if self.require_observe_before_detection:
            plan = self._ensure_observe_before_detection(plan)
        self._validate_hardware_safe_plan(plan, available_skills)
        if self._full_grasp_supervised_active() and self._is_full_grasp_plan(plan):
            self._audit(
                "llm_full_grasp_plan_generated",
                task=plan.task,
                skill_ids=[step.skill_id for step in plan.steps],
                repaired=True,
            )
        return plan

    def _build_messages(
        self,
        user_instruction: str,
        available_skills: list,
        context: dict | None = None,
        previous_plan_or_text=None,
        verification_result=None,
    ) -> list[dict]:
        prompt = {
            "user_instruction": user_instruction,
            "available_skills": [
                self._skill_to_prompt_dict(skill) for skill in available_skills
            ],
            "context": context or {},
            "hardware_mode": asdict(self.hardware_mode),
            "safety_rules": {
                "velocity_scale_max": 0.1,
                "velocity_scale_default": self.default_velocity_scale,
                "acceleration_scale_max": 0.1,
                "acceleration_scale_default": self.default_acceleration_scale,
            },
            "gripper_rules": {
                "type": "robotiq_2f_140",
                "max_width_m": 0.14,
                "min_width_m": self.hardware_mode.gripper.min_width_m,
                "max_speed_mps": self.hardware_mode.gripper.max_speed_mps,
                "max_force_N": self.hardware_mode.gripper.max_force_N,
                "safe_close_width_m": self.hardware_mode.gripper.safe_close_width_m,
                "allowed_skill_ids": [
                    "open_gripper",
                    "close_gripper",
                    "set_gripper_width",
                    "test_gripper_width",
                    "get_gripper_state",
                ],
                "instruction": (
                    "Use open_gripper, close_gripper, or set_gripper_width for "
                    "Robotiq 2F-140 gripper actions. Do not generate raw gripper "
                    "commands, ROS topics, driver calls, or hardware commands."
                    " Prefer test_gripper_width for a reviewed GraspGenX width; "
                    "the runtime, never the LLM, supplies exact YES confirmation. "
                    "In restricted_real_motion mode, do not close the gripper on "
                    "an object or during contact."
                ),
            },
            "perception_rules": {
                "allowed_skill_ids": [
                    "detect_object",
                    "build_object_point_cloud",
                    "generate_grasp_pose",
                    "review_grasp_candidate",
                ],
                "instruction": (
                    (
                        "Before every detect_object step, first add a separate "
                        "move_to_named_target step with target_name=observe. "
                        if self.require_observe_before_detection else ""
                    )
                    +
                    "Legacy planners used the phrases 'detect_object before pick_object' "
                    "and 'pick_object with the same query'; do not follow those direct-pick "
                    "rules in arm-safe hardware mode. detect_object requires a natural-language query "
                    'such as "red cube", "blue cup", or "object on the table". '
                    "Continue with point-cloud building, GraspGenX generation, review, "
                    "and plan-only previews. Do not invent object poses or "
                    "create raw camera/SAM3 commands."
                ),
            },
            "object_manipulation_rules": {
                "instruction": (
                    "Do not use move_to_named_target for pickup or place actions. "
                    "move_to_named_target may only use named targets explicitly "
                    "listed in its allowed_values, such as home or observe. In "
                    "restricted_real_motion mode, object picking must be staged "
                    "and must not use pick_object."
                ),
            },
            "arm_safe_pick_rules": self._arm_safe_pick_rules(),
            "output_schema": {
                "task": "string",
                "steps": [
                    {
                        "step_id": 1,
                        "skill_id": "string",
                        "description": "string",
                        "parameters": {},
                    }
                ],
            },
            "output_rules": [
                "Output only JSON.",
                "Do not output markdown.",
                "Do not output Python code.",
                "Do not output ROS commands.",
                "Do not output raw trajectories.",
                "Do not generate raw gripper commands.",
                "Do not generate pick_object in restricted-arm hardware mode.",
                "Do not generate approach_grasp, lift_object, or close_gripper on an object in restricted_real_motion mode.",
                "Do not claim or encode human confirmation in plan parameters.",
                "Use only skill_id values from available_skills.",
            ],
        }
        if previous_plan_or_text is not None or verification_result is not None:
            prompt["repair"] = {
                "previous_plan_or_text": self._serializable(previous_plan_or_text),
                "verification_result": self._serializable(verification_result),
                "instruction": "Fix the plan so it satisfies the verification errors. Output only corrected RobotPlan JSON.",
            }

        return [
            {"role": "system", "content": ROBOT_PLAN_JSON_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(prompt, indent=2, sort_keys=True)},
        ]

    @staticmethod
    def _skill_to_prompt_dict(skill) -> dict:
        if isinstance(skill, str):
            return {
                "skill_id": skill,
                "description": "",
                "parameters": [],
            }
        parameters = []
        for parameter in getattr(skill, "parameters", []) or []:
            parameters.append(
                {
                    "name": getattr(parameter, "name", ""),
                    "type": getattr(parameter, "type", ""),
                    "required": getattr(parameter, "required", False),
                    "default": getattr(parameter, "default", None),
                    "allowed_values": getattr(parameter, "allowed_values", None),
                    "min_value": getattr(parameter, "min_value", None),
                    "max_value": getattr(parameter, "max_value", None),
                }
            )
        return {
            "skill_id": getattr(skill, "skill_id", ""),
            "description": getattr(skill, "description", ""),
            "parameters": parameters,
        }

    @classmethod
    def _parse_robot_plan(cls, text: str) -> RobotPlan:
        json_text = cls._strip_markdown_fence(text)
        try:
            data = json.loads(json_text)
        except json.JSONDecodeError as e:
            raise PlanGenerationError(f"LLM did not return valid RobotPlan JSON: {e}") from e

        if isinstance(data, dict) and isinstance(data.get("steps"), list):
            data = dict(data)
            data["steps"] = [cls._normalize_step(step) for step in data["steps"]]

        try:
            return RobotPlan.model_validate(data)
        except ValidationError as e:
            raise PlanGenerationError(f"LLM RobotPlan JSON failed schema validation: {e}") from e

    @staticmethod
    def _normalize_step(step: Any) -> Any:
        if not isinstance(step, dict):
            return step
        normalized = dict(step)
        if "step_id" in normalized:
            normalized["step_id"] = str(normalized["step_id"])
        return normalized

    @staticmethod
    def _strip_markdown_fence(text: str) -> str:
        stripped = text.strip()
        if not stripped.startswith("```"):
            return stripped

        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()

    @staticmethod
    def _serializable(value):
        if hasattr(value, "model_dump"):
            return value.model_dump()
        if isinstance(value, RobotPlan):
            return value.model_dump()
        return value

    def _validate_hardware_safe_plan(self, plan: RobotPlan, available_skills: list) -> None:
        """Reject unsafe LLM output independently of prompt compliance."""
        available_ids = {
            skill if isinstance(skill, str) else getattr(skill, "skill_id", "")
            for skill in available_skills
        }
        forbidden_skill_ids = {
            "pick_object",
            "execute_trajectory",
            "execute_arm_motion",
            "move_real_arm",
            "send_ros_command",
            "raw_gripper_command",
        }
        restricted_forbidden_skill_ids = forbidden_skill_ids | {
            "approach_grasp",
            "lift_object",
            "execute_pick_sequence",
        }
        full_grasp_forbidden_skill_ids = forbidden_skill_ids | {
            "execute_pick_sequence",
            "execute_full_grasp",
            "execute_full_grasp_sequence",
            "batch_execute_grasp",
        }
        raw_parameter_names = {
            "ros_command",
            "command_topic",
            "action_name",
            "service_name",
            "driver_command",
            "trajectory",
            "joint_trajectory",
            "controller_command",
        }
        hardware_restricted = self._restricted_real_motion_active()
        full_grasp_supervised = self._full_grasp_supervised_active()
        for step in plan.steps:
            hardware_arm_safe = self.hardware_mode.mode in {
                "real_camera_real_gripper_arm_safe",
                "real_camera_real_gripper_restricted_arm",
                "real_full_grasp_supervised",
            }
            blocked_ids = (
                restricted_forbidden_skill_ids
                if hardware_restricted
                else full_grasp_forbidden_skill_ids
                if full_grasp_supervised
                else forbidden_skill_ids
            )
            if hardware_arm_safe and step.skill_id in blocked_ids:
                raise PlanGenerationError(
                    f"LLM generated hardware-mode-blocked skill_id: {step.skill_id}"
                )
            if step.skill_id not in available_ids:
                raise PlanGenerationError(
                    f"LLM generated unavailable skill_id: {step.skill_id}"
                )
            unsafe_parameters = raw_parameter_names.intersection(step.parameters)
            if unsafe_parameters:
                raise PlanGenerationError(
                    "LLM generated raw ROS/driver parameters: "
                    + ", ".join(sorted(unsafe_parameters))
                )
            if any(key in step.parameters for key in ("confirmed", "human_confirmed")):
                raise PlanGenerationError(
                    "LLM must not generate trusted human confirmation parameters."
                )
            unsafe_text = json.dumps(
                {"description": step.description, "parameters": step.parameters}
            ).lower()
            if hardware_restricted and step.skill_id == "close_gripper":
                if any(
                    marker in unsafe_text
                    for marker in ("during_contact", "contact", "on object", "object")
                ):
                    raise PlanGenerationError(
                        "LLM generated hardware-mode-blocked close_gripper during contact."
                    )
            if full_grasp_supervised and step.skill_id == "close_gripper":
                if step.parameters.get("during_contact") is not True:
                    raise PlanGenerationError(
                        "Full supervised grasp contact close requires during_contact=true."
                    )
            if full_grasp_supervised and self._is_batch_request(step):
                raise PlanGenerationError(
                    "LLM requested batch execution or one confirmation for multiple real steps."
                )
            if any(
                marker in unsafe_text
                for marker in (
                    "ros2 topic pub",
                    "ros2 action send_goal",
                    "ros2 service call",
                    "/_action/send_goal",
                    "control_msgs/action/grippercommand",
                )
            ):
                raise PlanGenerationError("LLM generated a raw ROS or driver command.")

        if full_grasp_supervised and self._is_full_grasp_plan(plan):
            self._validate_full_grasp_sequence(plan)

    @staticmethod
    def _is_observe_step(step: Any) -> bool:
        return (
            step.skill_id == "move_to_named_target"
            and (step.parameters.get("target_name") or step.parameters.get("target"))
            == "observe"
        )

    def _ensure_observe_before_detection(self, plan: RobotPlan) -> RobotPlan:
        """Deterministically insert an observe motion before every recognition step."""
        steps: list[PlanStep] = []
        for step in plan.steps:
            if step.skill_id == "detect_object" and (
                not steps or not self._is_observe_step(steps[-1])
            ):
                steps.append(PlanStep(
                    step_id="0",
                    skill_id="move_to_named_target",
                    description="Move to the observe pose before object recognition",
                    parameters={
                        "target_name": "observe",
                        "velocity_scale": self.default_velocity_scale,
                        "acceleration_scale": self.default_acceleration_scale,
                    },
                ))
            steps.append(step)
        for index, step in enumerate(steps, start=1):
            step.step_id = str(index)
        return RobotPlan(task=plan.task, steps=steps)

    def _arm_safe_pick_rules(self) -> dict[str, Any]:
        if self._full_grasp_supervised_active():
            return {
                "mode": "full_grasp_supervised",
                "real_arm_execution": "STATE_MACHINE_AND_PER_STEP_CONFIRMATION_ONLY",
                "direct_pick_object": "BLOCKED",
                "raw_ros_commands": "BLOCKED",
                "raw_gripper_driver_commands": "BLOCKED",
                "batch_confirmation": "BLOCKED",
                "compatibility_note": (
                    "This replaces the earlier real_camera_real_gripper_restricted_arm "
                    "pick policy; restricted_real_motion still blocks approach_grasp."
                ),
                "instruction": (
                    "For 'pick the X', generate exactly the staged full-grasp skill "
                    "sequence below. The runtime FullGraspStateMachine, never the LLM, "
                    "controls execution order. Never include confirmation values, raw "
                    "ROS commands, trajectories, controller calls, retries, or a batch "
                    "execution request. Every arm step asks EXECUTE separately and the "
                    "contact close asks YES separately at runtime."
                ),
                "required_skill_sequence": [
                    *(
                        ['move_to_named_target target_name="observe"']
                        if self.require_observe_before_detection else []
                    ),
                    'detect_object query="X"',
                    "build_object_point_cloud object_id=<resolved>",
                    "generate_grasp_pose object_id=<resolved>",
                    "review_grasp_candidate object_id=<resolved>",
                    "open_gripper",
                    "move_to_pregrasp object_id=<resolved>",
                    "approach_grasp object_id=<resolved>",
                    "close_gripper object_id=<resolved> during_contact=true",
                    "retreat_grasp object_id=<resolved>",
                    "verify_grasp object_id=<resolved>",
                ],
            }
        return {
            "mode": self.hardware_mode.mode,
            "real_arm_execution": "RESTRICTED_STAGED_ONLY",
            "direct_pick_object": "BLOCKED",
            "approach_grasp": "BLOCKED",
            "lift_object": "BLOCKED",
            "close_gripper_on_object": "BLOCKED",
            "raw_ros_commands": "BLOCKED",
            "raw_gripper_driver_commands": "BLOCKED",
            "instruction": (
                "In real_camera_real_gripper_restricted_arm mode, never generate "
                "pick_object, approach_grasp, lift_object, close_gripper on an "
                "object, or any autonomous pick sequence."
            ),
        }

    @staticmethod
    def _is_batch_request(step: Any) -> bool:
        if step.skill_id in {
            "execute_pick_sequence",
            "execute_full_grasp",
            "execute_full_grasp_sequence",
            "batch_execute_grasp",
        }:
            return True
        if any(
            step.parameters.get(key) is True
            for key in ("execute_all", "batch_execute", "confirm_all", "single_confirmation")
        ):
            return True
        text = f"{step.description} {step.parameters}".lower()
        return any(
            marker in text
            for marker in ("one confirmation", "single confirmation", "execute all steps")
        )

    @staticmethod
    def _is_full_grasp_plan(plan: RobotPlan) -> bool:
        return any(
            step.skill_id in {"approach_grasp", "retreat_grasp", "verify_grasp"}
            or (
                step.skill_id == "close_gripper"
                and step.parameters.get("during_contact") is True
            )
            for step in plan.steps
        )

    def _validate_full_grasp_sequence(self, plan: RobotPlan) -> None:
        required = [
            *(["move_to_named_target"] if self.require_observe_before_detection else []),
            "detect_object",
            "build_object_point_cloud",
            "generate_grasp_pose",
            "review_grasp_candidate",
            "open_gripper",
            "move_to_pregrasp",
            "approach_grasp",
            "close_gripper",
            "retreat_grasp",
            "verify_grasp",
        ]
        actual = [step.skill_id for step in plan.steps]
        if actual != required:
            if len(actual) == 1:
                raise PlanGenerationError(
                    "LLM generated hardware-mode-blocked skill_id: "
                    f"{actual[0]}; the full-grasp state-machine prerequisites are missing."
                )
            raise PlanGenerationError(
                "Full supervised grasp plan must use the exact state-machine skill "
                f"sequence: {', '.join(required)}."
            )
        close_step = plan.steps[8 if self.require_observe_before_detection else 7]
        if close_step.parameters.get("during_contact") is not True:
            raise PlanGenerationError(
                "Full supervised grasp close_gripper requires during_contact=true."
            )

    def _audit(self, event: str, **fields: Any) -> None:
        if self.audit_logger and hasattr(self.audit_logger, "log_hardware_safety_event"):
            self.audit_logger.log_hardware_safety_event(event, **fields)

    def _restricted_real_motion_active(self) -> bool:
        return (
            self.hardware_mode.mode == "real_camera_real_gripper_restricted_arm"
            or getattr(self.hardware_mode.arm, "execution_mode", None)
            == "restricted_real_motion"
        )

    def _full_grasp_supervised_active(self) -> bool:
        """Keep unsafe direct LLM steps blocked until the runtime state machine gates them."""
        return (
            self.hardware_mode.mode == "real_full_grasp_supervised"
            or getattr(self.hardware_mode.arm, "execution_mode", None)
            == "full_grasp_supervised"
        )
