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
from llm_skill_robot.core.plan import RobotPlan
from llm_skill_robot.safety.hardware_mode import HardwareMode, load_hardware_mode


class LLMPlanGenerator(BasePlanGenerator):
    """Generate RobotPlan objects by prompting an LLM for JSON only."""

    def __init__(
        self,
        llm_client,
        default_velocity_scale: float = 0.03,
        default_acceleration_scale: float = 0.03,
        hardware_mode: HardwareMode | None = None,
    ):
        self.llm_client = llm_client
        self.default_velocity_scale = default_velocity_scale
        self.default_acceleration_scale = default_acceleration_scale
        self.hardware_mode = hardware_mode or load_hardware_mode()

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
        self._validate_hardware_safe_plan(plan, available_skills)
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
        self._validate_hardware_safe_plan(plan, available_skills)
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
                    "the runtime, never the LLM, supplies exact YES confirmation."
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
                    "listed in its allowed_values, such as home or observe. For "
                    "object picking, use detect_object followed by pick_object."
                ),
            },
            "arm_safe_pick_rules": {
                "mode": self.hardware_mode.mode,
                "real_arm_execution": "BLOCKED",
                "direct_pick_object": "BLOCKED",
                "raw_ros_commands": "BLOCKED",
                "raw_gripper_driver_commands": "BLOCKED",
                "instruction": (
                    "In real_camera_real_gripper_arm_safe mode, never generate "
                    "pick_object or any real-arm execution step. For 'pick the X', "
                    "generate only a staged perception, candidate review, optional "
                    "confirmed gripper-width test, and plan-only motion preview. "
                    "Human approval is trusted runtime input and must never be "
                    "invented as a plan parameter."
                ),
                "safe_staged_plan": [
                    "move_to_named_target observe (plan-only or RViz/fake only)",
                    "detect_object query=X",
                    "build_object_point_cloud object_id=<resolved>",
                    "generate_grasp_pose object_id=<resolved>",
                    "review_grasp_candidate object_id=<resolved>",
                    "test_gripper_width using reviewed candidate width; exact YES is requested at runtime",
                    "move_to_pregrasp object_id=<resolved> (plan-only preview)",
                    "approach_grasp object_id=<resolved> (plan-only preview)",
                    "retreat_grasp object_id=<resolved> (plan-only preview)",
                ],
                "compatibility_note": (
                    "If a staged schema explicitly uses set_gripper_width, its "
                    "width_m must be resolved from candidate.gripper_width_m at "
                    "runtime and the executor must require exact YES. Prefer "
                    "test_gripper_width so the LLM does not invent a width."
                ),
            },
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
                "Do not generate pick_object in arm-safe hardware mode.",
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
        raw_parameter_names = {
            "ros_command",
            "command_topic",
            "action_name",
            "service_name",
            "driver_command",
        }
        for step in plan.steps:
            if step.skill_id not in available_ids:
                raise PlanGenerationError(
                    f"LLM generated unavailable skill_id: {step.skill_id}"
                )
            if (
                self.hardware_mode.mode == "real_camera_real_gripper_arm_safe"
                and step.skill_id in forbidden_skill_ids
            ):
                raise PlanGenerationError(
                    f"LLM generated hardware-mode-blocked skill_id: {step.skill_id}"
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
