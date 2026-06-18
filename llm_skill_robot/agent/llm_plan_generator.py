"""LLM-backed RobotPlan generator.

This module only converts natural language into structured RobotPlan JSON.
It does not execute robot motion, call MoveIt, or generate trajectories.
"""

import json
from typing import Any

from pydantic import ValidationError

from llm_skill_robot.agent.base_plan_generator import BasePlanGenerator
from llm_skill_robot.agent.llm_client import ROBOT_PLAN_JSON_SYSTEM_PROMPT
from llm_skill_robot.agent.mock_plan_generator import PlanGenerationError
from llm_skill_robot.core.plan import RobotPlan


class LLMPlanGenerator(BasePlanGenerator):
    """Generate RobotPlan objects by prompting an LLM for JSON only."""

    def __init__(
        self,
        llm_client,
        default_velocity_scale: float = 0.03,
        default_acceleration_scale: float = 0.03,
    ):
        self.llm_client = llm_client
        self.default_velocity_scale = default_velocity_scale
        self.default_acceleration_scale = default_acceleration_scale

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
        return self._parse_robot_plan(response_text)

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
        return self._parse_robot_plan(response_text)

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
            "safety_rules": {
                "velocity_scale_max": 0.1,
                "velocity_scale_default": self.default_velocity_scale,
                "acceleration_scale_max": 0.1,
                "acceleration_scale_default": self.default_acceleration_scale,
            },
            "gripper_rules": {
                "type": "robotiq_2f_140",
                "max_width_m": 0.14,
                "allowed_skill_ids": [
                    "open_gripper",
                    "close_gripper",
                    "set_gripper_width",
                    "get_gripper_state",
                ],
                "instruction": (
                    "Use open_gripper, close_gripper, or set_gripper_width for "
                    "Robotiq 2F-140 gripper actions. Do not generate raw gripper "
                    "commands, ROS topics, driver calls, or hardware commands."
                ),
            },
            "perception_rules": {
                "allowed_skill_ids": ["detect_object"],
                "instruction": (
                    "Use detect_object before pick_object when an object pose is "
                    "required. detect_object requires a natural-language query "
                    'such as "red cube", "blue cup", or "object on the table". '
                    "Then use pick_object with the same query unless a verified "
                    "object_id is already available. Do not invent object poses or "
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
