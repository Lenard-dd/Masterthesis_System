from llm_skill_robot.agent.base_plan_generator import BasePlanGenerator
from llm_skill_robot.core.plan import PlanStep, RobotPlan


class PlanGenerationError(ValueError):
    """Raised when a user instruction cannot be mapped to known skills."""


class MockPlanGenerator(BasePlanGenerator):
    """Rule-based plan generator used before integrating a real LLM provider."""

    def generate_plan(
        self,
        user_instruction: str,
        available_skills: list,
        context: dict | None = None,
    ) -> RobotPlan:
        del available_skills, context
        instruction = user_instruction.strip()
        normalized_instruction = instruction.lower()
        step_specs = []

        for target in ("home", "observe"):
            index = normalized_instruction.find(target)
            if index >= 0:
                step_specs.append((index, "motion", target))

        if "open gripper" in normalized_instruction or "open the gripper" in normalized_instruction:
            step_specs.append(
                (
                    self._first_index(normalized_instruction, "open gripper", "open the gripper"),
                    "open_gripper",
                    None,
                )
            )

        if "close gripper" in normalized_instruction or "close the gripper" in normalized_instruction:
            step_specs.append(
                (
                    self._first_index(normalized_instruction, "close gripper", "close the gripper"),
                    "close_gripper",
                    None,
                )
            )

        steps = [
            self._step_from_spec(step_id=index + 1, action=action, target=target)
            for index, (_, action, target) in enumerate(
                sorted(step_specs, key=lambda item: item[0])
            )
        ]

        if not steps:
            raise PlanGenerationError(
                f"Could not generate a RobotPlan from instruction: {user_instruction!r}"
            )

        return RobotPlan(task=user_instruction, steps=steps)

    @classmethod
    def _step_from_spec(cls, step_id: int, action: str, target: str | None) -> PlanStep:
        if action == "motion" and target is not None:
            return cls._motion_step(step_id=step_id, target=target)
        if action == "open_gripper":
            return PlanStep(
                step_id=str(step_id),
                skill_id="open_gripper",
                description="Open the gripper",
                parameters={},
            )
        if action == "close_gripper":
            return PlanStep(
                step_id=str(step_id),
                skill_id="close_gripper",
                description="Close the gripper",
                parameters={},
            )
        raise PlanGenerationError(f"Unsupported mock plan action: {action}")

    @staticmethod
    def _first_index(text: str, *needles: str) -> int:
        indexes = [text.find(needle) for needle in needles if text.find(needle) >= 0]
        return min(indexes)

    @staticmethod
    def _motion_step(step_id: int, target: str) -> PlanStep:
        return PlanStep(
            step_id=str(step_id),
            skill_id="move_to_named_target",
            description=f"Move to named target {target}",
            parameters={
                "target": target,
                "velocity_scale": 0.03,
                "acceleration_scale": 0.03,
            },
        )
