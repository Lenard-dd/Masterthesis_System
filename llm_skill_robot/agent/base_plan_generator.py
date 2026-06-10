from abc import ABC, abstractmethod

from llm_skill_robot.core.plan import RobotPlan


class BasePlanGenerator(ABC):
    """Interface for converting user instructions into structured robot plans."""

    @abstractmethod
    def generate_plan(
        self,
        user_instruction: str,
        available_skills: list,
        context: dict | None = None,
    ) -> RobotPlan:
        """Return a RobotPlan for the given natural-language instruction."""
