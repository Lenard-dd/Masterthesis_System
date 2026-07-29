"""Registry for task-level composite Agent skills."""

from __future__ import annotations

from typing import Callable

from llm_skill_robot.agent.composite_skills.base import BaseCompositeSkill, CompositeSkillCall
from llm_skill_robot.agent.composite_skills.safe_pick import SafePickObjectSkill


class CompositeSkillRegistry:
    """Create stateful composite skills without exposing backend methods to LLMs."""

    def __init__(self) -> None:
        self._factories: dict[str, Callable[[], BaseCompositeSkill]] = {
            SafePickObjectSkill.skill_name: SafePickObjectSkill,
        }

    def list_skills(self) -> list[dict[str, str]]:
        return [
            {
                "skill_name": name,
                "description": self._factories[name]().description,
            }
            for name in sorted(self._factories)
        ]

    def create(self, call: CompositeSkillCall) -> BaseCompositeSkill:
        factory = self._factories.get(call.skill_name)
        if factory is None:
            raise ValueError(f"Unknown composite skill: {call.skill_name}")
        skill = factory()
        skill.start(call.arguments)
        return skill
