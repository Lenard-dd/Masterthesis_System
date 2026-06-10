"""Convenience natural-language model interfaces for RobotPlan generation.

These wrappers can be constructed before an API key is available. Set the key
later with set_api_key(), or rely on provider-specific environment variables.
"""

import os
from typing import Any, Dict, List, Optional

from llm_skill_robot.agent.llm_client import (
    BaseLLMClient,
    LLMClientConfigurationError,
    OpenAILLMClient,
    load_llm_config,
)


class NaturalLanguageModelInterface(BaseLLMClient):
    """Provider-selectable LLM interface usable by LLMPlanGenerator."""

    def __init__(
        self,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_retries: Optional[int] = None,
        timeout_sec: Optional[float] = None,
    ):
        config = load_llm_config().get("llm", {})
        self.provider = (provider or config.get("provider", "openai")).lower()
        self.api_key = api_key
        self.model = model or config.get("model", "gpt-4.1-mini")
        self.temperature = temperature if temperature is not None else config.get("temperature", 0.0)
        self.max_retries = max_retries if max_retries is not None else config.get("max_retries", 2)
        self.timeout_sec = timeout_sec if timeout_sec is not None else config.get("timeout_sec", 30)

    def set_api_key(self, api_key: str) -> None:
        """Set the API key at runtime without storing it in the repository."""
        self.api_key = api_key

    def generate_text(self, messages: List[Dict[str, Any]]) -> str:
        """Send chat messages to the selected provider and return text."""
        return self._provider_client().generate_text(messages)

    def generate_robot_plan(
        self,
        user_instruction: str,
        available_skills: list,
        context: dict | None = None,
    ):
        """Generate a RobotPlan through LLMPlanGenerator without executing it."""
        from llm_skill_robot.agent.llm_plan_generator import LLMPlanGenerator

        return LLMPlanGenerator(self).generate_plan(
            user_instruction=user_instruction,
            available_skills=available_skills,
            context=context,
        )

    def _provider_client(self) -> BaseLLMClient:
        if self.provider in {"openai", "chatgpt"}:
            return self._openai_client()
        raise LLMClientConfigurationError(
            f"LLM provider '{self.provider}' is not implemented yet. "
            "Supported providers: openai, chatgpt."
        )

    def _provider_api_key(self) -> Optional[str]:
        if self.provider in {"openai", "chatgpt"}:
            return self.api_key or os.getenv("OPENAI_API_KEY")
        env_name = f"{self.provider.upper()}_API_KEY"
        return self.api_key or os.getenv(env_name)

    def _openai_client(self) -> OpenAILLMClient:
        api_key = self._provider_api_key()
        if not api_key:
            raise LLMClientConfigurationError(
                "OpenAI/ChatGPT API key is missing. Call set_api_key(...) or set OPENAI_API_KEY."
            )
        return OpenAILLMClient(
            api_key=api_key,
            model=self.model,
            temperature=self.temperature,
            max_retries=self.max_retries,
            timeout_sec=self.timeout_sec,
        )


class ChatGPTInterface(NaturalLanguageModelInterface):
    """Backward-compatible ChatGPT/OpenAI interface."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_retries: Optional[int] = None,
        timeout_sec: Optional[float] = None,
    ):
        super().__init__(
            provider="openai",
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_retries=max_retries,
            timeout_sec=timeout_sec,
        )
