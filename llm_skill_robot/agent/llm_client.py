"""LLM client interfaces for RobotPlan JSON generation only."""

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import yaml

from llm_skill_robot.utils import get_config_dir

ROBOT_PLAN_JSON_SYSTEM_PROMPT = (
    "You generate only RobotPlan JSON. Do not generate Python code, ROS commands, "
    "shell commands, raw trajectories, or prose. The JSON must describe task and "
    "steps only."
)


class LLMClientError(RuntimeError):
    """Base error for LLM client failures."""


class LLMClientConfigurationError(LLMClientError):
    """Raised when the LLM client is not configured correctly."""


class BaseLLMClient(ABC):
    """Interface for text-generating LLM clients."""

    @abstractmethod
    def generate_text(self, messages: List[Dict[str, Any]]) -> str:
        """Return a text response for chat-style messages."""


def load_llm_config() -> Dict[str, Any]:
    """Load LLM configuration from config/llm.yaml."""
    config_file = get_config_dir() / "llm.yaml"
    if not config_file.exists():
        return {}
    with open(config_file, "r") as f:
        return yaml.safe_load(f) or {}


class OpenAILLMClient(BaseLLMClient):
    """OpenAI chat client wrapper that returns text only."""

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: Optional[float] = None,
        max_retries: Optional[int] = None,
        timeout_sec: Optional[float] = None,
    ):
        config = load_llm_config().get("llm", {})
        self.model = model or config.get("model", "gpt-4.1-mini")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.temperature = (
            float(temperature)
            if temperature is not None
            else float(config.get("temperature", 0.0))
        )
        self.max_retries = (
            int(max_retries)
            if max_retries is not None
            else int(config.get("max_retries", 2))
        )
        self.timeout_sec = (
            float(timeout_sec)
            if timeout_sec is not None
            else float(config.get("timeout_sec", 30))
        )

        if not self.api_key:
            raise LLMClientConfigurationError(
                "OPENAI_API_KEY is not set. Set it in the environment before using OpenAILLMClient."
            )

    def generate_text(self, messages: List[Dict[str, Any]]) -> str:
        """Generate a text response with the official OpenAI SDK."""
        try:
            from openai import OpenAI
        except ImportError as e:
            raise LLMClientConfigurationError(
                "The optional 'openai' package is not installed. Install with: pip install -e '.[llm]'"
            ) from e

        client = OpenAI(
            api_key=self.api_key,
            max_retries=self.max_retries,
            timeout=self.timeout_sec,
        )
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
        )
        content = response.choices[0].message.content
        return content or ""
