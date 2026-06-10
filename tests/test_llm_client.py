import pytest

from llm_skill_robot.agent.llm_client import (
    BaseLLMClient,
    LLMClientConfigurationError,
    OpenAILLMClient,
    ROBOT_PLAN_JSON_SYSTEM_PROMPT,
    load_llm_config,
)


class DummyLLMClient(BaseLLMClient):
    def generate_text(self, messages: list[dict]) -> str:
        return "ok"


def test_base_llm_client_interface_exists():
    client = DummyLLMClient()

    assert client.generate_text([{"role": "user", "content": "hello"}]) == "ok"


def test_openai_client_missing_api_key_fails_clearly(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(LLMClientConfigurationError, match="OPENAI_API_KEY is not set"):
        OpenAILLMClient()


def test_openai_client_instantiates_with_dummy_api_key_without_network():
    client = OpenAILLMClient(api_key="dummy-api-key")

    assert client.api_key == "dummy-api-key"
    assert client.model


def test_llm_config_loads_default_file():
    config = load_llm_config()

    assert config["llm"]["provider"] == "openai"
    assert config["llm"]["model"] == "gpt-4.1-mini"


def test_robot_plan_prompt_forbids_code_commands_and_trajectories():
    assert "RobotPlan JSON" in ROBOT_PLAN_JSON_SYSTEM_PROMPT
    assert "Python code" in ROBOT_PLAN_JSON_SYSTEM_PROMPT
    assert "ROS commands" in ROBOT_PLAN_JSON_SYSTEM_PROMPT
    assert "raw trajectories" in ROBOT_PLAN_JSON_SYSTEM_PROMPT
