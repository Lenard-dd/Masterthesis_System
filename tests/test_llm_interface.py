import pytest

from llm_skill_robot.agent.llm_interface import (
    ChatGPTInterface,
    NaturalLanguageModelInterface,
)
from llm_skill_robot.agent.llm_client import BaseLLMClient, LLMClientConfigurationError


def test_llm_interface_implements_base_llm_client():
    interface = ChatGPTInterface()

    assert isinstance(interface, BaseLLMClient)


def test_llm_interface_can_be_created_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    interface = ChatGPTInterface()

    assert interface.api_key is None
    assert interface.model


def test_llm_interface_set_api_key_later():
    interface = ChatGPTInterface()

    interface.set_api_key("dummy-key")

    assert interface.api_key == "dummy-key"


def test_llm_interface_missing_key_fails_clearly(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    interface = ChatGPTInterface()

    with pytest.raises(LLMClientConfigurationError, match="OpenAI/ChatGPT API key is missing"):
        interface.generate_text([{"role": "user", "content": "hello"}])


def test_natural_language_model_interface_selects_openai_provider():
    interface = NaturalLanguageModelInterface(provider="openai", api_key="dummy-key")

    assert isinstance(interface, BaseLLMClient)
    assert interface.provider == "openai"
    assert interface.api_key == "dummy-key"


def test_natural_language_model_interface_rejects_unsupported_provider():
    interface = NaturalLanguageModelInterface(provider="anthropic", api_key="dummy-key")

    with pytest.raises(LLMClientConfigurationError, match="not implemented yet"):
        interface.generate_text([{"role": "user", "content": "hello"}])
