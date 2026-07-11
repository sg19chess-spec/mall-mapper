"""Tests for the provider-agnostic LLM wrapper in agents/base.py: Anthropic
and OpenAI both work independently, ask_llm() picks the right provider
(Anthropic preferred when both are configured), and JSON extraction works
for both. Fully mocked -- no real API keys or network calls.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.agents.base import Agent, AgentUnavailable


def make_anthropic_response(text: str):
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])


def make_openai_response(text: str):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def test_ask_claude_raises_when_no_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    agent = Agent()
    with pytest.raises(AgentUnavailable):
        agent.ask_claude("system", "prompt")


def test_ask_openai_raises_when_no_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    agent = Agent()
    with pytest.raises(AgentUnavailable):
        agent.ask_openai("system", "prompt")


def test_ask_claude_returns_text(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    agent = Agent()
    mock_client = MagicMock()
    mock_client.messages.create.return_value = make_anthropic_response("Nike is on Level 2.")
    agent._anthropic_client = mock_client

    result = agent.ask_claude("You are a helpful assistant.", "Where is Nike?")

    assert result == "Nike is on Level 2."
    mock_client.messages.create.assert_called_once()


def test_ask_openai_returns_text(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    agent = Agent()
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = make_openai_response("Nike is on Level 2.")
    agent._openai_client = mock_client

    result = agent.ask_openai("You are a helpful assistant.", "Where is Nike?")

    assert result == "Nike is on Level 2."
    mock_client.chat.completions.create.assert_called_once()


def test_ask_llm_prefers_anthropic_when_both_configured(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    agent = Agent()
    mock_anthropic = MagicMock()
    mock_anthropic.messages.create.return_value = make_anthropic_response("from claude")
    mock_openai = MagicMock()
    agent._anthropic_client = mock_anthropic
    agent._openai_client = mock_openai

    result = agent.ask_llm("system", "prompt")

    assert result == "from claude"
    mock_anthropic.messages.create.assert_called_once()
    mock_openai.chat.completions.create.assert_not_called()


def test_ask_llm_falls_back_to_openai_when_only_that_is_configured(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    agent = Agent()
    mock_openai = MagicMock()
    mock_openai.chat.completions.create.return_value = make_openai_response("from openai")
    agent._openai_client = mock_openai

    result = agent.ask_llm("system", "prompt")

    assert result == "from openai"
    mock_openai.chat.completions.create.assert_called_once()


def test_ask_llm_raises_when_neither_configured(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    agent = Agent()
    with pytest.raises(AgentUnavailable):
        agent.ask_llm("system", "prompt")


def test_ask_claude_json_extracts_object(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    agent = Agent()
    mock_client = MagicMock()
    mock_client.messages.create.return_value = make_anthropic_response(
        'Here is the answer: {"floor": 2, "store": "Nike"} -- hope that helps.'
    )
    agent._anthropic_client = mock_client

    result = agent.ask_claude_json("system", "prompt")

    assert result == {"floor": 2, "store": "Nike"}


def test_ask_openai_json_extracts_array(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    agent = Agent()
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = make_openai_response(
        'Sure, here you go: [{"store": "Apple"}, {"store": "Nike"}]'
    )
    agent._openai_client = mock_client

    result = agent.ask_openai_json("system", "prompt")

    assert result == [{"store": "Apple"}, {"store": "Nike"}]


def test_ask_llm_json_raises_on_unparseable_response(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    agent = Agent()
    mock_client = MagicMock()
    mock_client.messages.create.return_value = make_anthropic_response("no json here at all")
    agent._anthropic_client = mock_client

    with pytest.raises(ValueError):
        agent.ask_llm_json("system", "prompt")
