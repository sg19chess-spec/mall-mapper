"""Shared Claude API wrapper used by all five agents for the reasoning steps
that are genuinely reasoning (extraction from messy text, review judgment) --
never for the mechanical steps (HTTP requests, geometry math, rule checks),
which stay plain Python in agents/tools/.
"""
from __future__ import annotations

import json
import os
from typing import Any

_MODEL = os.environ.get("MALL_MAPPER_MODEL", "claude-sonnet-5")


class Agent:
    """Base class: every agent gets a name (for logging/audit) and an
    ask_claude() helper. Subclasses implement run()."""

    name: str = "agent"

    def __init__(self) -> None:
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic

            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                return None
            self._client = anthropic.Anthropic(api_key=api_key)
        return self._client

    def ask_claude(self, system: str, prompt: str, max_tokens: int = 1024) -> str:
        """Returns Claude's text response, or raises if no API key is configured.
        Callers in dev-mode paths should catch AgentUnavailable and fall back to
        a deterministic heuristic instead of failing the whole run."""
        client = self._get_client()
        if client is None:
            raise AgentUnavailable("ANTHROPIC_API_KEY not set")
        resp = client.messages.create(
            model=_MODEL,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in resp.content if block.type == "text")

    def ask_claude_json(self, system: str, prompt: str, max_tokens: int = 1024) -> Any:
        text = self.ask_claude(system, prompt, max_tokens=max_tokens)
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1:
            start, end = text.find("["), text.rfind("]")
        if start == -1 or end == -1:
            raise ValueError(f"No JSON found in Claude response: {text[:200]}")
        return json.loads(text[start : end + 1])


class AgentUnavailable(RuntimeError):
    """Raised when a Claude-backed reasoning step can't run (no API key)."""
