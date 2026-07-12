"""Shared LLM wrapper used by all five agents for the reasoning steps that
are genuinely reasoning (extraction from messy text, review judgment) --
never for the mechanical steps (HTTP requests, geometry math, rule checks),
which stay plain Python in agents/tools/.

Two providers are supported, Anthropic (Claude) and OpenAI, each usable
directly (ask_claude/ask_openai) or via the provider-agnostic ask_llm(),
which picks whichever is configured -- Anthropic first if both are set,
since it's this project's primary provider.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any

_ANTHROPIC_MODEL = os.environ.get("MALL_MAPPER_MODEL", "claude-sonnet-5")
_OPENAI_MODEL = os.environ.get("MALL_MAPPER_OPENAI_MODEL", "gpt-4o-mini")


class Agent:
    """Base class: every agent gets a name (for logging/audit) and LLM
    helpers. Subclasses implement run()."""

    name: str = "agent"

    def __init__(self) -> None:
        self._anthropic_client = None
        self._openai_client = None

    def _get_anthropic_client(self):
        if self._anthropic_client is None:
            import anthropic

            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                return None
            self._anthropic_client = anthropic.Anthropic(api_key=api_key)
        return self._anthropic_client

    def _get_openai_client(self):
        if self._openai_client is None:
            import openai

            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                return None
            self._openai_client = openai.OpenAI(api_key=api_key)
        return self._openai_client

    def ask_claude(self, system: str, prompt: str, max_tokens: int = 1024) -> str:
        """Returns Claude's text response, or raises if no API key is configured.
        Callers in dev-mode paths should catch AgentUnavailable and fall back to
        a deterministic heuristic instead of failing the whole run."""
        client = self._get_anthropic_client()
        if client is None:
            raise AgentUnavailable("ANTHROPIC_API_KEY not set")
        resp = client.messages.create(
            model=_ANTHROPIC_MODEL,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in resp.content if block.type == "text")

    def ask_openai(self, system: str, prompt: str, max_tokens: int = 1024) -> str:
        """Returns OpenAI's text response, or raises if no API key is configured."""
        client = self._get_openai_client()
        if client is None:
            raise AgentUnavailable("OPENAI_API_KEY not set")
        resp = client.chat.completions.create(
            model=_OPENAI_MODEL,
            max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content or ""

    def ask_llm(self, system: str, prompt: str, max_tokens: int = 1024) -> str:
        """Provider-agnostic entry point: uses whichever provider is
        configured (Anthropic preferred if both are set), raising
        AgentUnavailable if neither ANTHROPIC_API_KEY nor OPENAI_API_KEY is
        present. Agents that don't care which provider answers should call
        this instead of ask_claude/ask_openai directly."""
        if os.environ.get("ANTHROPIC_API_KEY"):
            return self.ask_claude(system, prompt, max_tokens=max_tokens)
        if os.environ.get("OPENAI_API_KEY"):
            return self.ask_openai(system, prompt, max_tokens=max_tokens)
        raise AgentUnavailable("Neither ANTHROPIC_API_KEY nor OPENAI_API_KEY is set")

    def ask_claude_json(self, system: str, prompt: str, max_tokens: int = 1024) -> Any:
        return _extract_json(self.ask_claude(system, prompt, max_tokens=max_tokens))

    def ask_openai_json(self, system: str, prompt: str, max_tokens: int = 1024) -> Any:
        return _extract_json(self.ask_openai(system, prompt, max_tokens=max_tokens))

    def ask_llm_json(self, system: str, prompt: str, max_tokens: int = 1024) -> Any:
        return _extract_json(self.ask_llm(system, prompt, max_tokens=max_tokens))

    def try_llm_json(self, system: str, prompt: str, max_tokens: int = 1024) -> Any | None:
        """Best-effort LLM call for steps that have a deterministic fallback.
        Returns the parsed JSON on success, or None if no provider is
        configured (AgentUnavailable) or the call/parse failed -- so a
        transient LLM outage degrades to the heuristic path instead of
        failing the whole pipeline run. The caller decides what to do with
        None. Errors are logged to stderr so a silent fallback is
        diagnosable rather than invisible."""
        try:
            return self.ask_llm_json(system, prompt, max_tokens=max_tokens)
        except AgentUnavailable:
            return None
        except Exception as exc:  # network blip, malformed JSON, rate limit, etc.
            print(f"[{self.name}] LLM call failed, falling back to heuristic: {type(exc).__name__}: {exc}", file=sys.stderr)
            return None

    def llm_available(self) -> bool:
        return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY"))


def _extract_json(text: str) -> Any:
    """Pulls the first JSON object/array out of a prose LLM response. Picks
    whichever bracket type opens *first* in the text -- a top-level array
    of objects (e.g. "[{"a":1},{"b":2}]") has both { and } present too, so
    naively preferring braces would slice out an invalid fragment
    ("{"a":1}, {"b":2}" with no wrapping brackets)."""
    brace_start, bracket_start = text.find("{"), text.find("[")
    if bracket_start != -1 and (brace_start == -1 or bracket_start < brace_start):
        start, end = bracket_start, text.rfind("]")
    else:
        start, end = brace_start, text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON found in response: {text[:200]}")
    return json.loads(text[start : end + 1])


class AgentUnavailable(RuntimeError):
    """Raised when an LLM-backed reasoning step can't run (no API key)."""
