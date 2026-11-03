"""LLM client interfaces. Importing this module does not contact a network.

Author: wanghaobo
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class Completion:
    text: str
    reasoning: str = ""
    model: str = "mock"
    usage: Usage = field(default_factory=Usage)


class LLMClient(Protocol):
    """One completion. The runner, not the client, decides whether code may run."""

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        effort: str,
    ) -> Completion:
        """Return the next assistant message for this conversation."""


class MockLLMClient:
    """Deterministic offline client used by stubs and the smoke check.

    `reply` is returned until a user message contains ``Sandbox output:``.
    After that, `final_reply` is returned when set, otherwise a short mock
    closing message. Nothing is sent off the machine.
    """

    def __init__(self, reply: str, final_reply: str | None = None) -> None:
        self.reply = reply
        self.final_reply = final_reply

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        effort: str,
    ) -> Completion:
        del effort
        saw_sandbox = any(
            message.get("role") == "user" and "Sandbox output:" in message.get("content", "")
            for message in messages
        )
        if saw_sandbox:
            text = self.final_reply or "Answer: mock\nConfidence: 0\n"
        else:
            text = self.reply
        return Completion(
            text=text,
            reasoning="offline mock; no model was called",
            model=model or "mock",
            usage=Usage(),
        )


class UnconfiguredLLMClient:
    """Placeholder for a real tool-free backend.

    TODO(model-call): implement this against a local Codex CLI (`codex exec`)
    or another client that can run with shell, web, and file tools disabled.
    Follow the posture recorded in the logs repo's `harness/codex_config.toml`.
    Do not call a paid API from import time, and do not require credentials to
    import this package.
    """

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        effort: str,
    ) -> Completion:
        del messages, model, effort
        raise RuntimeError(
            "No live LLM client is configured. Use MockLLMClient for offline "
            "runs, or implement UnconfiguredLLMClient.complete (TODO model-call)."
        )
