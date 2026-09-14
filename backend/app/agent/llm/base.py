"""Provider-neutral LLM interface.

Two capabilities are all the agent needs:

  * `converse` — a tool-calling turn, used during investigation;
  * `structured` — a JSON-schema-constrained turn, used to produce the
    decision.

Keeping the surface this small means a provider adapter is about eighty
lines, and swapping Gemini for Claude or GPT is a config change. The
`deterministic` provider implements the same interface with rules instead
of a model, so the whole system runs with no network access at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

Role = Literal["user", "assistant", "tool"]


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Turn:
    role: Role
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    # Set when role == "tool".
    tool_call_id: str | None = None
    tool_name: str | None = None


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class LLMError(RuntimeError):
    """Raised when a provider cannot be reached or returns something unusable.

    The graph catches this and degrades to the deterministic planner rather
    than failing the run — a purchasing system should not stop working
    because an inference endpoint is down.
    """


class LLMClient(Protocol):
    name: str
    model: str

    def converse(
        self,
        system: str,
        turns: list[Turn],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse: ...

    def structured(
        self,
        system: str,
        turns: list[Turn],
        schema: dict[str, Any],
    ) -> dict[str, Any]: ...
