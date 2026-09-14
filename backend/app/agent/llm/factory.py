"""Provider selection, with a safety net.

`get_client` honours configuration. `resilient_client` wraps whatever it
returns so that a provider failure degrades to the deterministic planner
instead of failing the run — the same principle as a circuit breaker in
front of any other third-party dependency.
"""

from __future__ import annotations

import logging
from typing import Any

from app.agent.llm.base import LLMError, LLMResponse, Turn
from app.agent.llm.deterministic import DeterministicClient

logger = logging.getLogger(__name__)


def get_client(provider: str, model: str, api_key: str, temperature: float = 0.0,
               timeout: int = 60, max_retries: int = 2):
    if provider == "deterministic" or not api_key:
        if provider != "deterministic":
            logger.warning(
                "Provider %s selected but no API key present; using the deterministic planner.",
                provider,
            )
        return DeterministicClient()

    if provider == "gemini":
        from app.agent.llm.gemini import GeminiClient

        return GeminiClient(api_key, model, temperature, timeout, max_retries)
    if provider == "anthropic":
        from app.agent.llm.anthropic import AnthropicClient

        return AnthropicClient(api_key, model, temperature, timeout, max_retries)
    if provider == "openai":
        from app.agent.llm.openai import OpenAIClient

        return OpenAIClient(api_key, model, temperature, timeout, max_retries)

    raise ValueError(f"Unknown LLM provider: {provider}")


class ResilientClient:
    """Falls back to rules when the model cannot be reached or answers badly.

    Records every fallback so a run's trace shows plainly that a degraded
    path was taken — a silent fallback would make evaluation meaningless.
    """

    def __init__(self, primary):
        self.primary = primary
        self.fallback = DeterministicClient()
        self.name = primary.name
        self.model = getattr(primary, "model", "")
        self.degraded = False
        self.degradation_reasons: list[str] = []

    def _degrade(self, where: str, exc: Exception) -> None:
        self.degraded = True
        message = f"{where}: {type(exc).__name__}: {exc}"
        self.degradation_reasons.append(message)
        logger.warning("LLM degraded to deterministic planner — %s", message)

    def converse(self, system: str, turns: list[Turn], tools: list[dict[str, Any]] | None = None) -> LLMResponse:
        if self.degraded:
            return self.fallback.converse(system, turns, tools)
        try:
            return self.primary.converse(system, turns, tools)
        except (LLMError, Exception) as exc:  # noqa: BLE001
            self._degrade("converse", exc)
            return self.fallback.converse(system, turns, tools)

    def structured(self, system: str, turns: list[Turn], schema: dict[str, Any]) -> dict[str, Any]:
        if self.degraded:
            return self.fallback.structured(system, turns, schema)
        try:
            return self.primary.structured(system, turns, schema)
        except (LLMError, Exception) as exc:  # noqa: BLE001
            self._degrade("structured", exc)
            return self.fallback.structured(system, turns, schema)


def resilient_client(provider: str, model: str, api_key: str, **kwargs) -> ResilientClient:
    return ResilientClient(get_client(provider, model, api_key, **kwargs))
