"""Anthropic Claude adapter (Messages API, via httpx)."""

from __future__ import annotations

import json
import time
from typing import Any

import httpx

from app.agent.llm.base import LLMError, LLMResponse, ToolCall, Turn

BASE_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"


def _to_messages(turns: list[Turn]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for turn in turns:
        if turn.role == "user":
            messages.append({"role": "user", "content": turn.content or ""})
        elif turn.role == "assistant":
            blocks: list[dict[str, Any]] = []
            if turn.content:
                blocks.append({"type": "text", "text": turn.content})
            for call in turn.tool_calls:
                blocks.append(
                    {"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments}
                )
            messages.append({"role": "assistant", "content": blocks})
        else:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": turn.tool_call_id or "call_0",
                            "content": turn.content or "{}",
                        }
                    ],
                }
            )
    return messages


class AnthropicClient:
    name = "anthropic"

    def __init__(self, api_key: str, model: str, temperature: float = 0.0, timeout: int = 60, max_retries: int = 2):
        if not api_key:
            raise LLMError("ANTHROPIC_API_KEY is not set.")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": API_VERSION,
            "content-type": "application/json",
        }
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = httpx.post(BASE_URL, headers=headers, json=payload, timeout=self.timeout)
                if response.status_code == 429 or response.status_code >= 500:
                    raise LLMError(f"Anthropic {response.status_code}: {response.text[:300]}")
                if response.status_code >= 400:
                    raise LLMError(f"Anthropic {response.status_code}: {response.text[:500]}")
                return response.json()
            except (httpx.HTTPError, LLMError) as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(1.5 * (attempt + 1))
                    continue
        raise LLMError(f"Anthropic request failed: {last_error}")

    @staticmethod
    def _parse(data: dict[str, Any]) -> LLMResponse:
        text_chunks: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in data.get("content", []) or []:
            if block.get("type") == "text":
                text_chunks.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.get("id", "call_0"),
                        name=block.get("name", ""),
                        arguments=block.get("input", {}) or {},
                    )
                )
        usage = data.get("usage", {})
        return LLMResponse(
            text="".join(text_chunks).strip(),
            tool_calls=tool_calls,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            stop_reason=data.get("stop_reason", ""),
            raw=data,
        )

    def converse(
        self, system: str, turns: list[Turn], tools: list[dict[str, Any]] | None = None
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 4096,
            "temperature": self.temperature,
            "system": system,
            "messages": _to_messages(turns),
        }
        if tools:
            payload["tools"] = [
                {
                    "name": t["name"],
                    "description": t["description"],
                    "input_schema": t["parameters"],
                }
                for t in tools
            ]
        return self._parse(self._post(payload))

    def structured(self, system: str, turns: list[Turn], schema: dict[str, Any]) -> dict[str, Any]:
        """Structured output via a forced single-tool call."""
        payload = {
            "model": self.model,
            "max_tokens": 4096,
            "temperature": self.temperature,
            "system": system,
            "messages": _to_messages(turns),
            "tools": [
                {
                    "name": "emit_decision",
                    "description": "Emit the final structured purchasing decision.",
                    "input_schema": schema,
                }
            ],
            "tool_choice": {"type": "tool", "name": "emit_decision"},
        }
        response = self._parse(self._post(payload))
        if not response.tool_calls:
            try:
                return json.loads(response.text)
            except json.JSONDecodeError as exc:
                raise LLMError("Claude did not emit the decision tool call.") from exc
        return response.tool_calls[0].arguments
