"""OpenAI adapter (Chat Completions, via httpx)."""

from __future__ import annotations

import json
import time
from typing import Any

import httpx

from app.agent.llm.base import LLMError, LLMResponse, ToolCall, Turn

BASE_URL = "https://api.openai.com/v1/chat/completions"


def _to_messages(system: str, turns: list[Turn]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for turn in turns:
        if turn.role == "user":
            messages.append({"role": "user", "content": turn.content or ""})
        elif turn.role == "assistant":
            message: dict[str, Any] = {"role": "assistant", "content": turn.content or None}
            if turn.tool_calls:
                message["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
                    }
                    for call in turn.tool_calls
                ]
            messages.append(message)
        else:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": turn.tool_call_id or "call_0",
                    "content": turn.content or "{}",
                }
            )
    return messages


class OpenAIClient:
    name = "openai"

    def __init__(self, api_key: str, model: str, temperature: float = 0.0, timeout: int = 60, max_retries: int = 2):
        if not api_key:
            raise LLMError("OPENAI_API_KEY is not set.")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = httpx.post(BASE_URL, headers=headers, json=payload, timeout=self.timeout)
                if response.status_code == 429 or response.status_code >= 500:
                    raise LLMError(f"OpenAI {response.status_code}: {response.text[:300]}")
                if response.status_code >= 400:
                    raise LLMError(f"OpenAI {response.status_code}: {response.text[:500]}")
                return response.json()
            except (httpx.HTTPError, LLMError) as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(1.5 * (attempt + 1))
                    continue
        raise LLMError(f"OpenAI request failed: {last_error}")

    @staticmethod
    def _parse(data: dict[str, Any]) -> LLMResponse:
        choices = data.get("choices") or []
        if not choices:
            raise LLMError("OpenAI returned no choices.")
        message = choices[0].get("message", {})
        tool_calls = [
            ToolCall(
                id=tc.get("id", "call_0"),
                name=tc.get("function", {}).get("name", ""),
                arguments=json.loads(tc.get("function", {}).get("arguments") or "{}"),
            )
            for tc in message.get("tool_calls") or []
        ]
        usage = data.get("usage", {})
        return LLMResponse(
            text=(message.get("content") or "").strip(),
            tool_calls=tool_calls,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            stop_reason=choices[0].get("finish_reason", ""),
            raw=data,
        )

    def converse(
        self, system: str, turns: list[Turn], tools: list[dict[str, Any]] | None = None
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": _to_messages(system, turns),
        }
        if tools:
            payload["tools"] = [{"type": "function", "function": t} for t in tools]
        return self._parse(self._post(payload))

    def structured(self, system: str, turns: list[Turn], schema: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": _to_messages(system, turns),
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "purchasing_decision", "schema": schema, "strict": False},
            },
        }
        response = self._parse(self._post(payload))
        try:
            return json.loads(response.text)
        except json.JSONDecodeError as exc:
            raise LLMError(f"OpenAI returned invalid JSON: {response.text[:400]}") from exc
