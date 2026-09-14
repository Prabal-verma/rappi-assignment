"""Google Gemini adapter (REST, via httpx).

Hand-rolled rather than SDK-based: the surface we need is two endpoints and
a JSON shape, and a direct client keeps the dependency tree small and the
failure modes visible.

Two Gemini specifics worth knowing:
  * function declarations use the OpenAPI subset, with `$ref`/`$defs`
    unsupported — schemas passed here must be flat;
  * tools and `responseSchema` are mutually exclusive, which is why
    investigation (tools) and decision (structured JSON) are separate calls.
"""

from __future__ import annotations

import json
import time
from typing import Any

import httpx

from app.agent.llm.base import LLMError, LLMResponse, ToolCall, Turn

BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


def _to_contents(turns: list[Turn]) -> list[dict[str, Any]]:
    contents: list[dict[str, Any]] = []
    for turn in turns:
        if turn.role == "user":
            contents.append({"role": "user", "parts": [{"text": turn.content or ""}]})
        elif turn.role == "assistant":
            parts: list[dict[str, Any]] = []
            if turn.content:
                parts.append({"text": turn.content})
            for call in turn.tool_calls:
                parts.append({"functionCall": {"name": call.name, "args": call.arguments}})
            contents.append({"role": "model", "parts": parts or [{"text": ""}]})
        else:  # tool result
            contents.append(
                {
                    "role": "user",
                    "parts": [
                        {
                            "functionResponse": {
                                "name": turn.tool_name or "tool",
                                "response": json.loads(turn.content or "{}"),
                            }
                        }
                    ],
                }
            )
    return contents


def _upper_types(schema: Any) -> Any:
    """Gemini's schema dialect wants SCREAMING type names."""
    if isinstance(schema, dict):
        out = {}
        for key, value in schema.items():
            if key == "type" and isinstance(value, str):
                out[key] = value.upper()
            else:
                out[key] = _upper_types(value)
        return out
    if isinstance(schema, list):
        return [_upper_types(v) for v in schema]
    return schema


class GeminiClient:
    name = "gemini"

    def __init__(self, api_key: str, model: str, temperature: float = 0.0, timeout: int = 60, max_retries: int = 2):
        if not api_key:
            raise LLMError("GEMINI_API_KEY is not set.")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{BASE_URL}/models/{self.model}:generateContent"
        # The key travels in a header, never in the query string.
        headers = {"x-goog-api-key": self.api_key, "Content-Type": "application/json"}

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = httpx.post(url, headers=headers, json=payload, timeout=self.timeout)
                if response.status_code == 429 or response.status_code >= 500:
                    raise LLMError(f"Gemini {response.status_code}: {response.text[:300]}")
                if response.status_code >= 400:
                    # Client errors will not fix themselves on retry.
                    raise LLMError(f"Gemini {response.status_code}: {response.text[:500]}")
                return response.json()
            except (httpx.HTTPError, LLMError) as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(1.5 * (attempt + 1))
                    continue
        raise LLMError(f"Gemini request failed after {self.max_retries + 1} attempts: {last_error}")

    @staticmethod
    def _parse(data: dict[str, Any]) -> LLMResponse:
        candidates = data.get("candidates") or []
        if not candidates:
            feedback = data.get("promptFeedback", {})
            raise LLMError(f"Gemini returned no candidates. Feedback: {feedback}")

        candidate = candidates[0]
        parts = candidate.get("content", {}).get("parts", []) or []
        text_chunks: list[str] = []
        tool_calls: list[ToolCall] = []

        for index, part in enumerate(parts):
            if "text" in part:
                text_chunks.append(part["text"])
            elif "functionCall" in part:
                fc = part["functionCall"]
                tool_calls.append(
                    ToolCall(
                        id=f"call_{index}",
                        name=fc.get("name", ""),
                        arguments=fc.get("args", {}) or {},
                    )
                )

        usage = data.get("usageMetadata", {})
        return LLMResponse(
            text="".join(text_chunks).strip(),
            tool_calls=tool_calls,
            input_tokens=usage.get("promptTokenCount", 0),
            output_tokens=usage.get("candidatesTokenCount", 0),
            stop_reason=candidate.get("finishReason", ""),
            raw=data,
        )

    def converse(
        self, system: str, turns: list[Turn], tools: list[dict[str, Any]] | None = None
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": _to_contents(turns),
            "generationConfig": {"temperature": self.temperature},
        }
        if tools:
            payload["tools"] = [{"functionDeclarations": [_upper_types(t) for t in tools]}]
        return self._parse(self._post(payload))

    def structured(
        self, system: str, turns: list[Turn], schema: dict[str, Any]
    ) -> dict[str, Any]:
        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": _to_contents(turns),
            "generationConfig": {
                "temperature": self.temperature,
                "responseMimeType": "application/json",
                "responseSchema": _upper_types(schema),
            },
        }
        response = self._parse(self._post(payload))
        try:
            return json.loads(response.text)
        except json.JSONDecodeError as exc:
            raise LLMError(f"Gemini returned invalid JSON: {response.text[:400]}") from exc
