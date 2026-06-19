from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    raw: dict[str, Any] | None = None


class LLMAPIError(RuntimeError):
    def __init__(self, provider: str, status_code: int | None, message: str):
        self.provider = provider
        self.status_code = status_code
        super().__init__(
            f"{provider} API error"
            + (f" {status_code}" if status_code else "")
            + f": {message}"
        )


def raise_for_response(provider: str, response: Any) -> None:
    if response.ok:
        return
    try:
        payload = response.json()
    except Exception:
        payload = response.text
    raise LLMAPIError(provider, response.status_code, json.dumps(payload, default=str))


class LLMAdapter(Protocol):
    provider: str
    model: str

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        tool_choice: str | None = None,
    ) -> LLMResponse: ...


def parse_json_arguments(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Tool arguments were not valid JSON: {value}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("Tool arguments must decode to a JSON object.")
        return parsed
    raise ValueError(f"Unsupported tool argument payload: {type(value).__name__}")


def usage_from_openai(payload: dict[str, Any]) -> dict[str, int]:
    usage = payload.get("usage") or {}
    return {
        "input_tokens": int(
            usage.get("prompt_tokens") or usage.get("input_tokens") or 0
        ),
        "output_tokens": int(
            usage.get("completion_tokens") or usage.get("output_tokens") or 0
        ),
        "total_tokens": int(usage.get("total_tokens") or 0),
    }


def normalize_usage(*, input_tokens: int = 0, output_tokens: int = 0) -> dict[str, int]:
    return {
        "input_tokens": int(input_tokens or 0),
        "output_tokens": int(output_tokens or 0),
        "total_tokens": int((input_tokens or 0) + (output_tokens or 0)),
    }
