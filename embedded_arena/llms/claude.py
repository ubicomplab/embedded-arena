from __future__ import annotations

import json
import os
from typing import Any

import requests

from llms.base import (
    LLMResponse,
    ToolCall,
    normalize_usage,
    parse_json_arguments,
    raise_for_response,
)


class ClaudeAdapter:
    provider = "claude"

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        reasoning: str | None = None,
        timeout_seconds: int | None = None,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.reasoning = reasoning
        self.timeout_seconds = timeout_seconds or int(
            os.environ.get("LLM_TIMEOUT_SECONDS", "900")
        )
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is required for claude models.")

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        tool_choice: str | None = None,
    ) -> LLMResponse:
        system = ""
        anthropic_messages: list[dict[str, Any]] = []
        for message in messages:
            role = message["role"]
            if role == "system":
                system = f"{system}\n\n{message.get('content', '')}".strip()
            elif role == "tool":
                anthropic_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": message.get("tool_call_id", "tool_call"),
                                "content": str(message.get("content", "")),
                            }
                        ],
                    }
                )
            elif role == "assistant" and message.get("tool_calls"):
                content: list[dict[str, Any]] = []
                if message.get("content"):
                    content.append({"type": "text", "text": message["content"]})
                for call in message["tool_calls"]:
                    fn = call.get("function", {})
                    content.append(
                        {
                            "type": "tool_use",
                            "id": call.get("id", fn.get("name", "tool_call")),
                            "name": fn.get("name", ""),
                            "input": parse_json_arguments(fn.get("arguments")),
                        }
                    )
                anthropic_messages.append({"role": "assistant", "content": content})
            else:
                content = message.get("content", "")
                if isinstance(content, list):
                    anthropic_content: list[dict[str, Any]] = []
                    for block in content:
                        btype = block.get("type")
                        if btype == "text":
                            anthropic_content.append({"type": "text", "text": block.get("text", "")})
                        elif btype == "image_url":
                            url = block.get("image_url", {}).get("url", "")
                            if url.startswith("data:image/"):
                                header, b64_data = url.split(",", 1)
                                media_type = header.split(":")[1].split(";")[0]
                                anthropic_content.append({
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": media_type,
                                        "data": b64_data,
                                    },
                                })
                    anthropic_messages.append({
                        "role": "assistant" if role == "assistant" else "user",
                        "content": anthropic_content,
                    })
                else:
                    anthropic_messages.append(
                        {
                            "role": "assistant" if role == "assistant" else "user",
                            "content": str(content),
                        }
                    )

        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": anthropic_messages,
        }
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = [self._convert_tool(tool) for tool in tools]
        if self.reasoning:
            payload["thinking"] = {"type": "adaptive"}
            payload["output_config"] = {"effort": self.reasoning}

        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout_seconds,
        )
        raise_for_response("claude", response)
        data = response.json()
        texts = []
        tool_calls = []
        for block in data.get("content", []):
            if block.get("type") == "text":
                texts.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.get("id") or block.get("name") or "tool_call",
                        name=block.get("name") or "",
                        arguments=parse_json_arguments(block.get("input")),
                    )
                )
        usage = data.get("usage") or {}
        return LLMResponse(
            content="\n".join(texts).strip() or None,
            tool_calls=tool_calls,
            usage=normalize_usage(
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
            ),
            raw=data,
        )

    @staticmethod
    def _convert_tool(tool: dict[str, Any]) -> dict[str, Any]:
        fn = tool["function"]
        return {
            "name": fn["name"],
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
        }


def build(model: str, **kwargs: Any) -> ClaudeAdapter:
    return ClaudeAdapter(model, **kwargs)
