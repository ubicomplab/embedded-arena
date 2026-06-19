from __future__ import annotations

import os
from typing import Any

import requests

from llms.base import (
    LLMResponse,
    ToolCall,
    parse_json_arguments,
    raise_for_response,
)


class GeminiAdapter:
    provider = "gemini"

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        reasoning: str | None = None,
        timeout_seconds: int | None = None,
    ):
        self.model = model
        self.api_key = (
            api_key
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
        )
        self.reasoning = reasoning
        self.timeout_seconds = timeout_seconds or int(
            os.environ.get("LLM_TIMEOUT_SECONDS", "900")
        )
        if not self.api_key:
            raise RuntimeError(
                "GEMINI_API_KEY or GOOGLE_API_KEY is required for gemini models."
            )

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        tool_choice: str | None = None,
    ) -> LLMResponse:
        system_parts = []
        contents: list[dict[str, Any]] = []
        for message in messages:
            role = message["role"]
            if role == "system":
                system_parts.append({"text": str(message.get("content", ""))})
            elif role == "tool":
                contents.append(
                    {
                        "role": "function",
                        "parts": [
                            {
                                "functionResponse": {
                                    "name": message.get("name", "tool"),
                                    "response": {
                                        "result": str(message.get("content", ""))
                                    },
                                }
                            }
                        ],
                    }
                )
            elif role == "assistant" and message.get("tool_calls"):
                parts = []
                if message.get("content"):
                    parts.append({"text": message["content"]})
                for call in message["tool_calls"]:
                    fn = call.get("function", {})
                    part = {
                        "functionCall": {
                            "name": fn.get("name", ""),
                            "args": parse_json_arguments(fn.get("arguments")),
                        }
                    }
                    metadata = call.get("metadata") or {}
                    if metadata.get("thoughtSignature"):
                        part["thoughtSignature"] = metadata["thoughtSignature"]
                    parts.append(part)
                contents.append({"role": "model", "parts": parts})
            else:
                content = message.get("content", "")
                if isinstance(content, list):
                    parts: list[dict[str, Any]] = []
                    for block in content:
                        btype = block.get("type")
                        if btype == "text":
                            parts.append({"text": block.get("text", "")})
                        elif btype == "image_url":
                            url = block.get("image_url", {}).get("url", "")
                            if url.startswith("data:image/"):
                                header, b64_data = url.split(",", 1)
                                mime_type = header.split(":")[1].split(";")[0]
                                parts.append({
                                    "inlineData": {
                                        "mimeType": mime_type,
                                        "data": b64_data,
                                    }
                                })
                    contents.append({
                        "role": "model" if role == "assistant" else "user",
                        "parts": parts,
                    })
                else:
                    contents.append(
                        {
                            "role": "model" if role == "assistant" else "user",
                            "parts": [{"text": str(content)}],
                        }
                    )

        payload: dict[str, Any] = {"contents": contents}
        if system_parts:
            payload["systemInstruction"] = {"parts": system_parts}
        if tools:
            payload["tools"] = [
                {
                    "functionDeclarations": [
                        {
                            "name": tool["function"]["name"],
                            "description": tool["function"].get("description", ""),
                            "parameters": tool["function"].get(
                                "parameters", {"type": "object"}
                            ),
                        }
                        for tool in tools
                    ]
                }
            ]

        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent",
            params={"key": self.api_key},
            json=payload,
            timeout=self.timeout_seconds,
        )
        raise_for_response("gemini", response)
        data = response.json()
        candidate = (data.get("candidates") or [{}])[0]
        parts = candidate.get("content", {}).get("parts", [])
        texts = []
        tool_calls = []
        for index, part in enumerate(parts):
            if "text" in part:
                texts.append(part["text"])
            if "functionCall" in part:
                call = part["functionCall"]
                metadata = {}
                if part.get("thoughtSignature"):
                    metadata["thoughtSignature"] = part["thoughtSignature"]
                tool_calls.append(
                    ToolCall(
                        id=f"{call.get('name', 'tool_call')}_{index}",
                        name=call.get("name", ""),
                        arguments=parse_json_arguments(call.get("args")),
                        metadata=metadata,
                    )
                )
        usage = data.get("usageMetadata") or {}
        input_tokens = int(usage.get("promptTokenCount") or 0)
        candidate_tokens = int(usage.get("candidatesTokenCount") or 0)
        thinking_tokens = int(usage.get("thoughtsTokenCount") or 0)
        output_tokens = candidate_tokens + thinking_tokens
        total_tokens = int(
            usage.get("totalTokenCount") or (input_tokens + output_tokens)
        )
        return LLMResponse(
            content="\n".join(texts).strip() or None,
            tool_calls=tool_calls,
            usage={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "thinking_tokens": thinking_tokens,
            },
            raw=data,
        )


def build(model: str, **kwargs: Any) -> GeminiAdapter:
    return GeminiAdapter(model, **kwargs)
