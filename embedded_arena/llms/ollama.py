from __future__ import annotations

import os
import json
from typing import Any

import requests

from llms.base import (
    LLMResponse,
    ToolCall,
    normalize_usage,
    parse_json_arguments,
    raise_for_response,
)


class OllamaAdapter:
    provider = "ollama"

    def __init__(
        self,
        model: str,
        *,
        base_url: str = "http://localhost:11434",
        reasoning: str | None = None,
        timeout_seconds: int | None = None,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.reasoning = reasoning
        self.timeout_seconds = timeout_seconds or int(
            os.environ.get("LLM_TIMEOUT_SECONDS", "900")
        )

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        tool_choice: str | None = None,
    ) -> LLMResponse:
        ollama_messages = self._convert_messages(messages)
        if tools:
            ollama_messages = self._add_tool_protocol(ollama_messages, tools)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": ollama_messages,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
        if self.reasoning:
            payload["think"] = True
        response = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=self.timeout_seconds,
        )
        raise_for_response("ollama", response)
        data = response.json()
        message = data.get("message", {})
        tool_calls = []
        for index, call in enumerate(message.get("tool_calls") or []):
            fn = call.get("function") or {}
            tool_calls.append(
                ToolCall(
                    id=call.get("id") or f"{fn.get('name', 'tool_call')}_{index}",
                    name=fn.get("name") or "",
                    arguments=parse_json_arguments(fn.get("arguments")),
                )
            )
        content = message.get("content")
        if not tool_calls:
            tool_calls = self._parse_emulated_tool_calls(content)
        return LLMResponse(
            content=None if tool_calls else content,
            tool_calls=tool_calls,
            usage=normalize_usage(
                input_tokens=data.get("prompt_eval_count", 0),
                output_tokens=data.get("eval_count", 0),
            ),
            raw=data,
        )

    @staticmethod
    def _convert_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        updated = []
        for message in messages:
            role = message["role"]
            if role == "tool":
                updated.append(
                    {
                        "role": "tool",
                        "tool_name": message.get("name", "tool"),
                        "content": str(message.get("content", "")),
                    }
                )
            elif role == "assistant" and message.get("tool_calls"):
                tool_calls = []
                for index, call in enumerate(message["tool_calls"]):
                    fn = call.get("function", {})
                    tool_calls.append(
                        {
                            "type": "function",
                            "function": {
                                "index": index,
                                "name": fn.get("name", ""),
                                "arguments": parse_json_arguments(fn.get("arguments")),
                            },
                        }
                    )
                updated.append(
                    {
                        "role": "assistant",
                        "content": str(message.get("content") or ""),
                        "tool_calls": tool_calls,
                    }
                )
            else:
                content = message.get("content", "")
                if isinstance(content, list):
                    text_parts = [
                        block.get("text", "")
                        for block in content
                        if block.get("type") == "text"
                    ]
                    content = "\n".join(text_parts)
                updated.append({"role": role, "content": str(content)})
        return updated

    @staticmethod
    def _add_tool_protocol(
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        tool_docs = []
        for tool in tools:
            fn = tool["function"]
            tool_docs.append(
                {
                    "name": fn["name"],
                    "description": fn.get("description", ""),
                    "parameters": fn.get("parameters", {}),
                }
            )
        instruction = (
            "You have access to tools through the Ollama tool schema. If native tool_calls are unavailable "
            "or you choose to express a tool call in text, return only this JSON shape and no markdown: "
            '{"tool_calls":[{"name":"WRITE_FILE","arguments":{"path":"file.py","text":"..."}}]}. '
            "Call tools before returning final check JSON. Never return a sandbox path in the final JSON "
            "unless a tool has created or verified that path. Available tools: "
            f"{json.dumps(tool_docs, separators=(',', ':'))}"
        )
        updated: list[dict[str, Any]] = []
        inserted = False
        for message in messages:
            if message["role"] == "system" and not inserted:
                updated.append(
                    {
                        **message,
                        "content": f"{message.get('content', '')}\n\n{instruction}",
                    }
                )
                inserted = True
            else:
                updated.append(message)
        if not inserted:
            updated.insert(0, {"role": "system", "content": instruction})
        return updated

    @staticmethod
    def _parse_emulated_tool_calls(content: str | None) -> list[ToolCall]:
        if not content:
            return []
        text = content.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return []
        calls = parsed.get("tool_calls")
        if calls is None and parsed.get("tool_call") is not None:
            calls = [parsed["tool_call"]]
        if not isinstance(calls, list):
            return []
        tool_calls = []
        for index, call in enumerate(calls):
            if not isinstance(call, dict):
                continue
            name = call.get("name")
            arguments = call.get("arguments") or {}
            if isinstance(name, str) and isinstance(arguments, dict):
                tool_calls.append(
                    ToolCall(
                        id=f"{name}_{index}",
                        name=name,
                        arguments=arguments,
                    )
                )
        return tool_calls


def build(model: str, **kwargs: Any) -> OllamaAdapter:
    return OllamaAdapter(model, **kwargs)
