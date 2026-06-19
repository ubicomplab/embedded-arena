from __future__ import annotations

import os
from typing import Any

import requests

from llms.base import (
    LLMResponse,
    ToolCall,
    normalize_usage,
    parse_json_arguments,
    raise_for_response,
    usage_from_openai,
)


class OpenAIAdapter:
    provider = "openai"

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        reasoning: str | None = None,
        timeout_seconds: int | None = None,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = (
            base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        ).rstrip("/")
        self.reasoning = reasoning
        self.timeout_seconds = timeout_seconds or int(
            os.environ.get("LLM_TIMEOUT_SECONDS", "900")
        )
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required for openai models.")

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        tool_choice: str | None = None,
    ) -> LLMResponse:
        if self._use_responses_api():
            return self._complete_responses(messages, tools)
        return self._complete_chat(messages, tools, tool_choice=tool_choice)

    def _use_responses_api(self) -> bool:
        if os.environ.get("OPENAI_USE_CHAT_COMPLETIONS") == "1":
            return False
        return self.model.startswith(("gpt-5", "o1", "o3", "o4"))

    def _complete_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        tool_choice: str | None = None,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice or "auto"
        if self.reasoning:
            payload["reasoning_effort"] = self.reasoning

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout_seconds,
        )
        raise_for_response("openai", response)
        data = response.json()
        message = data["choices"][0]["message"]
        tool_calls = []
        for call in message.get("tool_calls") or []:
            fn = call.get("function") or {}
            tool_calls.append(
                ToolCall(
                    id=call.get("id") or fn.get("name") or "tool_call",
                    name=fn.get("name") or "",
                    arguments=parse_json_arguments(fn.get("arguments")),
                )
            )
        return LLMResponse(
            content=message.get("content"),
            tool_calls=tool_calls,
            usage=usage_from_openai(data),
            raw=data,
        )

    def _complete_responses(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMResponse:
        instructions, input_items = self._messages_to_responses_input(messages)
        payload: dict[str, Any] = {
            "model": self.model,
            "input": input_items,
        }
        if instructions:
            payload["instructions"] = instructions
        if tools:
            payload["tools"] = [
                self._convert_tool_for_responses(tool) for tool in tools
            ]
            payload["tool_choice"] = "auto"
        if self.reasoning:
            payload["reasoning"] = {"effort": self.reasoning}

        response = requests.post(
            f"{self.base_url}/responses",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout_seconds,
        )
        raise_for_response("openai", response)
        data = response.json()
        texts: list[str] = []
        tool_calls: list[ToolCall] = []
        for item in data.get("output") or []:
            item_type = item.get("type")
            if item_type == "message":
                for content in item.get("content") or []:
                    if content.get("type") in {"output_text", "text"}:
                        texts.append(content.get("text", ""))
            elif item_type == "function_call":
                tool_calls.append(
                    ToolCall(
                        id=item.get("call_id")
                        or item.get("id")
                        or item.get("name")
                        or "tool_call",
                        name=item.get("name") or "",
                        arguments=parse_json_arguments(item.get("arguments")),
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
    def _convert_tool_for_responses(tool: dict[str, Any]) -> dict[str, Any]:
        fn = tool["function"]
        converted = {
            "type": "function",
            "name": fn["name"],
            "description": fn.get("description", ""),
            "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
        }
        if fn.get("strict") is not None:
            converted["strict"] = fn["strict"]
        return converted

    @staticmethod
    def _messages_to_responses_input(
        messages: list[dict[str, Any]],
    ) -> tuple[str, list[dict[str, Any]]]:
        instructions: list[str] = []
        input_items: list[dict[str, Any]] = []
        for message in messages:
            role = message["role"]
            if role == "system":
                instructions.append(str(message.get("content", "")))
            elif role == "tool":
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": message.get("tool_call_id", "tool_call"),
                        "output": str(message.get("content", "")),
                    }
                )
            elif role == "assistant" and message.get("tool_calls"):
                if message.get("content"):
                    input_items.append(
                        {"role": "assistant", "content": str(message["content"])}
                    )
                for call in message["tool_calls"]:
                    fn = call.get("function", {})
                    input_items.append(
                        {
                            "type": "function_call",
                            "call_id": call.get("id") or fn.get("name") or "tool_call",
                            "name": fn.get("name", ""),
                            "arguments": fn.get("arguments", "{}"),
                        }
                    )
            elif role in {"user", "assistant"}:
                content = message.get("content", "")
                if isinstance(content, list):
                    oai_content: list[dict[str, Any]] = []
                    for block in content:
                        btype = block.get("type")
                        if btype == "text":
                            oai_content.append({"type": "input_text", "text": block.get("text", "")})
                        elif btype == "image_url":
                            url = block.get("image_url", {}).get("url", "")
                            oai_content.append({"type": "input_image", "image_url": url})
                    input_items.append({"role": role, "content": oai_content})
                else:
                    input_items.append(
                        {"role": role, "content": str(content)}
                    )
        return "\n\n".join(part for part in instructions if part), input_items


def build(model: str, **kwargs: Any) -> OpenAIAdapter:
    return OpenAIAdapter(model, **kwargs)
