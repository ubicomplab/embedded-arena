from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from llms.base import LLMResponse, ToolCall, normalize_usage, parse_json_arguments


class CLIAdapter:
    provider = "cli"

    def __init__(
        self,
        model: str = "manual",
        *,
        reasoning: str | None = None,
        script_path: str | None = None,
        **_: Any,
    ):
        self.model = model
        self.reasoning = reasoning
        self.call_index = 0
        configured_script = script_path or os.environ.get("CLI_LLM_SCRIPT")
        self.script_items = self._load_script(configured_script) if configured_script else []

    @staticmethod
    def _load_script(script_path: str) -> list[dict[str, Any]]:
        path = Path(script_path)
        if not path.exists():
            raise FileNotFoundError(f"CLI_LLM_SCRIPT does not exist: {script_path}")
        items: list[dict[str, Any]] = []
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                item = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{script_path}:{line_number} is not valid JSON"
                ) from exc
            if not isinstance(item, dict):
                raise ValueError(f"{script_path}:{line_number} must be a JSON object")
            items.append(item)
        return items

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        tool_choice: str | None = None,
    ) -> LLMResponse:
        self.call_index += 1
        if self.script_items:
            if self.call_index > len(self.script_items):
                raise RuntimeError(
                    "CLI_LLM_SCRIPT ran out of responses at "
                    f"LLM call {self.call_index}."
                )
            return self._response_from_object(self.script_items[self.call_index - 1])

        self._print_context(messages, tools)
        while True:
            raw = input("cli-agent> ").strip()
            if not raw:
                continue
            if raw in {":help", "help"}:
                self._print_help()
                continue
            if raw in {":messages", "messages"}:
                self._print_messages(messages)
                continue
            if raw in {":tools", "tools"}:
                self._print_tools(tools)
                continue
            if raw.startswith(":final "):
                content = raw[len(":final ") :].strip()
                return LLMResponse(
                    content=content,
                    usage=normalize_usage(),
                    raw={"mode": "interactive"},
                )
            if raw.startswith(":tool "):
                return self._tool_response_from_command(raw)
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as exc:
                print(f"Invalid JSON or command: {exc}", file=sys.stderr)
                self._print_help()
                continue
            if not isinstance(parsed, dict):
                print("Response must be a JSON object.", file=sys.stderr)
                continue
            return self._response_from_object(parsed)

    def _response_from_object(self, payload: dict[str, Any]) -> LLMResponse:
        tool_calls = []
        for index, call in enumerate(payload.get("tool_calls") or []):
            if not isinstance(call, dict):
                raise ValueError(f"tool_calls[{index}] must be an object")
            name = call.get("name")
            if not isinstance(name, str) or not name:
                raise ValueError(f"tool_calls[{index}].name must be a nonempty string")
            arguments = parse_json_arguments(call.get("arguments") or {})
            tool_calls.append(
                ToolCall(
                    id=str(call.get("id") or f"cli_{self.call_index}_{index}"),
                    name=name,
                    arguments=arguments,
                )
            )
        content = payload.get("content")
        if content is not None and not isinstance(content, str):
            content = json.dumps(content, separators=(",", ":"))
        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            usage=normalize_usage(),
            raw={"mode": "script" if self.script_items else "interactive"},
        )

    def _tool_response_from_command(self, raw: str) -> LLMResponse:
        remainder = raw[len(":tool ") :].strip()
        if " " not in remainder:
            raise ValueError("Use :tool TOOL_NAME {json_arguments}")
        name, arguments_text = remainder.split(" ", 1)
        arguments = parse_json_arguments(arguments_text)
        return LLMResponse(
            content=None,
            tool_calls=[
                ToolCall(
                    id=f"cli_{self.call_index}_0",
                    name=name,
                    arguments=arguments,
                )
            ],
            usage=normalize_usage(),
            raw={"mode": "interactive"},
        )

    def _print_context(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> None:
        print(f"\n=== CLI agent call {self.call_index} ===", file=sys.stderr)
        print(f"Available tools: {', '.join(self._tool_names(tools)) or '(none)'}", file=sys.stderr)
        print("Latest conversation items:", file=sys.stderr)
        for message in messages[-8:]:
            role = message.get("role", "?")
            if role == "assistant" and message.get("tool_calls"):
                calls = [
                    {
                        "id": call.get("id"),
                        "name": call.get("function", {}).get("name"),
                        "arguments": call.get("function", {}).get("arguments"),
                    }
                    for call in message.get("tool_calls", [])
                ]
                text = json.dumps(calls, indent=2)
            else:
                text = str(message.get("content", ""))
            print(f"\n--- {role} ---\n{text[-6000:]}", file=sys.stderr)
        self._print_help()

    @staticmethod
    def _print_messages(messages: list[dict[str, Any]]) -> None:
        print(json.dumps(messages, indent=2, default=str), file=sys.stderr)

    @staticmethod
    def _tool_names(tools: list[dict[str, Any]]) -> list[str]:
        return [tool.get("function", {}).get("name", "") for tool in tools]

    @staticmethod
    def _print_tools(tools: list[dict[str, Any]]) -> None:
        print(json.dumps(tools, indent=2, default=str), file=sys.stderr)

    @staticmethod
    def _print_help() -> None:
        print(
            "\nCommands:\n"
            "  :tool TOOL_NAME {json_arguments}\n"
            "  :final {flat_final_json}\n"
            "  {\"tool_calls\":[{\"name\":\"WRITE_FILE\",\"arguments\":{...}}]}\n"
            "  {\"content\":{\"model_path\":\"...\",\"data_path\":\"...\"}}\n"
            "  :messages    :tools    :help\n",
            file=sys.stderr,
        )


def build(model: str, **kwargs: Any) -> CLIAdapter:
    return CLIAdapter(model, **kwargs)
