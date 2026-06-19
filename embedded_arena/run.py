from __future__ import annotations

import argparse
import base64
from collections import Counter, defaultdict
import importlib
import importlib.util
import json
import re
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
import yaml
from dotenv import load_dotenv

SRC_DIR = Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sandbox.sandbox import DEFAULT_SANDBOX_PATH, Sandbox
from exceptions import ExperimentSetupError
from llms.base import LLMAPIError
from schemas import CheckResult, Config, RunState


class JsonlLogger:
    def __init__(
        self,
        path: Path,
        secondary_path: Path | None = None,
        secondary_max_length: int | None = None,
        secondary_max_line_length: int = 1000,
    ):
        self.path = path
        self.secondary_path = secondary_path
        self.secondary_max_length = secondary_max_length
        self.secondary_max_line_length = secondary_max_line_length
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.secondary_path:
            self.secondary_path.parent.mkdir(parents=True, exist_ok=True)

    def _truncate_value(self, value: Any, max_length: int) -> Any:
        """Recursively truncate string values to max_length."""
        if isinstance(value, str):
            if len(value) > max_length:
                return value[:max_length] + f"...[{len(value) - max_length} more chars]"
            return value
        if isinstance(value, dict):
            return {k: self._truncate_value(v, max_length) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(self._truncate_value(v, max_length) for v in value)
        return value

    def _truncate_line(self, line: str, max_length: int) -> str:
        """Truncate a JSON line to max_length, preserving closing brace."""
        if len(line) <= max_length:
            return line
        # Keep space for closing brace and truncation marker
        reserved = len('","__truncated__":true}')
        target_length = max_length - reserved
        if target_length < 50:
            target_length = max_length - 5
        truncated = line[:target_length].rstrip(',"')
        return truncated + '","__truncated__":true}'

    def log(self, event: str, **payload: Any) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        record = {
            "timestamp": timestamp,
            "event": event,
            **payload,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Write full log
        with self.path.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")
        # Write truncated log if secondary path is specified
        if self.secondary_path and self.secondary_max_length:
            self.secondary_path.parent.mkdir(parents=True, exist_ok=True)
            truncated_payload = self._truncate_value(payload, self.secondary_max_length)
            short_record = {
                "timestamp": timestamp,
                "event": event,
                **truncated_payload,
            }
            line = json.dumps(short_record, default=str)
            # Truncate line if it exceeds max length
            if len(line) > self.secondary_max_line_length:
                line = self._truncate_line(line, self.secondary_max_line_length)
            with self.secondary_path.open("a") as f:
                f.write(line + "\n")


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key == "_base_":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml_with_base(path: Path, base_dir: Path) -> dict[str, Any]:
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    base_name = data.get("_base_")
    if not base_name:
        return data
    base_path = (base_dir / base_name).resolve()
    return deep_merge(load_yaml_with_base(base_path, base_dir), data)


def set_dotted_key(data: dict[str, Any], dotted_key: str, value: Any) -> None:
    if not dotted_key or dotted_key.startswith(".") or dotted_key.endswith("."):
        raise ValueError(f"Invalid override key: {dotted_key!r}")
    current: Any = data
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        if not isinstance(current, dict) or part not in current:
            raise KeyError(f"Override path does not exist: {dotted_key!r}")
        current = current[part]
    if not isinstance(current, dict) or parts[-1] not in current:
        raise KeyError(f"Override path does not exist: {dotted_key!r}")
    current[parts[-1]] = value


def parse_config_overrides(items: list[str]) -> dict[str, Any]:
    if len(items) % 2 != 0:
        raise ValueError("Config overrides must be KEY VALUE pairs, e.g. task.trials 1")
    overrides: dict[str, Any] = {}
    for key, raw_value in zip(items[0::2], items[1::2], strict=True):
        overrides[key] = yaml.safe_load(raw_value)
    return overrides


def apply_config_overrides(
    raw: dict[str, Any], overrides: dict[str, Any]
) -> dict[str, Any]:
    updated = dict(raw)
    for key, value in overrides.items():
        set_dotted_key(updated, key, value)
    return updated


def load_config(path: Path, overrides: dict[str, Any] | None = None) -> Config:
    raw = load_yaml_with_base(path.resolve(), path.resolve().parent)
    if "environment" in raw:
        env = raw["environment"]
        if isinstance(env, dict) and "_base_" in env:
            raw["environment"] = deep_merge(
                load_yaml_with_base(
                    (REPO_ROOT / "configs" / "environments" / env["_base_"]).resolve(),
                    REPO_ROOT / "configs" / "environments",
                ),
                env,
            )
    if overrides:
        raw = apply_config_overrides(raw, overrides)
    return Config(**raw)


def model_json_schema(model: type) -> dict[str, Any]:
    return model.model_json_schema()


def load_check_module(check_name: str):
    path = SRC_DIR / "checks" / check_name
    module_name = f"checks.{Path(check_name).stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load check module {check_name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_checks(config: Config) -> dict[str, Any]:
    checks = {}
    for check in config.checks:
        module = load_check_module(check.name)
        if not hasattr(module, "Input") or not hasattr(module, "check"):
            raise ValueError(f"{check.name} must define Input and check(state, input)")
        # YAMLInput is optional - some checks only have Input
        checks[check.name] = module
    return checks


def build_llm(llm_ref: str, reasoning: str | None):
    if "/" not in llm_ref:
        raise ValueError("--llm must use provider/model format, e.g. openai/gpt-4.1")
    provider, model = llm_ref.split("/", 1)
    module = importlib.import_module(f"llms.{provider}")
    return module.build(model, reasoning=reasoning)


def load_tool_registry(state: RunState) -> dict[str, dict[str, Any]]:
    registry: dict[str, dict[str, Any]] = {}
    for tool_module in state.config.environment.tools:
        module = importlib.import_module(f"tools.{Path(tool_module).stem}")
        registry.update(module.tools(state))
    return registry


def tool_message(call_id: str, name: str, content: str) -> dict[str, Any]:
    return {"role": "tool", "tool_call_id": call_id, "name": name, "content": content}


def sanitize_tool_result_for_model(content: str, limit: int = 12000) -> str:
    if (
        "Weights only load failed" in content
        or "Unsupported global:" in content
        or "arbitrary code execution" in content
        or "pickle.UnpicklingError" in content
    ):
        return (
            "ERROR: PyTorch checkpoint loading failed in safe weights-only mode. "
            "The checkpoint appears to require project-specific model classes that "
            "are not available in the sandbox. Inspect the checkpoint with safer "
            "metadata-only methods or use a known-compatible model artifact."
        )
    sanitized = content.replace("arbitrary code execution", "unsafe deserialization")
    sanitized = sanitized.replace(
        "Do it only if you got the file from a trusted source.",
        "Only load trusted checkpoint files with full deserialization.",
    )
    sanitized = re.sub(
        r"do those steps only if you trust[^.]*\.",
        "only use trusted checkpoint files.",
        sanitized,
        flags=re.IGNORECASE,
    )
    if len(sanitized) > limit:
        sanitized = (
            sanitized[:limit] + "\n...[tool output truncated for context length]..."
        )
    return sanitized


def sanitize_tool_arguments_for_model(
    name: str, arguments: dict[str, Any]
) -> dict[str, Any]:
    return arguments


def assistant_tool_call_message(response) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": response.content,
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(
                        sanitize_tool_arguments_for_model(call.name, call.arguments)
                    ),
                },
                "metadata": call.metadata,
            }
            for call in response.tool_calls
        ],
    }


def extract_json_object(text: str | None) -> dict[str, Any]:
    if not text:
        raise ValueError("model returned no final content")
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        candidates: list[dict[str, Any]] = []
        for match in re.finditer(r"\{", stripped):
            try:
                candidate, _ = decoder.raw_decode(stripped[match.start() :])
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict):
                candidates.append(candidate)
        if not candidates:
            raise
        parsed = candidates[-1]
    if not isinstance(parsed, dict):
        raise ValueError("final answer must be a JSON object")
    return parsed


def input_field_names(module: Any) -> set[str]:
    return set(module.Input.model_fields)


def union_field_names(checks: dict[str, Any], config: Config) -> set[str]:
    """All agent-settable input fields across all checks (YAML-fixed fields excluded)."""
    fixed: set[str] = set()
    for cc in config.checks:
        fixed.update(cc.params.keys())
    fields: set[str] = set()
    for module in checks.values():
        fields.update(input_field_names(module))
    return fields - fixed


def normalize_final_inputs(
    checks: dict[str, Any],
    final_inputs: dict[str, Any],
    config: Config,
) -> dict[str, Any]:
    allowed_fields = union_field_names(checks, config)
    allowed_fields.add("notes")
    unknown_fields = sorted(set(final_inputs) - allowed_fields)
    if unknown_fields:
        raise ValueError(
            "Final union JSON object contains fields that are not in the active "
            f"check input union schema: {unknown_fields}. Return only these keys "
            f"when needed: {sorted(allowed_fields)}"
        )
    check_config_map = {cc.name: cc for cc in config.checks}
    normalized = {}
    for name, module in checks.items():
        cc = check_config_map[name]
        yaml_fixed = set(cc.params.keys())
        agent_fields = input_field_names(module) - yaml_fixed

        # Check if this module has split Input/YAMLInput pattern
        if hasattr(module, "YAMLInput"):
            # Split pattern: validate both Input and YAMLInput separately
            value = {field: final_inputs[field] for field in agent_fields if field in final_inputs}
            value.update(cc.params)   # YAML params take precedence
            module.Input(**value)     # validate the agent input

            # Validate YAML input separately
            yaml_value = {}
            yaml_fields = set(module.YAMLInput.model_fields.keys())
            for field in yaml_fields:
                if field in cc.params:
                    yaml_value[field] = cc.params[field]
                elif field in final_inputs:
                    yaml_value[field] = final_inputs[field]
            module.YAMLInput(**yaml_value)  # validate the YAML input
        else:
            # Original pattern: single Input class
            value = {field: final_inputs[field] for field in agent_fields if field in final_inputs}
            value.update(cc.params)   # YAML params take precedence over agent values
            module.Input(**value)     # validate the merged input
        
        normalized[name] = value
    return normalized


def validate_final_inputs(
    checks: dict[str, Any],
    final_inputs: dict[str, Any],
    config: Config,
) -> None:
    normalize_final_inputs(checks, final_inputs, config)


def check_input_instructions(checks: dict[str, Any], config: Config) -> str:
    check_config_map = {cc.name: cc for cc in config.checks}
    properties: dict[str, Any] = {}
    required: set[str] = set()
    consumed_by: dict[str, list[str]] = {}

    for name, module in checks.items():
        cc = check_config_map[name]
        yaml_fixed = set(cc.params.keys())
        schema = model_json_schema(module.Input)
        for field, field_schema in schema.get("properties", {}).items():
            if field in yaml_fixed:
                continue   # YAML controls this — agent must not set it
            properties.setdefault(field, field_schema)
            consumed_by.setdefault(field, []).append(name)
        for req_field in schema.get("required", []):
            if req_field not in yaml_fixed:
                required.add(req_field)

    # Build per-check catalog shown to the agent
    check_catalog: dict[str, Any] = {}
    for name, module in checks.items():
        cc = check_config_map[name]
        desc = cc.description
        if not desc and hasattr(module, "check"):
            doc = (module.check.__doc__ or "").strip()
            desc = doc.split("\n")[0] if doc else ""
        check_catalog[name] = {
            "description": desc or "",
            "yaml_controlled_params": cc.params,
            "gives_feedback": cc.feedback,
        }

    union_schema = {
        "type": "object",
        "properties": {
            **properties,
            "notes": {
                "type": "string",
                "description": (
                    "Optional short private note for your next iteration only. You will "
                    "not have access to this iteration's chat history later; use `notes` "
                    "(and sandbox files like notes.md) to record what you created, what "
                    "failed, and what to try next."
                ),
            },
        },
        "required": sorted(required),
        "additionalProperties": False,
    }
    payload = {
        "checks": check_catalog,
        "final_answer_contract": (
            "Return one flat JSON object matching union_schema. Each check receives the "
            "subset of keys it declares. yaml_controlled_params are set by the harness "
            "and must NOT be included in your response. Shared keys such as project_dir "
            "have one value consumed by every check that declares them."
        ),
        "union_schema": union_schema,
        "field_consumers": consumed_by,
        "harness_fields": {
            "notes": (
                "Stored in the run summary and shown at the start of the next iteration "
                "as previous_iteration_note (with a clear prefix). It is not passed to checks."
            )
        },
    }
    return json.dumps(payload, indent=2)


def system_prompt(config: Config, checks: dict[str, Any]) -> str:
    optimization_goal = (
        "\nOptimization goal: first make every active check pass. Once all checks "
        "pass, improve the submitted artifacts to maximize the check scores. A score "
        "increase is useful only if the submission still passes every active check.\n"
    )
    return f"""You are an autonomous hardware-in-the-loop coding agent.

Task: {config.task.name}

Instructions:
{config.task.instructions}
{optimization_goal}

You are working in a persistent sandbox that survives across iterations in the current trial. The sandbox contains the task assets and the files you create; it does not contain the experiment harness or check source files. Use the schema, task instructions, and check feedback as the authoritative description of what to return instead of spending tool calls searching outside the sandbox root.

You may create files, leave notes for yourself, and preserve artifacts from earlier iterations. If the task asks you to create, edit, inspect, run, or verify files, you must use the available tools to do that work before returning the final union JSON object. Do not return a path for a file that you have not created or verified with a tool in this iteration or a previous iteration. At the end of each iteration, return only one flat JSON object matching the union schema below. Do not nest values under check file names. The harness will split this one object into per-check subsets after validation. Paths must be relative to the sandbox root and should point to files you created or edited. Shared keys are intentionally shared by all checks that consume them; for example one project_dir is passed to every check that declares project_dir.

Before you emit your final JSON, remember you will not have access to this iteration's chat history in the next iteration—leave durable reminders in the optional `notes` field and in sandbox files (for example notes.md) so your next turn can continue without that transcript.

Final union output schema:
{check_input_instructions(checks, config)}

Do not include prose outside the final JSON object when you are done with an iteration."""


def preserved_recommendations(value: str, *, limit: int) -> str:
    recommendation_lines = [
        line for line in value.splitlines() if "recommendation=" in line
    ]
    if not recommendation_lines:
        return ""
    block = "\n".join(recommendation_lines)
    if len(block) > limit:
        block = (
            block[:limit]
            + "\n...[recommendations truncated in prompt; full value is in run.log and summary.json]..."
        )
    return "Preserved recommendations from truncated feedback:\n" + block + "\n\n"


def compact_for_prompt(value: Any, *, string_limit: int = 4000) -> Any:
    if isinstance(value, str):
        if len(value) <= string_limit:
            return value
        recommendations = preserved_recommendations(
            value,
            limit=max(500, string_limit // 2),
        )
        if recommendations:
            remaining = max(500, string_limit - len(recommendations))
            return (
                recommendations
                + value[:remaining]
                + "\n...[truncated in prompt; full value is in run.log and summary.json]..."
            )
        return (
            value[:string_limit]
            + "\n...[truncated in prompt; full value is in run.log and summary.json]..."
        )
    if isinstance(value, list):
        return [compact_for_prompt(item, string_limit=string_limit) for item in value]
    if isinstance(value, dict):
        return {
            key: compact_for_prompt(item, string_limit=string_limit)
            for key, item in value.items()
        }
    return value


def summarize_prior_iterations(trial_summary: dict[str, Any]) -> dict[str, Any]:
    iterations = trial_summary["iterations"]
    if not iterations:
        return {
            "all_iterations": [],
            "last_iteration": None,
            "best_scores_by_check": {},
            "failed_checks_last_iteration": [],
            "repeated_submission_warning": None,
        }

    best_scores: dict[str, dict[str, Any]] = {}
    for iteration in iterations:
        for name, result in iteration.get("checks", {}).items():
            score = result.get("score")
            if not isinstance(score, (int, float)):
                continue
            current = best_scores.get(name)
            if current is None or score > current["score"]:
                best_scores[name] = {
                    "score": score,
                    "score_unit": result.get("score_unit"),
                    "success": result.get("success"),
                    "iteration": iteration.get("iteration"),
                }

    last = iterations[-1]
    failed = [
        name
        for name, result in last.get("checks", {}).items()
        if not result.get("success")
    ]
    repeated_warning = None
    if failed and len(iterations) >= 2:
        previous_inputs = iterations[-2].get("inputs")
        last_inputs = last.get("inputs")
        if previous_inputs == last_inputs:
            repeated_warning = (
                "The last two iterations submitted identical final inputs while at "
                "least one check still failed. Do not submit the same paths/values "
                "again unless you have modified the referenced files in the sandbox "
                "and verified that the failing check should now behave differently."
            )

    return {
        "all_iterations": iterations,
        "last_iteration": last,
        "best_scores_by_check": best_scores,
        "failed_checks_last_iteration": failed,
        "repeated_submission_warning": repeated_warning,
    }


def union_inputs_from_iteration(iteration: dict[str, Any] | None) -> dict[str, Any]:
    if not iteration:
        return {}
    union: dict[str, Any] = {}
    conflicts: dict[str, list[Any]] = {}
    for check_inputs in iteration.get("inputs", {}).values():
        if not isinstance(check_inputs, dict):
            continue
        for key, value in check_inputs.items():
            if key in union and union[key] != value:
                conflicts.setdefault(key, [union[key]]).append(value)
            else:
                union[key] = value
    if conflicts:
        union["_conflicting_values"] = conflicts
    return union


def final_notes(final_inputs: dict[str, Any]) -> str | None:
    notes = final_inputs.get("notes")
    return notes if isinstance(notes, str) and notes.strip() else None


def build_next_action_requirements(
    *,
    checks: dict[str, Any],
    prior_summary: dict[str, Any] | None,
    current_sandbox_files: list[str],
) -> list[str]:
    requirements: list[str] = []
    if not prior_summary or not prior_summary.get("last_iteration"):
        requirements.append(
            "If the task asks for new or compressed artifacts, create the requested candidate files before returning final JSON; do not only inspect the starting files."
        )
        return requirements

    failed = prior_summary.get("failed_checks_last_iteration") or []
    last_union = union_inputs_from_iteration(prior_summary.get("last_iteration"))
    has_failed_checks = bool(failed)
    if has_failed_checks:
        requirements.append(
            "At least one check failed last iteration. Before returning final JSON, make a relevant change with WRITE_FILE or a file-producing RUN_COMMAND, then submit the changed path/value."
        )

    for key, value in sorted(last_union.items()):
        if key.endswith("_path") and isinstance(value, str) and value:
            if value not in current_sandbox_files:
                requirements.append(
                    f"Last submitted {key}={value!r} is not visible in the current sandbox inventory. Create it, correct it, or choose a visible path before submitting."
                )

    if prior_summary.get("repeated_submission_warning"):
        requirements.append(prior_summary["repeated_submission_warning"])
    return requirements


def _collect_feedback_images(
    trial_summary: dict[str, Any],
    config: Config,
) -> list[bytes]:
    """Load PNG bytes for feedback_image_paths from the last iteration's results."""
    if not trial_summary.get("iterations"):
        return []
    last_iter = trial_summary["iterations"][-1]
    feedback_enabled = {c.name: c.feedback for c in config.checks}
    images: list[bytes] = []
    for check_name, result_dict in last_iter.get("checks", {}).items():
        if not feedback_enabled.get(check_name, False):
            continue
        for img_path in result_dict.get("feedback_image_paths", []):
            try:
                images.append(Path(img_path).read_bytes())
            except (OSError, IOError):
                pass
    return images


def build_iteration_messages(
    *,
    state: RunState,
    config: Config,
    checks: dict[str, Any],
    trial_summary: dict[str, Any],
    iteration_index: int,
) -> list[dict[str, Any]]:
    current_sandbox_files = sandbox_file_inventory(state)
    tool_budget = {
        "max_tool_calls_this_iteration": config.task.max_tool_calls,
        "tool_calls_remaining_this_iteration": config.task.max_tool_calls,
        "tool_call_accounting": (
            "Every individual tool call counts against this iteration budget. "
            "If you request multiple tool calls in one response, the harness executes "
            "that whole batch, then disables tools once the budget has been reached "
            "or exceeded. After each tool batch, a harness note states how many "
            "calls remain this iteration."
        ),
    }
    if iteration_index == 0:
        status = {
            "iteration": 0,
            "prior_iterations": [],
            "current_sandbox_files": current_sandbox_files,
            "tool_budget": tool_budget,
            "next_action_requirements": build_next_action_requirements(
                checks=checks,
                prior_summary=None,
                current_sandbox_files=current_sandbox_files,
            ),
            "instructions": (
                "Begin iteration 0. Use the available tools to perform any requested "
                "file or code work, then return the final flat union JSON object."
            ),
        }
    else:
        prior_summary = summarize_prior_iterations(trial_summary)
        next_action_requirements = build_next_action_requirements(
            checks=checks,
            prior_summary=prior_summary,
            current_sandbox_files=current_sandbox_files,
        )
        last_iter = prior_summary["last_iteration"]
        raw_prev_notes = last_iter.get("notes") if last_iter else None
        if isinstance(raw_prev_notes, str) and raw_prev_notes.strip():
            previous_iteration_note = (
                "Previous iteration note: " + raw_prev_notes.strip()
            )
        else:
            previous_iteration_note = None
        status = {
            "iteration": iteration_index,
            "prior_iterations": compact_for_prompt(prior_summary["all_iterations"]),
            "last_iteration": compact_for_prompt(prior_summary["last_iteration"]),
            "last_union_inputs": union_inputs_from_iteration(
                prior_summary["last_iteration"]
            ),
            "previous_iteration_note": previous_iteration_note,
            "best_scores_by_check": prior_summary["best_scores_by_check"],
            "failed_checks_last_iteration": prior_summary[
                "failed_checks_last_iteration"
            ],
            "repeated_submission_warning": prior_summary["repeated_submission_warning"],
            "current_sandbox_files": current_sandbox_files,
            "tool_budget": tool_budget,
            "next_action_requirements": next_action_requirements,
            "instructions": (
                "Continue from the same persistent sandbox. The conversation context "
                "has been reset to keep LLM calls short; use your files in the sandbox "
                "as durable memory. Read previous_iteration_note (if present), inspect prior "
                "model versions, and prior "
                "artifacts as needed before returning the next final flat union JSON object. "
                "If any check failed in the last iteration, make a relevant change to the "
                "referenced files or final input values before resubmitting. Prefer using "
                "the current_sandbox_files inventory over re-listing the root. Repeating an "
                "identical failing submission wastes the iteration. Satisfy every item in "
                "next_action_requirements before returning final JSON."
            ),
        }
    user_text = (
        "Iteration status summary:\n"
        f"{json.dumps(status, indent=2, default=str)}\n\n"
        "Treat next_action_requirements as mandatory for this iteration. "
        "You will not have access to this iteration's chat history when the next "
        "iteration starts—leave essential information in the `notes` field of your "
        "final JSON for your next self. You may include a short `notes` string to "
        "record what files you created, what failed, and what to try next. "
        "Keep durable detail in sandbox files too, such as notes.md, "
        "and save past model versions or architecture YAMLs under versioned "
        "paths so you can inspect or restore them later."
    )

    feedback_images = _collect_feedback_images(trial_summary, config)
    if feedback_images:
        user_content: str | list[dict[str, Any]] = [{"type": "text", "text": user_text}]
        for img_bytes in feedback_images:
            b64 = base64.b64encode(img_bytes).decode()
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            })
    else:
        user_content = user_text

    return [
        {"role": "system", "content": system_prompt(config, checks)},
        {"role": "user", "content": user_content},
    ]


def should_include_in_inventory(relative_path: str) -> bool:
    parts = Path(relative_path).parts
    if "__MACOSX" in parts:
        return False
    if any(
        part.startswith("._") or part in {".DS_Store", "__pycache__"} for part in parts
    ):
        return False
    suffix = Path(relative_path).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}:
        return False
    return True


def extension_summary(paths: list[Path]) -> str:
    counts = Counter(path.suffix.lower() or "[no extension]" for path in paths)
    return ", ".join(f"{suffix}={count}" for suffix, count in sorted(counts.items()))


def sandbox_file_inventory(
    state: RunState,
    *,
    max_files: int = 200,
    max_files_per_directory: int = 20,
) -> list[str]:
    root = Path(state.sandbox.sandbox_path)
    by_directory: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        try:
            rel_path = path.relative_to(root)
        except ValueError:
            continue
        rel = rel_path.as_posix()
        if not should_include_in_inventory(rel):
            continue
        directory = rel_path.parent.as_posix()
        if directory == ".":
            directory = ""
        by_directory[directory].append(rel_path)

    inventory: list[str] = []
    for directory in sorted(by_directory):
        paths = sorted(by_directory[directory], key=lambda item: item.as_posix())
        visible = paths[:max_files_per_directory]
        for rel_path in visible:
            inventory.append(rel_path.as_posix())
            if len(inventory) >= max_files:
                inventory.append(f"...truncated after {max_files} inventory entries...")
                return inventory
        omitted = paths[max_files_per_directory:]
        if omitted:
            prefix = f"{directory}/" if directory else ""
            inventory.append(
                f"{prefix}... {len(omitted)} more files in this directory ({extension_summary(omitted)})"
            )
            if len(inventory) >= max_files:
                inventory.append(f"...truncated after {max_files} inventory entries...")
                return inventory
    return inventory


def build_forced_final_messages(
    *,
    state: RunState,
    checks: dict[str, Any],
    trial_summary: dict[str, Any],
    reason: str,
) -> list[dict[str, Any]]:
    user_payload = {
        "iteration": state.iteration_index,
        "reason": reason,
        "current_sandbox_files": sandbox_file_inventory(state),
        "prior_iteration_status": compact_for_prompt(
            summarize_prior_iterations(trial_summary),
            string_limit=1800,
        ),
        "instruction": (
            "Return only one flat final union JSON object now. Use only "
            "sandbox-relative paths from current_sandbox_files, or paths to files you "
            "created earlier in this trial. Do not request more tools. "
            "You will not have access to this iteration's chat history in the next "
            "iteration—put essential reminders in the optional `notes` field and in "
            "sandbox files if needed."
        ),
    }
    return [
        {"role": "system", "content": system_prompt(state.config, checks)},
        {"role": "user", "content": json.dumps(user_payload, indent=2, default=str)},
    ]


def run_single_iteration(
    *,
    llm,
    state: RunState,
    checks: dict[str, Any],
    logger: JsonlLogger,
    summary: dict[str, Any],
    trial_summary: dict[str, Any],
    messages: list[dict[str, Any]],
) -> dict[str, Any]:
    tool_registry = load_tool_registry(state)
    tool_specs = [tool["spec"] for tool in tool_registry.values()]
    max_tool_calls = state.config.task.max_tool_calls
    tool_calls_used = 0
    forced_final_attempts = 0
    forced_final_compacted = False

    logger.log(
        "iteration_start",
        trial=state.trial_index,
        iteration=state.iteration_index,
        tools=list(tool_registry.keys()),
    )

    while True:
        active_tools = tool_specs if tool_calls_used < max_tool_calls else []
        if not active_tools and not forced_final_compacted:
            messages = build_forced_final_messages(
                state=state,
                checks=checks,
                trial_summary=trial_summary,
                reason="tool-call budget exhausted",
            )
            forced_final_compacted = True
        for attempt in range(3):
            llm_started = time.perf_counter()
            try:
                response = llm.complete(messages, active_tools)
                break
            except LLMAPIError as exc:
                retryable = exc.status_code == 429 or (
                    exc.status_code is not None and exc.status_code >= 500
                )
                if not retryable:
                    raise
                llm_elapsed = time.perf_counter() - llm_started
                logger.log(
                    "llm_call_error",
                    trial=state.trial_index,
                    iteration=state.iteration_index,
                    elapsed_seconds=llm_elapsed,
                    attempt=attempt + 1,
                    error=traceback.format_exc(),
                )
                if attempt == 2:
                    raise
                wait = 60 if exc.status_code == 429 else 2**attempt
                logger.log(
                    "llm_rate_limit_wait" if exc.status_code == 429 else "llm_retry_wait",
                    trial=state.trial_index,
                    iteration=state.iteration_index,
                    wait_seconds=wait,
                    attempt=attempt + 1,
                )
                time.sleep(wait)
            except requests.exceptions.RequestException as exc:
                llm_elapsed = time.perf_counter() - llm_started
                logger.log(
                    "llm_call_error",
                    trial=state.trial_index,
                    iteration=state.iteration_index,
                    elapsed_seconds=llm_elapsed,
                    attempt=attempt + 1,
                    error=traceback.format_exc(),
                )
                if attempt == 2:
                    raise LLMAPIError(
                        getattr(llm, "provider", "llm"),
                        None,
                        f"request failed after 3 attempts: {exc}",
                    ) from exc
                time.sleep(2**attempt)
        llm_elapsed = time.perf_counter() - llm_started
        summary["tokens"]["input"] += response.usage.get("input_tokens", 0)
        summary["tokens"]["output"] += response.usage.get("output_tokens", 0)
        summary["tokens"]["total"] += response.usage.get("total_tokens", 0)
        logger.log(
            "llm_call",
            trial=state.trial_index,
            iteration=state.iteration_index,
            elapsed_seconds=llm_elapsed,
            usage=response.usage,
            messages=messages,
            response={
                "content": response.content,
                "tool_calls": [
                    {
                        "id": call.id,
                        "name": call.name,
                        "arguments": call.arguments,
                        "metadata": call.metadata,
                    }
                    for call in response.tool_calls
                ],
            },
        )

        if response.tool_calls and tool_calls_used < max_tool_calls:
            messages.append(assistant_tool_call_message(response))
            for call in response.tool_calls:
                if call.name not in tool_registry:
                    result = f"ERROR: tool {call.name} is not available."
                else:
                    tool_started = time.perf_counter()
                    try:
                        raw_result = tool_registry[call.name]["fn"](call.arguments)
                        result = "" if raw_result is None else str(raw_result)
                    except Exception:
                        result = f"ERROR:\n{traceback.format_exc()}"
                    elapsed = time.perf_counter() - tool_started
                    logger.log(
                        "tool_call",
                        trial=state.trial_index,
                        iteration=state.iteration_index,
                        elapsed_seconds=elapsed,
                        name=call.name,
                        arguments=call.arguments,
                        result=result,
                    )
                tool_calls_used += 1
                summary["tool_calls"] += 1
                messages.append(
                    tool_message(
                        call.id,
                        call.name,
                        sanitize_tool_result_for_model(result),
                    )
                )
            remaining = max(0, max_tool_calls - tool_calls_used)
            if remaining > 0:
                if remaining <= 3:
                    budget_note = (
                        f"Harness note: only {remaining} tool call(s) remaining this "
                        f"iteration ({tool_calls_used}/{max_tool_calls} used). Consider "
                        "beginning to change code (if you have not already) and working "
                        "toward the final union JSON."
                    )
                else:
                    budget_note = (
                        f"Harness note: {remaining} tool call(s) remaining this iteration "
                        f"({tool_calls_used}/{max_tool_calls} used)."
                    )
            else:
                budget_note = (
                    f"Harness note: tool-call budget exhausted for this iteration "
                    f"({tool_calls_used}/{max_tool_calls} used). On your next response "
                    "return only the final flat union JSON object; do not request tools."
                )
            messages.append({"role": "user", "content": budget_note})
            continue

        if response.tool_calls and tool_calls_used >= max_tool_calls:
            forced_final_attempts += 1
            if forced_final_attempts > 2:
                raise RuntimeError(
                    "model continued requesting tools after the tool-call budget was exhausted"
                )
            messages = build_forced_final_messages(
                state=state,
                checks=checks,
                trial_summary=trial_summary,
                reason=(
                    "tool-call budget exhausted and the previous response still "
                    "requested tools"
                ),
            )
            continue

        if not response.content:
            forced_final_attempts += 1
            if forced_final_attempts > 2:
                raise ValueError("model returned no final content")
            messages = build_forced_final_messages(
                state=state,
                checks=checks,
                trial_summary=trial_summary,
                reason="previous response returned no final content",
            )
            continue

        try:
            final_inputs = extract_json_object(response.content)
            notes = final_notes(final_inputs)
            final_inputs = normalize_final_inputs(checks, final_inputs, state.config)
        except Exception as exc:
            forced_final_attempts += 1
            if forced_final_attempts > 2:
                raise
            messages.append({"role": "assistant", "content": response.content or ""})
            remaining = max(0, max_tool_calls - tool_calls_used)
            if remaining <= 0:
                budget_part = (
                    "Tool-call budget for this iteration is exhausted; do not request tools."
                )
            elif remaining <= 3:
                budget_part = (
                    f"Only {remaining} tool call(s) remaining this iteration; consider "
                    "beginning to change code (if you have not already) and working "
                    "toward the final union JSON."
                )
            else:
                budget_part = (
                    f"You have {remaining} tool call(s) remaining this iteration."
                )
            invalid_body = (
                f"The final union JSON object was invalid: {exc} {budget_part} "
                "Use tools if more file work is needed, otherwise return only a corrected final JSON object."
            )
            messages.append({"role": "user", "content": invalid_body})
            continue

        messages.append({"role": "assistant", "content": response.content or ""})
        if notes is not None:
            final_inputs["_notes"] = notes
        return final_inputs


def failed_check(message: str) -> CheckResult:
    return CheckResult(success=False, score=0.0, score_unit="error", feedback=message)


def _write_check_summaries_to_iter_result(
    output_dir: Path,
    trial_index: int,
    iteration_index: int,
    results: dict[str, CheckResult],
    config: Config,
) -> None:
    """Append a 'checks' section to iter_result.json with per-check harness summary."""
    iter_dir = output_dir / f"trial_{trial_index}" / f"iter_{iteration_index}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    path = iter_dir / "iter_result.json"
    existing: dict[str, Any] = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}
    check_config_map = {cc.name: cc for cc in config.checks}
    check_summaries: dict[str, Any] = {}
    for check_name, result in results.items():
        cc = check_config_map.get(check_name)
        check_summaries[check_name] = {
            "success": result.success,
            "score": result.score,
            "score_unit": result.score_unit,
            "feedback": result.feedback if (cc is not None and cc.feedback) else None,
        }
    existing["checks"] = check_summaries
    path.write_text(json.dumps(existing, indent=2, default=str), encoding="utf-8")


def run_checks(
    *,
    state: RunState,
    checks: dict[str, Any],
    final_inputs: dict[str, Any],
    logger: JsonlLogger,
) -> dict[str, CheckResult]:
    results: dict[str, CheckResult] = {}
    for check_config in state.config.checks:
        name = check_config.name
        module = checks[name]

        # Skip if any declared dependency did not pass
        failed_deps = [
            dep for dep in check_config.depends_on
            if not results.get(dep, CheckResult(success=False, score=0.0, score_unit="skipped", feedback="")).success
        ]
        if failed_deps:
            skip_result = CheckResult(
                success=False,
                score=0.0,
                score_unit="skipped",
                feedback=f"skipped: dependencies did not pass: {failed_deps}",
            )
            results[name] = skip_result
            logger.log(
                "check",
                trial=state.trial_index,
                iteration=state.iteration_index,
                elapsed_seconds=0.0,
                name=name,
                input=final_inputs.get(name),
                result=skip_result.model_dump(),
            )
            continue

        started = time.perf_counter()
        try:
            if name not in final_inputs:
                raise ValueError(f"Missing input for {name}")
            
            # Check if the module has both Input and YAMLInput (split pattern)
            if hasattr(module, "YAMLInput"):
                # Split pattern: extract fixed/yaml-controlled inputs
                agent_input = module.Input(**final_inputs[name])
                yaml_input_data = {}
                # Collect YAML-controlled parameters from final_inputs[name]
                yaml_fields = set(module.YAMLInput.model_fields.keys())
                for field in yaml_fields:
                    if field in final_inputs[name]:
                        yaml_input_data[field] = final_inputs[name][field]
                yaml_input = module.YAMLInput(**yaml_input_data)
                state.metadata["feedback_enabled"] = check_config.feedback
                result = module.check(state, agent_input, yaml_input)
            else:
                # Original pattern: single Input class
                check_input = module.Input(**final_inputs[name])
                result = module.check(state, check_input)
        except ExperimentSetupError:
            elapsed = time.perf_counter() - started
            logger.log(
                "setup_error",
                trial=state.trial_index,
                iteration=state.iteration_index,
                elapsed_seconds=elapsed,
                name=name,
                input=final_inputs.get(name),
                error=traceback.format_exc(),
            )
            raise
        except Exception:
            result = failed_check(traceback.format_exc())
        elapsed = time.perf_counter() - started
        results[name] = result
        logger.log(
            "check",
            trial=state.trial_index,
            iteration=state.iteration_index,
            elapsed_seconds=elapsed,
            name=name,
            input=final_inputs.get(name),
            result=result.model_dump(),
        )
    return results


_SNAPSHOT_SKIP_DIRS = {"__pycache__", ".git", "MAX78000_example_code", "ESP32_example_code"}
_SNAPSHOT_SKIP_EXTENSIONS = {".o", ".d", ".a", ".su"}
_SNAPSHOT_SKIP_DIR_PATTERNS = {"CMakeFiles", "cmake_install.cmake", ".cmake"}


def snapshot_sandbox(sandbox_path: str, snapshot_dir: Path) -> None:
    """Copy sandbox source files to snapshot_dir, skipping build intermediates."""
    sandbox_root = Path(sandbox_path)
    if not sandbox_root.exists():
        return
    for src_file in sorted(sandbox_root.rglob("*")):
        if not src_file.is_file():
            continue
        rel = src_file.relative_to(sandbox_root)
        parts = rel.parts
        if any(p in _SNAPSHOT_SKIP_DIRS for p in parts):
            continue
        if any(p in _SNAPSHOT_SKIP_DIR_PATTERNS for p in parts):
            continue
        if src_file.suffix.lower() in _SNAPSHOT_SKIP_EXTENSIONS:
            continue
        dst = snapshot_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dst)


def prepare_sandbox(config: Config, sandbox: Sandbox, config_path: Path) -> None:
    sandbox.clean()
    for file in config.environment.files:
        src = Path(file.src)
        if not src.is_absolute():
            src = (config_path.parent / src).resolve()
            if not src.exists():
                src = (REPO_ROOT / file.src).resolve()
        if not src.exists():
            raise FileNotFoundError(f"Configured input file does not exist: {file.src}")
        sandbox.copy_to_playground(str(src), file.dst)


def _trial_checkpoint_path(output_dir: Path, trial_index: int) -> Path:
    return output_dir / f"trial_{trial_index}" / "trial_checkpoint.json"


def _save_trial_checkpoint(output_dir: Path, trial_index: int, trial_summary: dict[str, Any]) -> None:
    path = _trial_checkpoint_path(output_dir, trial_index)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(trial_summary, indent=2, default=str), encoding="utf-8")


def _load_trial_checkpoint(output_dir: Path, trial_index: int) -> dict[str, Any] | None:
    path = _trial_checkpoint_path(output_dir, trial_index)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _checkpoint_results_by_iteration(
    trial_summary: dict[str, Any],
) -> list[dict[str, CheckResult]]:
    hydrated: list[dict[str, CheckResult]] = []
    for iteration in trial_summary.get("iterations", []):
        checks = iteration.get("checks") if isinstance(iteration, dict) else None
        if not isinstance(checks, dict):
            continue
        iteration_results: dict[str, CheckResult] = {}
        for name, payload in checks.items():
            if not isinstance(payload, dict):
                continue
            try:
                iteration_results[str(name)] = CheckResult(**payload)
            except Exception:
                continue
        hydrated.append(iteration_results)
    return hydrated



def safe_rmtree(path: Path, *, purpose: str) -> None:
    target = path.resolve()
    repo = REPO_ROOT.resolve()
    forbidden = {Path("/").resolve(), Path.home().resolve(), repo, SRC_DIR.resolve()}
    if target in forbidden:
        raise RuntimeError(f"Refusing to delete unsafe {purpose} directory: {target}")
    if target.exists() and ((target / ".git").exists() or (target / "pyproject.toml").exists()):
        raise RuntimeError(f"Refusing to delete {purpose} directory that looks like a project root: {target}")
    shutil.rmtree(target)


def resolve_output_dir(args: argparse.Namespace, config_path: Path) -> Path:
    if getattr(args, "output_dir", None):
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = (REPO_ROOT / output_dir).resolve()
        return output_dir
    experiment_name = args.output_name or config_path.stem
    return (REPO_ROOT / "outputs" / experiment_name).resolve()

def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    load_dotenv()
    if args.overwrite and args.resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")
    config_path = Path(args.config).resolve()
    config_overrides = parse_config_overrides(args.config_overrides)
    if args.iterations is not None:
        config_overrides["task.iterations"] = args.iterations
    if args.trials is not None:
        config_overrides["task.trials"] = args.trials
    config = load_config(config_path, config_overrides)
    checks = load_checks(config)
    llm = build_llm(args.llm, args.reasoning)
    experiment_name = args.output_name or config_path.stem
    output_dir = resolve_output_dir(args, config_path)
    if output_dir.exists() and args.overwrite:
        safe_rmtree(output_dir, purpose="output")
    elif output_dir.exists() and args.resume:
        summary_path = output_dir / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(
                f"{summary_path} does not exist; cannot resume this output directory"
            )
    elif output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"{output_dir} already exists and is not empty; pass --overwrite, --resume, or choose --output-name"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = JsonlLogger(
        output_dir / "run.log",
        secondary_path=output_dir / "run_short.log",
        secondary_max_length=200,
    )

    if args.resume:
        summary = json.loads((output_dir / "summary.json").read_text())
        summary.setdefault("tokens", {"input": 0, "output": 0, "total": 0})
        summary.setdefault("tool_calls", 0)
        summary.setdefault("trials", [])
        summary["resume_count"] = int(summary.get("resume_count") or 0) + 1
        summary["last_resumed_at"] = datetime.now(timezone.utc).isoformat()
        summary["resume_config_path"] = str(config_path)
        summary["resume_config_overrides"] = config_overrides
        summary["llm"] = args.llm
        summary["reasoning"] = args.reasoning
    else:
        summary: dict[str, Any] = {
            "experiment": experiment_name,
            "config_path": str(config_path),
            "config_overrides": config_overrides,
            "llm": args.llm,
            "reasoning": args.reasoning,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "total_time_seconds": 0.0,
            "tokens": {"input": 0, "output": 0, "total": 0},
            "tool_calls": 0,
            "trials": [],
        }
    starting_total_time = float(summary.get("total_time_seconds") or 0.0)
    # Completed trials are in summary["trials"]; check for a partial trial checkpoint
    # at the next trial index so we can resume mid-trial after a crash.
    completed_trial_count = len(summary.get("trials", []))
    start_trial_index = completed_trial_count

    run_started = time.perf_counter()
    sandbox = Sandbox(
        sandbox_path=args.sandbox_path,
        network_access=config.environment.permissions.network,
    )
    logger.log(
        "experiment_resume_start" if args.resume else "experiment_start",
        config=config.model_dump(),
        config_overrides=config_overrides,
        llm=args.llm,
        reasoning=args.reasoning,
        resume=args.resume,
        start_trial_index=start_trial_index,
        additional_trials=config.task.trials,
    )
    summary["total_time_seconds"] = (
        starting_total_time + time.perf_counter() - run_started
    )
    summary["last_updated_at"] = datetime.now(timezone.utc).isoformat()
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    for trial_offset in range(config.task.trials):
        trial_index = start_trial_index + trial_offset

        # Resume mid-trial if a checkpoint exists for this trial index.
        checkpoint = _load_trial_checkpoint(output_dir, trial_index) if args.resume else None
        resumed_results_by_iteration: list[dict[str, CheckResult]] = []
        if checkpoint and checkpoint.get("iterations"):
            # Restore sandbox: seed initial files first (example code etc. excluded
            # from snapshots), then overlay the agent's snapshot on top.
            last_iter = len(checkpoint["iterations"]) - 1
            snapshot_dir = (
                output_dir / f"trial_{trial_index}" / f"iter_{last_iter}" / "sandbox_snapshot"
            )
            if snapshot_dir.exists():
                trial_summary = checkpoint
                start_iteration_index = len(trial_summary["iterations"])
                resumed_results_by_iteration = _checkpoint_results_by_iteration(
                    trial_summary
                )
                logger.log(
                    "trial_resume",
                    trial=trial_index,
                    resuming_from_iteration=start_iteration_index,
                )
                prepare_sandbox(config, sandbox, config_path)
                for src_file in sorted(snapshot_dir.rglob("*")):
                    if not src_file.is_file():
                        continue
                    rel = src_file.relative_to(snapshot_dir)
                    dst = Path(sandbox.sandbox_path) / rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_file, dst)
            else:
                logger.log(
                    "trial_checkpoint_ignored",
                    trial=trial_index,
                    reason=(
                        "Partial trial checkpoint has no sandbox_snapshot. "
                        "Restarting this trial from iteration 0 because "
                        "--snapshot-sandbox is disabled by default."
                    ),
                    checkpoint_iterations=len(checkpoint["iterations"]),
                )
                prepare_sandbox(config, sandbox, config_path)
                trial_summary = {"trial": trial_index, "iterations": []}
                start_iteration_index = 0
                logger.log("trial_start", trial=trial_index)
        else:
            prepare_sandbox(config, sandbox, config_path)
            trial_summary = {"trial": trial_index, "iterations": []}
            start_iteration_index = 0
            logger.log("trial_start", trial=trial_index)

        state = RunState(
            config=config,
            sandbox=sandbox,
            trial_index=trial_index,
            iteration_index=start_iteration_index,
            metadata={"output_dir": str(output_dir)},
            results_by_iteration=resumed_results_by_iteration,
        )

        for iteration_index in range(start_iteration_index, config.task.iterations):
            state.iteration_index = iteration_index
            messages = build_iteration_messages(
                state=state,
                config=config,
                checks=checks,
                trial_summary=trial_summary,
                iteration_index=iteration_index,
            )
            try:
                final_inputs = run_single_iteration(
                    llm=llm,
                    state=state,
                    checks=checks,
                    logger=logger,
                    summary=summary,
                    trial_summary=trial_summary,
                    messages=messages,
                )
            except LLMAPIError:
                summary["total_time_seconds"] = (
                    starting_total_time + time.perf_counter() - run_started
                )
                summary["ended_at"] = datetime.now(timezone.utc).isoformat()
                summary["status"] = "llm_error"
                summary["llm_error"] = traceback.format_exc()
                (output_dir / "summary.json").write_text(
                    json.dumps(summary, indent=2, default=str)
                )
                logger.log(
                    "llm_error",
                    trial=trial_index,
                    iteration=iteration_index,
                    error=traceback.format_exc(),
                )
                sandbox.clean()
                raise
            except Exception:
                logger.log(
                    "iteration_error",
                    trial=trial_index,
                    iteration=iteration_index,
                    error=traceback.format_exc(),
                )
                final_inputs = {}
            iteration_notes = final_notes(final_inputs)
            final_inputs.pop("notes", None)
            try:
                results = run_checks(
                    state=state,
                    checks=checks,
                    final_inputs=final_inputs,
                    logger=logger,
                )
            except ExperimentSetupError:
                summary["total_time_seconds"] = (
                    starting_total_time + time.perf_counter() - run_started
                )
                summary["ended_at"] = datetime.now(timezone.utc).isoformat()
                summary["status"] = "setup_error"
                summary["setup_error"] = traceback.format_exc()
                (output_dir / "summary.json").write_text(
                    json.dumps(summary, indent=2, default=str)
                )
                logger.log("experiment_setup_error", summary=summary)
                sandbox.clean()
                raise
            state.results_by_iteration.append(results)
            _write_check_summaries_to_iter_result(
                output_dir, trial_index, iteration_index, results, state.config
            )
            result_dump = {}
            for check_config in state.config.checks:
                result = results[check_config.name]
                dumped = result.model_dump()
                if not check_config.feedback:
                    dumped["feedback"] = None
                result_dump[check_config.name] = dumped
            trial_summary["iterations"].append(
                {
                    "iteration": iteration_index,
                    "inputs": final_inputs,
                    "notes": iteration_notes,
                    "checks": result_dump,
                    "sandbox_files_after_iteration": sandbox_file_inventory(state),
                }
            )
            if args.snapshot_sandbox:
                snapshot_dir = (
                    output_dir
                    / f"trial_{trial_index}"
                    / f"iter_{iteration_index}"
                    / "sandbox_snapshot"
                )
                snapshot_sandbox(sandbox.sandbox_path, snapshot_dir)
            _save_trial_checkpoint(output_dir, trial_index, trial_summary)
        summary["trials"].append(trial_summary)
        logger.log("trial_end", trial=trial_index, summary=trial_summary)
        summary["total_time_seconds"] = (
            starting_total_time + time.perf_counter() - run_started
        )
        summary["last_updated_at"] = datetime.now(timezone.utc).isoformat()
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, default=str)
        )
        sandbox.clean()

    summary["total_time_seconds"] = (
        starting_total_time + time.perf_counter() - run_started
    )
    summary["ended_at"] = datetime.now(timezone.utc).isoformat()
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    logger.log(
        "experiment_resume_end" if args.resume else "experiment_end", summary=summary
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a hardware-in-the-loop agent experiment."
    )
    parser.add_argument("config", help="Path to an experiment YAML config.")
    parser.add_argument(
        "config_overrides",
        nargs="*",
        help=(
            "Optional KEY VALUE config overrides using dotted keys, e.g. "
            "task.trials 1 task.iterations 2. Values are parsed as YAML scalars."
        ),
    )
    parser.add_argument(
        "--llm", required=True, help="LLM in provider/model format, e.g. openai/gpt-4.1"
    )
    parser.add_argument(
        "--reasoning",
        default=None,
        help="Optional reasoning effort hint for providers that support it.",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Override outputs/<experiment> directory name.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Write run logs/results to this directory. Relative paths are resolved from the repo root.",
    )
    parser.add_argument(
        "--sandbox-path", default=DEFAULT_SANDBOX_PATH, help="Sandbox host directory."
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Override task.iterations from the config.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=None,
        help="Override task.trials from the config.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace an existing output directory."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Append task.trials additional trials to an existing output directory. "
            "Requires outputs/<experiment>/summary.json and preserves run.log."
        ),
    )
    parser.add_argument(
        "--snapshot-sandbox",
        action="store_true",
        help=(
            "Copy sandbox source files to outputs/<experiment>/trial_*/iter_*/"
            "sandbox_snapshot after each iteration. Disabled by default because "
            "snapshots can use substantial storage."
        ),
    )
    return parser.parse_args()


def main() -> None:
    run_experiment(parse_args())


if __name__ == "__main__":
    main()
