#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/edgedl-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/edgedl-cache")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUTS_DIR = REPO_ROOT / "outputs"
FEEDBACK_FLOAT_RE = re.compile(r"([A-Za-z0-9_]+)=(-?\d+(?:\.\d+)?)")
EXPERIMENT_DISPLAY_ORDER = {
    "compression-no-feedback": 0,
    "compression-documentation": 1,
    "compression-feedback": 2,
}
EXPERIMENT_DISPLAY_NAMES = {
    "compression-no-feedback": "No Feedback",
    "compression-documentation": "Documentation",
    "compression-feedback": "Feedback",
}
HUMAN_BASELINES = {
    "max78000": ("mAP50-95", 0.14147799729340133),
    "stm32": ("PAR", 0.5144761043641971),
}
PLOT_EXPERIMENT_FAMILIES = [
    "compression-no-feedback",
    "compression-documentation",
    "compression-feedback",
]
MODEL_DISPLAY_NAMES = {
    "claude-opus-4-7": "Claude Opus 4.7",
    "claude-sonnet-4-6": "Claude Sonnet 4.6",
    "gpt-5.4": "GPT-5.4",
    "gpt-5.4-mini": "GPT-5.4 mini",
    "gemini-3.1-pro-preview": "Gemini 3.1 Pro",
    "gemini-3-flash-preview": "Gemini 3 Flash",
}
MODEL_COLORS = {
    "claude-opus-4-7": "#c51f1a",
    "claude-sonnet-4-6": "#ec5a57",
    "gpt-5.4": "#238b45",
    "gpt-5.4-mini": "#55a868",
    "gemini-3.1-pro-preview": "#1f78d1",
    "gemini-3-flash-preview": "#6aa4ed",
}
MODEL_DISPLAY_ORDER = {key: index for index, key in enumerate(MODEL_DISPLAY_NAMES)}
EXPERIMENT_TRACE_STYLES = {
    "compression-feedback": ("-", "o", "Feedback"),
    "compression-documentation": ("--", "s", "Documentation"),
    "compression-no-feedback": (":", "^", "No Feedback"),
}

CSV_COLUMNS = [
    "experiment",
    "model",
    "reasoning",
    "trials",
    "first_success_avg_check_score",
    "fully_successful_iterations",
    "peak_gpu_memory_mb",
    "peak_gpu_power_watts",
    "peak_gpu_utilization_percent",
    "max_context_size",
    "time_used_seconds",
    "tokens_used",
    "tool_calls",
    "min_epochs_used_for_training",
    "max_epochs_used_for_training",
    "passing_gradient_flow_iterations",
    "passing_train_iterations",
    "passing_synthesis_iterations",
    "best_training_score_from_passing_iteration",
    "first_training_score_from_passing_iteration",
]
ITERATION_CSV_COLUMNS = [
    "experiment",
    "config",
    "task",
    "model",
    "thinking_lelvel",
    "trial",
    "iteration",
    "all_success",
    "train_score",
    "tok_input",
    "tok_output",
    "tok_total",
    "time_s",
    "tool_calls",
]


def parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def mean(values: Sequence[float | None]) -> float | None:
    clean = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    if not clean:
        return None
    return float(np.asarray(clean, dtype=float).mean())


def sem(values: list[float]) -> float:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if len(clean) <= 1:
        return 0.0
    return float(np.asarray(clean, dtype=float).std(ddof=1) / math.sqrt(len(clean)))


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "unknown"


def target_device(experiment: str) -> str:
    lowered = experiment.lower()
    if "stm32" in lowered or "nucleo" in lowered:
        return "stm32"
    if "max78000" in lowered:
        return "max78000"
    return "other"


def experiment_family(experiment: str) -> str:
    if experiment.startswith("compression-no-feedback"):
        return "compression-no-feedback"
    if experiment.startswith("compression-documentation"):
        return "compression-documentation"
    if experiment.startswith("compression-feedback"):
        return "compression-feedback"
    return experiment


def experiment_display_name(experiment: str) -> str:
    return EXPERIMENT_DISPLAY_NAMES.get(experiment_family(experiment), experiment)


def experiment_sort_key(experiment: str) -> tuple[int, str, str]:
    family = experiment_family(experiment)
    return (
        EXPERIMENT_DISPLAY_ORDER.get(family, 100),
        experiment_display_name(experiment),
        experiment,
    )


def model_plot_label(row: dict[str, Any]) -> str:
    model = str(row.get("model") or "unknown").split("/")[-1]
    reasoning = row.get("reasoning")
    return f"{model} [{reasoning}]" if reasoning else model


def model_key(row: dict[str, Any]) -> str:
    return str(row.get("model") or "unknown").split("/")[-1]


def model_display_base(row: dict[str, Any]) -> str:
    key = model_key(row)
    return MODEL_DISPLAY_NAMES.get(key, key)


def model_display_label(row: dict[str, Any], duplicate_models: set[str]) -> str:
    base = model_display_base(row)
    reasoning = str(row.get("reasoning") or "")
    if model_key(row) in duplicate_models and reasoning:
        return f"{base} [{reasoning}]"
    return base


def model_color(row_or_key: dict[str, Any] | str) -> str:
    key = model_key(row_or_key) if isinstance(row_or_key, dict) else row_or_key
    return MODEL_COLORS.get(key, "#555555")


def model_sort_key(label: str, key: str) -> tuple[int, str]:
    return MODEL_DISPLAY_ORDER.get(key, 100), label


def device_display_name(device: str) -> str:
    return {
        "stm32": "STM32N6",
        "max78000": "MAX78000",
        "other": "Other",
    }.get(device, device)


def plot_devices_from_experiments(experiments: list[str]) -> list[str]:
    order = {"max78000": 0, "stm32": 1, "other": 2}
    return sorted(
        {target_device(experiment) for experiment in experiments},
        key=lambda device: (order.get(device, 100), device),
    )


def model_labels_with_success(trial_rows: list[dict[str, Any]]) -> list[str]:
    labels: set[str] = set()
    for row in trial_rows:
        value = row.get("fully_successful_iterations")
        if (
            isinstance(value, int | float)
            and math.isfinite(float(value))
            and float(value) > 0
        ):
            labels.add(model_plot_label(row))
    return sorted(labels)


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if not path.exists():
        return events
    with path.open(errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                events.append(event)
    return events


def experiment_name(summary: dict[str, Any], output_dir: Path) -> str:
    config_path = summary.get("config_path")
    if isinstance(config_path, str) and config_path:
        return Path(config_path).stem
    raw = str(summary.get("experiment") or output_dir.name)
    return raw.split("__", 1)[0]


def check_success(iteration: dict[str, Any], predicate) -> bool:
    checks = iteration.get("checks") or {}
    for name, result in checks.items():
        if predicate(str(name)) and bool((result or {}).get("success")):
            return True
    return False


def all_checks_pass(iteration: dict[str, Any]) -> bool:
    checks = iteration.get("checks") or {}
    return bool(checks) and all(
        bool((result or {}).get("success")) for result in checks.values()
    )


def train_result(iteration: dict[str, Any]) -> dict[str, Any]:
    return (iteration.get("checks") or {}).get("train.py") or {}


def average_check_score(iteration: dict[str, Any]) -> float | None:
    scores: list[float] = []
    for result in (iteration.get("checks") or {}).values():
        score = (result or {}).get("score")
        if isinstance(score, int | float) and math.isfinite(float(score)):
            scores.append(float(score))
    return mean(scores)


def train_score(iteration: dict[str, Any]) -> float | None:
    score = train_result(iteration).get("score")
    if isinstance(score, int | float) and math.isfinite(float(score)):
        return float(score)
    return None


def train_epochs(iteration: dict[str, Any]) -> float | None:
    train_input = (iteration.get("inputs") or {}).get("train.py") or {}
    epochs = train_input.get("epochs")
    if isinstance(epochs, int | float) and math.isfinite(float(epochs)):
        return float(epochs)
    return None


def numeric_feedback_fields(feedback: str | None) -> dict[str, float]:
    fields: dict[str, float] = {}
    if not isinstance(feedback, str) or not feedback:
        return fields
    for key, value in FEEDBACK_FLOAT_RE.findall(feedback):
        try:
            fields[key] = float(value)
        except ValueError:
            continue
    return fields


def update_gpu_metrics(
    metrics: dict[str, float | None], iteration: dict[str, Any]
) -> None:
    feedback = train_result(iteration).get("feedback")
    fields = numeric_feedback_fields(feedback if isinstance(feedback, str) else None)
    mapping = {
        "peak_gpu_memory_mb": ("gpu_peak_memory_mb", "peak_gpu_memory_mb"),
        "peak_gpu_power_watts": ("gpu_peak_power_watts", "peak_gpu_power_watts"),
        "peak_gpu_utilization_percent": (
            "gpu_peak_utilization_percent",
            "peak_gpu_utilization_percent",
        ),
    }
    for metric, keys in mapping.items():
        values = [fields[key] for key in keys if key in fields]
        if not values:
            continue
        value = max(values)
        current = metrics.get(metric)
        metrics[metric] = value if current is None else max(float(current), value)


def usage_value(usage: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = usage.get(key)
        if isinstance(value, int | float) and math.isfinite(float(value)):
            return float(value)
    return None


def parse_trial_event_metrics(
    events: list[dict[str, Any]],
) -> dict[int, dict[str, float | None]]:
    rows: dict[int, dict[str, float | None]] = defaultdict(
        lambda: {
            "max_context_size": None,
            "tokens_used": 0.0,
            "tool_calls": 0.0,
            "time_used_seconds": None,
            "peak_gpu_memory_mb": None,
            "peak_gpu_power_watts": None,
            "peak_gpu_utilization_percent": None,
        }
    )
    starts: dict[int, datetime] = {}
    ends: dict[int, datetime] = {}
    for event in events:
        trial_value = event.get("trial")
        if trial_value is None:
            continue
        try:
            trial = int(trial_value)
        except (TypeError, ValueError):
            continue
        event_name = event.get("event")
        timestamp = parse_timestamp(event.get("timestamp"))
        if event_name == "trial_start" and timestamp is not None:
            starts[trial] = timestamp
        elif event_name == "trial_end" and timestamp is not None:
            ends[trial] = timestamp
        elif event_name == "tool_call":
            rows[trial]["tool_calls"] = float(rows[trial]["tool_calls"] or 0.0) + 1.0
        elif event_name == "llm_call":
            usage = event.get("usage") or {}
            if not isinstance(usage, dict):
                continue
            input_tokens = usage_value(usage, "input_tokens", "prompt_tokens")
            total_tokens = usage_value(usage, "total_tokens")
            if total_tokens is None:
                prompt = usage_value(usage, "input_tokens", "prompt_tokens") or 0.0
                completion = (
                    usage_value(usage, "output_tokens", "completion_tokens") or 0.0
                )
                total_tokens = prompt + completion
            if input_tokens is not None:
                current = rows[trial]["max_context_size"]
                rows[trial]["max_context_size"] = (
                    input_tokens
                    if current is None
                    else max(float(current), input_tokens)
                )
            rows[trial]["tokens_used"] = (
                float(rows[trial]["tokens_used"] or 0.0) + total_tokens
            )
        elif event_name == "check" and event.get("name") == "train.py":
            result = event.get("result") or {}
            if not isinstance(result, dict):
                continue
            fields = numeric_feedback_fields(result.get("feedback"))
            mapping = {
                "peak_gpu_memory_mb": ("gpu_peak_memory_mb", "peak_gpu_memory_mb"),
                "peak_gpu_power_watts": (
                    "gpu_peak_power_watts",
                    "peak_gpu_power_watts",
                ),
                "peak_gpu_utilization_percent": (
                    "gpu_peak_utilization_percent",
                    "peak_gpu_utilization_percent",
                ),
            }
            for metric, keys in mapping.items():
                values = [fields[key] for key in keys if key in fields]
                if values:
                    current = rows[trial][metric]
                    rows[trial][metric] = max(
                        values if current is None else [float(current), *values]
                    )
    for trial, start in starts.items():
        end = ends.get(trial)
        if end is not None:
            rows[trial]["time_used_seconds"] = (end - start).total_seconds()
    return rows


def task_name_from_events(events: list[dict[str, Any]]) -> str:
    for event in events:
        if event.get("event") not in {"experiment_start", "experiment_resume_start"}:
            continue
        config = event.get("config") or {}
        if not isinstance(config, dict):
            continue
        task = config.get("task") or {}
        if isinstance(task, dict) and isinstance(task.get("name"), str):
            return task["name"]
    return ""


def parse_iteration_event_metrics(
    events: list[dict[str, Any]],
) -> dict[tuple[int, int], dict[str, float]]:
    rows: dict[tuple[int, int], dict[str, float]] = defaultdict(
        lambda: {
            "tok_input": 0.0,
            "tok_output": 0.0,
            "tok_total": 0.0,
            "time_s": 0.0,
            "tool_calls": 0.0,
        }
    )
    for event in events:
        trial_value = event.get("trial")
        iteration_value = event.get("iteration")
        if trial_value is None or iteration_value is None:
            continue
        try:
            key = (int(trial_value), int(iteration_value))
        except (TypeError, ValueError):
            continue
        event_name = event.get("event")
        elapsed = event.get("elapsed_seconds")
        if isinstance(elapsed, int | float) and math.isfinite(float(elapsed)):
            rows[key]["time_s"] += float(elapsed)
        if event_name == "tool_call":
            rows[key]["tool_calls"] += 1.0
        elif event_name == "llm_call":
            usage = event.get("usage") or {}
            if not isinstance(usage, dict):
                continue
            input_tokens = usage_value(usage, "input_tokens", "prompt_tokens") or 0.0
            output_tokens = (
                usage_value(usage, "output_tokens", "completion_tokens") or 0.0
            )
            total_tokens = usage_value(usage, "total_tokens")
            if total_tokens is None:
                total_tokens = input_tokens + output_tokens
            if not output_tokens and total_tokens >= input_tokens:
                output_tokens = total_tokens - input_tokens
            rows[key]["tok_input"] += input_tokens
            rows[key]["tok_output"] += output_tokens
            rows[key]["tok_total"] += total_tokens
    return rows


def parse_run(
    output_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    summary = load_json(output_dir / "summary.json")
    if summary is None:
        return [], [], []
    events = load_jsonl(output_dir / "run.log")
    event_metrics = parse_trial_event_metrics(events)
    iteration_event_metrics = parse_iteration_event_metrics(events)
    experiment = experiment_name(summary, output_dir)
    config_stem = (
        Path(str(summary.get("config_path"))).stem
        if summary.get("config_path")
        else experiment
    )
    task_name = task_name_from_events(events)
    model = str(summary.get("llm") or "unknown")
    reasoning = (
        "" if summary.get("reasoning") is None else str(summary.get("reasoning"))
    )

    trial_rows: list[dict[str, Any]] = []
    passing_train_points: list[dict[str, Any]] = []
    iteration_rows: list[dict[str, Any]] = []
    for trial_summary in summary.get("trials") or []:
        if not isinstance(trial_summary, dict):
            continue
        trial = int(trial_summary.get("trial") or 0)
        iterations = [
            item
            for item in trial_summary.get("iterations") or []
            if isinstance(item, dict)
        ]
        fully_successful = [item for item in iterations if all_checks_pass(item)]
        first_success = fully_successful[0] if fully_successful else None
        first_success_iteration = (
            int(first_success.get("iteration") or 0) + 1 if first_success else None
        )
        passing_train_scores = [
            score
            for item in fully_successful
            for score in [train_score(item)]
            if score is not None
        ]
        epochs = [
            value
            for item in iterations
            for value in [train_epochs(item)]
            if value is not None
        ]
        gpu_metrics: dict[str, float | None] = {
            "peak_gpu_memory_mb": None,
            "peak_gpu_power_watts": None,
            "peak_gpu_utilization_percent": None,
        }
        best_passing_score_so_far: float | None = None
        max_iteration_index = max(
            (int(item.get("iteration") or 0) for item in iterations),
            default=-1,
        )
        points_by_iteration: dict[int, dict[str, Any]] = {}
        for iteration in iterations:
            update_gpu_metrics(gpu_metrics, iteration)
            iteration_number = int(iteration.get("iteration") or 0) + 1
            iteration_index = iteration_number - 1
            current_train_score = train_score(iteration)
            fully_successful_iteration = all_checks_pass(iteration)
            per_iteration_metrics = iteration_event_metrics.get(
                (trial, iteration_index), {}
            )
            iteration_rows.append(
                {
                    "experiment": output_dir.name,
                    "config": config_stem,
                    "task": task_name,
                    "model": model,
                    "thinking_lelvel": reasoning,
                    "trial": trial,
                    "iteration": iteration_number,
                    "all_success": fully_successful_iteration,
                    "train_score": current_train_score,
                    "tok_input": per_iteration_metrics.get("tok_input", 0.0),
                    "tok_output": per_iteration_metrics.get("tok_output", 0.0),
                    "tok_total": per_iteration_metrics.get("tok_total", 0.0),
                    "time_s": per_iteration_metrics.get("time_s", 0.0),
                    "tool_calls": per_iteration_metrics.get("tool_calls", 0.0),
                }
            )
            if fully_successful_iteration:
                score = current_train_score
                if score is not None:
                    best_passing_score_so_far = (
                        score
                        if best_passing_score_so_far is None
                        else max(best_passing_score_so_far, score)
                    )
            if best_passing_score_so_far is not None or current_train_score is not None:
                points_by_iteration[iteration_number] = {
                    "experiment": experiment,
                    "model": model,
                    "reasoning": reasoning,
                    "trial": trial,
                    "iteration": iteration_number,
                    "training_score": best_passing_score_so_far,
                    "current_training_score": current_train_score,
                    "fully_successful": fully_successful_iteration,
                }
        for iteration_number in range(1, max_iteration_index + 2):
            point = points_by_iteration.get(iteration_number)
            if point is not None:
                passing_train_points.append(point)
        row = {
            "experiment": experiment,
            "model": model,
            "reasoning": reasoning,
            "trial": trial,
            "first_success_iteration": (
                float(first_success_iteration)
                if first_success_iteration is not None
                else None
            ),
            "first_success_avg_check_score": (
                average_check_score(first_success) if first_success else None
            ),
            "fully_successful_iterations": float(len(fully_successful)),
            "min_epochs_used_for_training": min(epochs) if epochs else None,
            "max_epochs_used_for_training": max(epochs) if epochs else None,
            "_train_epoch_values": epochs,
            "passing_gradient_flow_iterations": float(
                sum(
                    1
                    for item in iterations
                    if check_success(item, lambda name: name == "gradient_flow.py")
                )
            ),
            "passing_train_iterations": float(
                sum(
                    1
                    for item in iterations
                    if check_success(item, lambda name: name == "train.py")
                )
            ),
            "passing_synthesis_iterations": float(
                sum(
                    1
                    for item in iterations
                    if check_success(item, lambda name: name.startswith("synthesis_"))
                )
            ),
            "best_training_score_from_passing_iteration": (
                max(passing_train_scores) if passing_train_scores else None
            ),
            "first_training_score_from_passing_iteration": (
                passing_train_scores[0] if passing_train_scores else None
            ),
        }
        row.update(gpu_metrics)
        for key, value in event_metrics.get(trial, {}).items():
            if key.startswith("peak_gpu_"):
                if isinstance(value, int | float):
                    current = row.get(key)
                    row[key] = (
                        float(value)
                        if current is None
                        else max(float(current), float(value))
                    )
            else:
                row[key] = value
        if (
            row.get("tokens_used") in (None, 0.0)
            and len(summary.get("trials") or []) == 1
        ):
            tokens = summary.get("tokens") or {}
            if isinstance(tokens, dict) and isinstance(
                tokens.get("total"), int | float
            ):
                row["tokens_used"] = float(tokens["total"])
        if (
            row.get("tool_calls") in (None, 0.0)
            and len(summary.get("trials") or []) == 1
        ):
            if isinstance(summary.get("tool_calls"), int | float):
                row["tool_calls"] = float(summary["tool_calls"])
        if (
            row.get("time_used_seconds") is None
            and len(summary.get("trials") or []) == 1
        ):
            if isinstance(summary.get("total_time_seconds"), int | float):
                row["time_used_seconds"] = float(summary["total_time_seconds"])
        trial_rows.append(row)
    return trial_rows, passing_train_points, iteration_rows


def group_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return str(row["experiment"]), str(row["model"]), str(row["reasoning"])


def aggregate_rows(trial_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in trial_rows:
        grouped[group_key(row)].append(row)

    aggregate: list[dict[str, Any]] = []
    metric_columns = [
        column
        for column in CSV_COLUMNS
        if column not in {"experiment", "model", "reasoning", "trials"}
    ]
    for (experiment, model, reasoning), rows in sorted(grouped.items()):
        output = {
            "experiment": experiment,
            "model": model,
            "reasoning": reasoning,
            "trials": len(rows),
        }
        for column in metric_columns:
            output[column] = mean([row.get(column) for row in rows])
        aggregate.append(output)
    return aggregate


def write_csv(
    path: Path, rows: list[dict[str, Any]], columns: list[str] = CSV_COLUMNS
) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    column: "" if row.get(column) is None else row.get(column)
                    for column in columns
                }
            )


def collect_grouped_metric(
    trial_rows: list[dict[str, Any]], metric: str
) -> tuple[
    list[str], list[str], dict[tuple[str, str], list[float]], dict[str, list[float]]
]:
    experiments = sorted(
        {str(row["experiment"]) for row in trial_rows}, key=experiment_sort_key
    )
    models = sorted({model_plot_label(row) for row in trial_rows})
    by_model: dict[tuple[str, str], list[float]] = defaultdict(list)
    overall: dict[str, list[float]] = defaultdict(list)
    for row in trial_rows:
        value = row.get(metric)
        if isinstance(value, int | float) and math.isfinite(float(value)):
            experiment = str(row["experiment"])
            model = model_plot_label(row)
            by_model[(experiment, model)].append(float(value))
            overall[experiment].append(float(value))
    return experiments, models, by_model, overall


def grouped_bar_chart(
    trial_rows: list[dict[str, Any]],
    *,
    metric: str,
    title: str,
    ylabel: str,
    output_path: Path,
    include_overall: bool = True,
) -> None:
    devices = plot_devices_from_experiments(
        [str(row["experiment"]) for row in trial_rows]
    )
    model_columns = model_labels_with_success(trial_rows)
    if include_overall:
        model_columns = ["Overall", *model_columns]
    if not model_columns:
        model_columns = ["No successful models"]
    fig, axes = plt.subplots(
        len(devices),
        len(model_columns),
        figsize=(4.2 * len(model_columns), max(5.0, 4.2 * len(devices))),
        squeeze=False,
        sharey="row",
    )
    families = PLOT_EXPERIMENT_FAMILIES
    for row_idx, device in enumerate(devices):
        for col_idx, column_label in enumerate(model_columns):
            ax = axes[row_idx, col_idx]
            heights: list[float] = []
            errors: list[float] = []
            failed_bars: list[bool] = []
            for family in families:
                values: list[float] = []
                for row in trial_rows:
                    if target_device(str(row["experiment"])) != device:
                        continue
                    if experiment_family(str(row["experiment"])) != family:
                        continue
                    if (
                        column_label != "Overall"
                        and model_plot_label(row) != column_label
                    ):
                        continue
                    value = row.get(metric)
                    if isinstance(value, int | float) and math.isfinite(float(value)):
                        values.append(float(value))
                avg = mean(values)
                if avg is None and metric == "first_success_iteration":
                    heights.append(0.0)
                    failed_bars.append(True)
                else:
                    heights.append(np.nan if avg is None else avg)
                    failed_bars.append(False)
                errors.append(sem(values))
            x = np.arange(len(families), dtype=float)
            ax.bar(x, heights, width=0.58, yerr=errors, capsize=3)
            ax.set_title(f"{device_display_name(device)} / {column_label}")
            if col_idx == 0:
                ax.set_ylabel(ylabel)
            ax.set_xticks(x)
            ax.set_xticklabels(
                [EXPERIMENT_DISPLAY_NAMES[family] for family in families],
                rotation=0,
                ha="center",
            )
            ax.grid(axis="y", alpha=0.25)
            finite_heights = [
                float(value) for value in heights if math.isfinite(float(value))
            ]
            no_success = (
                not finite_heights
                or (
                    metric == "fully_successful_iterations"
                    and max(finite_heights, default=0.0) <= 0.0
                )
                or (
                    metric == "first_success_iteration"
                    and failed_bars
                    and all(failed_bars)
                )
            )
            if no_success:
                ax.text(
                    0.5,
                    0.5,
                    "Checks never all pass",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                )
            else:
                for bar_x, failed in zip(x, failed_bars, strict=True):
                    if failed:
                        ax.text(
                            bar_x,
                            0.03,
                            "Failed Checks",
                            rotation=90,
                            transform=ax.get_xaxis_transform(),
                            ha="center",
                            va="bottom",
                            fontsize="small",
                            color="dimgray",
                        )
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def scale_test_score(score: float) -> float:
    return score * 100.0


def plot_test_score(score: float, device: str) -> float:
    if device == "stm32":
        return scale_test_score(1.0 - score)
    return scale_test_score(score)


def score_axis_label(device: str) -> str:
    if device == "stm32":
        return "PER"
    if device in HUMAN_BASELINES:
        return HUMAN_BASELINES[device][0]
    return "Test score"


def human_baseline_for_plot(device: str) -> float | None:
    if device == "stm32":
        return None
    if device not in HUMAN_BASELINES:
        return None
    _metric, baseline = HUMAN_BASELINES[device]
    return plot_test_score(float(baseline), device)


def score_axis_limits(
    device: str, y_values: list[float]
) -> tuple[float, float, float, bool]:
    if device == "stm32" and y_values:
        high_values = [value for value in y_values if value >= 50.0]
        if high_values:
            y_min = max(0.0, math.floor(min(high_values) / 5.0) * 5.0 - 5.0)
            y_min = min(y_min, 90.0)
            y_max = max(101.5, math.ceil(max(high_values) / 5.0) * 5.0 + 1.5)
            return y_min, y_max, 5.0, True
    y_max = max(20.0, math.ceil((max(y_values) if y_values else 0.0) / 5.0) * 5.0)
    y_max = min(max(y_max, 20.0), 100.0)
    y_step = 20.0 if y_max > 40.0 else 5.0
    return -1.0, y_max, y_step, False


def draw_y_axis_break(ax: Any, y_center: float, y_step: float) -> None:
    slash_height = y_step * 0.36
    slash_gap = y_step * 0.24
    ax.plot(
        [0.0, 0.0],
        [y_center - slash_gap / 2.0, y_center + slash_gap / 2.0],
        transform=ax.get_yaxis_transform(),
        color="white",
        linewidth=3.2,
        solid_capstyle="butt",
        clip_on=False,
        zorder=8,
    )
    for center in (y_center - slash_gap / 2.0, y_center + slash_gap / 2.0):
        ax.plot(
            [-0.016, 0.016],
            [center - slash_height / 2.0, center + slash_height / 2.0],
            transform=ax.get_yaxis_transform(),
            color="black",
            linewidth=1.4,
            clip_on=False,
            zorder=9,
        )


def reasoning_rows(points: list[dict[str, Any]]) -> list[str]:
    present = {str(point.get("reasoning") or "") for point in points}
    ordered = [reasoning for reasoning in ("high", "low") if reasoning in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered or [""]


def training_score_line_chart(points: list[dict[str, Any]], output_path: Path) -> None:
    devices = plot_devices_from_experiments(
        [str(point["experiment"]) for point in points]
    )
    if not devices:
        return
    max_iteration = max(
        (
            int(point["iteration"])
            for point in points
            if isinstance(point.get("iteration"), int)
        ),
        default=0,
    )
    reasoning_values = reasoning_rows(points)
    fig, axes = plt.subplots(
        len(reasoning_values),
        len(devices),
        figsize=(8.6 * len(devices), 4.0 * len(reasoning_values)),
        squeeze=False,
    )
    duplicate_models: set[str] = set()

    seen_model_handles: dict[tuple[str, str], Line2D] = {}
    seen_family_handles: dict[str, Line2D] = {}

    for row_idx, reasoning in enumerate(reasoning_values):
        for col_idx, device in enumerate(devices):
            ax = axes[row_idx][col_idx]
            device_points = [
                point
                for point in points
                if target_device(str(point["experiment"])) == device
                and str(point.get("reasoning") or "") == reasoning
            ]
            panel_index = row_idx * len(devices) + col_idx
            panel_letter = chr(ord("a") + panel_index)

            y_values: list[float] = []
            series: dict[tuple[str, str, str, str], dict[int, list[float]]] = (
                defaultdict(lambda: defaultdict(list))
            )
            for point in device_points:
                family = experiment_family(str(point["experiment"]))
                if family not in EXPERIMENT_TRACE_STYLES:
                    continue
                iteration = point.get("iteration")
                score = point.get("training_score")
                if isinstance(score, int | float) and isinstance(iteration, int):
                    if math.isfinite(float(score)):
                        value = plot_test_score(float(score), device)
                        label = model_display_label(point, duplicate_models)
                        key = (
                            label,
                            model_key(point),
                            str(point.get("reasoning") or ""),
                            family,
                        )
                        series[key][iteration].append(value)
                        y_values.append(value)
                current_score = point.get("current_training_score")
                if isinstance(current_score, int | float) and isinstance(
                    iteration, int
                ):
                    if math.isfinite(float(current_score)):
                        value = plot_test_score(float(current_score), device)
                        y_values.append(value)

            y_min, y_max, y_step, has_axis_break = score_axis_limits(device, y_values)

            for point in device_points:
                family = experiment_family(str(point["experiment"]))
                if family not in EXPERIMENT_TRACE_STYLES:
                    continue
                iteration = point.get("iteration")
                current_score = point.get("current_training_score")
                if not isinstance(iteration, int) or not isinstance(
                    current_score, int | float
                ):
                    continue
                if not math.isfinite(float(current_score)):
                    continue
                value = plot_test_score(float(current_score), device)
                linestyle, marker, family_label = EXPERIMENT_TRACE_STYLES[family]
                color = model_color(point)
                filled = bool(point.get("fully_successful"))
                ax.scatter(
                    [iteration],
                    [value],
                    marker=marker,
                    s=44,
                    facecolors=color if filled else "none",
                    edgecolors=color,
                    linewidths=1.25,
                    alpha=0.86 if filled else 0.42,
                    zorder=3 if filled else 2,
                )
                seen_family_handles.setdefault(
                    family,
                    Line2D(
                        [0],
                        [0],
                        color="#555555",
                        linestyle=linestyle,
                        marker=marker,
                        linewidth=2.0,
                        markersize=6.0,
                        label=family_label,
                    ),
                )

            for (label, key, _reasoning, family), points_by_iteration in sorted(
                series.items(),
                key=lambda item: (
                    model_sort_key(item[0][0], item[0][1]),
                    EXPERIMENT_DISPLAY_ORDER.get(item[0][3], 100),
                ),
            ):
                linestyle, marker, _family_label = EXPERIMENT_TRACE_STYLES[family]
                xs: list[int] = []
                ys: list[float] = []
                for iteration_number in sorted(points_by_iteration):
                    value = mean(points_by_iteration[iteration_number])
                    if value is not None:
                        xs.append(iteration_number)
                        ys.append(float(value))
                if not xs:
                    continue
                color = model_color(key)
                ax.plot(
                    xs,
                    ys,
                    color=color,
                    linestyle=linestyle,
                    marker=marker,
                    markersize=5.5,
                    linewidth=2.2,
                    drawstyle="steps-post",
                    zorder=4,
                )
                seen_model_handles.setdefault(
                    (label, key),
                    Line2D(
                        [0],
                        [0],
                        color=color,
                        linestyle="-",
                        marker="o",
                        linewidth=2.2,
                        markersize=5.5,
                        label=label,
                    ),
                )

            baseline_y = human_baseline_for_plot(device)
            if baseline_y is not None and 0.0 <= baseline_y <= y_max:
                ax.axhline(
                    baseline_y,
                    color="#6b4c8a",
                    linestyle=(0, (4, 4)),
                    linewidth=1.4,
                    zorder=1,
                )
                ax.text(
                    max_iteration + 0.18,
                    baseline_y,
                    "human",
                    color="#4b1f75",
                    fontsize=11,
                    fontweight="bold",
                    va="center",
                    clip_on=False,
                )

            ax.text(
                -0.15,
                1.03,
                panel_letter,
                transform=ax.transAxes,
                fontsize=20,
                fontweight="bold",
                ha="left",
                va="bottom",
            )
            ax.text(
                -0.105,
                1.0,
                r"$\downarrow$" if device == "stm32" else r"$\uparrow$",
                transform=ax.transAxes,
                fontsize=18,
                ha="center",
                va="top",
            )
            ax.set_title("")
            if col_idx == 0:
                ax.text(
                    -0.2,
                    0.5,
                    reasoning.capitalize() if reasoning else "Unspecified",
                    transform=ax.transAxes,
                    rotation=90,
                    fontsize=12,
                    fontweight="bold",
                    ha="center",
                    va="center",
                )
            ax.set_xlabel("Iteration #", fontsize=12, fontweight="bold")
            ax.set_ylabel(score_axis_label(device), fontsize=12, fontweight="bold")
            ax.set_xlim(0, max_iteration + 0.25 if max_iteration else 10.25)
            if max_iteration > 0:
                ax.set_xticks(list(range(0, max_iteration + 1)))
            ax.set_ylim(y_min, y_max)
            first_tick = max(0.0, math.ceil(y_min / y_step) * y_step)
            y_ticks = list(np.arange(first_tick, y_max + 0.1, y_step))
            ax.set_yticks(y_ticks)
            if has_axis_break:
                ax.set_yticklabels(
                    [
                        "0" if index == 0 else f"{tick:g}"
                        for index, tick in enumerate(y_ticks)
                    ]
                )
                draw_y_axis_break(ax, first_tick + y_step / 2.0, y_step)
            ax.grid(True, color="#d9d9d9", linestyle="--", linewidth=0.7, alpha=0.55)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_linewidth(1.0)
            ax.spines["bottom"].set_linewidth(1.0)
            ax.tick_params(axis="both", labelsize=11)
            if not series:
                ax.text(
                    0.5,
                    0.5,
                    "Checks never all pass",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                )

    ordered_model_handles = [
        handle
        for (_label, key), handle in sorted(
            seen_model_handles.items(),
            key=lambda item: model_sort_key(item[0][0], item[0][1]),
        )
    ]
    ordered_family_handles = [
        seen_family_handles[family]
        for family in PLOT_EXPERIMENT_FAMILIES[::-1]
        if family in seen_family_handles
    ]
    status_handles = [
        Line2D(
            [0],
            [0],
            color="#555555",
            marker="o",
            linestyle="None",
            markersize=6.5,
            label="Fully successful",
        ),
        Line2D(
            [0],
            [0],
            color="#888888",
            marker="o",
            markerfacecolor="none",
            linestyle="None",
            markersize=6.5,
            label="Not fully successful",
        ),
    ]
    spacer = Line2D([], [], linestyle="None", label="")
    handles = [
        *ordered_model_handles,
        spacer,
        *ordered_family_handles,
        spacer,
        *status_handles,
    ]
    fig.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(0.895, 0.5),
        frameon=False,
        fontsize=11,
        handlelength=2.2,
    )
    fig.subplots_adjust(
        left=0.08,
        right=0.86,
        bottom=0.1,
        top=0.94,
        wspace=0.28,
        hspace=0.42,
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_summary_md(
    path: Path,
    *,
    runs: list[Path],
    aggregate_rows_: list[dict[str, Any]],
    trial_rows: list[dict[str, Any]],
    passing_train_points: list[dict[str, Any]],
) -> None:
    total_trials = len(trial_rows)
    total_successes = sum(
        float(row.get("fully_successful_iterations") or 0.0) for row in trial_rows
    )
    best_by_device: dict[str, dict[str, Any]] = {}
    for row in aggregate_rows_:
        score = row.get("best_training_score_from_passing_iteration")
        if not isinstance(score, int | float) or not math.isfinite(float(score)):
            continue
        device = target_device(str(row.get("experiment") or ""))
        current = best_by_device.get(device)
        if current is None or float(score) > float(
            current["best_training_score_from_passing_iteration"]
        ):
            best_by_device[device] = row
    lines = [
        "# Experiment Results Summary",
        "",
        f"- Runs parsed: {len(runs)}",
        f"- Trial rows averaged: {total_trials}",
        f"- Fully successful iterations: {total_successes:g}",
        f"- Test-score iteration samples in line plot: {len(passing_train_points)}",
        "",
        "## Outputs",
        "",
        "- `results.csv`",
        "- `iterations.csv`",
        "- `iterations_until_first_success.png`",
        "- `fully_successful_iterations.png`",
        "- `test_score_by_iteration.png`",
    ]
    totals_by_model: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "time_used_seconds": 0.0,
            "tokens_used": 0.0,
            "tool_calls": 0.0,
            "train_epochs_sum": 0.0,
            "train_epochs_count": 0.0,
            "trials": 0.0,
        }
    )
    for row in trial_rows:
        label = model_plot_label(row)
        totals_by_model[label]["trials"] += 1.0
        for metric in ("time_used_seconds", "tokens_used", "tool_calls"):
            value = row.get(metric)
            if isinstance(value, int | float) and math.isfinite(float(value)):
                totals_by_model[label][metric] += float(value)
        for epoch in row.get("_train_epoch_values") or []:
            if isinstance(epoch, int | float) and math.isfinite(float(epoch)):
                totals_by_model[label]["train_epochs_sum"] += float(epoch)
                totals_by_model[label]["train_epochs_count"] += 1.0
    if totals_by_model:
        lines.extend(
            [
                "",
                "## Totals By Model",
                "",
                "| Model | Trials | Time Used (hours) | Tokens Used | Tool Calls | Avg Train Epochs |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for model, totals in sorted(totals_by_model.items()):
            avg_epochs = (
                totals["train_epochs_sum"] / totals["train_epochs_count"]
                if totals["train_epochs_count"] > 0
                else math.nan
            )
            avg_epochs_text = "" if math.isnan(avg_epochs) else f"{avg_epochs:.1f}"
            lines.append(
                "| "
                f"{model} | "
                f"{int(totals['trials'])} | "
                f"{totals['time_used_seconds'] / 3600:.2f} | "
                f"{int(totals['tokens_used']):,} | "
                f"{int(totals['tool_calls']):,} | "
                f"{avg_epochs_text} |"
            )
    lines.extend(handwritten_log_observations())
    lines.extend(handwritten_stm32n6_synthesis_check_summary())
    if best_by_device:
        lines.extend(
            [
                "",
                "## Best Passing Test Score",
                "",
                "| Device | Best Agent Score | Metric | Human Baseline | Source Run |",
                "|---|---:|---|---:|---|",
            ]
        )
        for device in sorted(best_by_device):
            best = best_by_device[device]
            metric, baseline = HUMAN_BASELINES.get(device, ("score", math.nan))
            baseline_text = "" if math.isnan(baseline) else f"{baseline}"
            lines.append(
                "| "
                f"{device_display_name(device)} | "
                f"{best['best_training_score_from_passing_iteration']} | "
                f"{metric} | "
                f"{baseline_text} | "
                f"`{best['experiment']}` / `{best['model']}` / `{best['reasoning']}` |"
            )
    path.write_text("\n".join(lines) + "\n")


def handwritten_log_observations() -> list[str]:
    return [
        "",
        "## Interesting Patterns",
        "",
        "- The strongest runs tend to be from the larger frontier models rather than the smaller or faster variants. In the current batch, `openai/gpt-5.4 [high]` is the most consistently useful model across both device families, while `claude/claude-opus-4-7 [high]` and `gemini/gemini-3.1-pro-preview [high]` show pockets of strong performance.",
        "- The smaller/fast models often get through gradient flow or training but fail to produce hardware-valid artifacts. This is especially visible in STM32N6 runs where plausible compact speech models still exceed the stricter activation SRAM budget.",
        "- Feedback matters most when it helps the model repair concrete hardware failures. The logs show several cases where agents move from invalid paths or full reference checkpoints toward compact Python model definitions after receiving check feedback.",
        "- Documentation-only context is mixed: it helps when the model reads the relevant device constraints, but it also tempts some agents into exploration-heavy behavior. Runs that spend many tool calls reading docs without writing a valid candidate tend to end with missing `model_path` or stale submissions.",
        "- STM32N6 synthesis failures are often near-misses rather than total codegen failures. Many candidates successfully export to ONNX and run through STEdgeAI/Neural-ART, then fail only because activations/IO exceed the stricter NUCLEO-N657X0-Q SRAM budget.",
        "- MAX78000 failures are more often structural: invalid YAML sidecars, unsupported ai8x synthesis patterns, or missing final inputs. Passing MAX78000 requires both a PyTorch definition and an ai8x-compatible architecture config to line up.",
        "- Test scores are noisy enough that the best-so-far curve is more informative than raw per-iteration scores. Some agents rediscover a previously good architecture and then regress, so the cumulative best plot better represents search quality.",
    ]


def handwritten_stm32n6_synthesis_check_summary() -> list[str]:
    return [
        "",
        "## STM32N6 Synthesis Checks Beyond Codegen",
        "",
        "The stricter STM32N6 check does more than ask STEdgeAI/STM32Cube.AI to emit C code:",
        "",
        "- Forces the N6 path with `--target stm32n6` and `--st-neural-art`, then verifies the output shows the Neural-ART compiler (`atonn`) and `stm32n6npu` target.",
        "- Exports Python PyTorch submissions to deployable ONNX internally and rejects dynamic or invalid input shapes before synthesis.",
        "- Enforces NUCLEO-N657X0-Q-specific budgets: 16 MB model partition with headroom, a 400 KiB app activation SRAM budget, and the NPU AXISRAM pool budget.",
        "- Parses generated `network_c_info.json` / `c_info.json` to confirm the target metadata, memory pools, hardware-mapped epochs, and static supported input/output tensors.",
        "- Checks that generated Neural-ART raw blobs exist and fit their declared AXISRAM/xSPI memory pools.",
        "- Parses final generated Flash/RAM totals and fails oversized activation/IO memory even when C generation itself succeeded.",
        "- Surfaces model-level incompatibilities such as non-quantized artifacts, unsupported export behavior, missing `model_path`, or submitting the full reference checkpoint instead of a compact model definition.",
    ]


def discover_runs(
    outputs_dir: Path, result_dir: Path, include: list[str]
) -> list[Path]:
    candidates: list[Path] = []
    for summary_path in outputs_dir.glob("*/summary.json"):
        output_dir = summary_path.parent
        if output_dir.resolve() == result_dir.resolve():
            continue
        if include and not any(
            output_dir.match(pattern) or output_dir.name == pattern
            for pattern in include
        ):
            continue
        candidates.append(output_dir)
    return sorted(candidates)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize and plot EdgeDL experiment outputs."
    )
    parser.add_argument(
        "output_name", help="Write results under outputs/<output_name>."
    )
    parser.add_argument(
        "--outputs-dir",
        default=str(DEFAULT_OUTPUTS_DIR),
        help="Directory containing experiment output subdirectories.",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="Optional output directory name or glob to include. Can be repeated.",
    )
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir).resolve()
    result_dir = outputs_dir / args.output_name
    result_dir.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(outputs_dir, result_dir, args.include)
    trial_rows: list[dict[str, Any]] = []
    passing_train_points: list[dict[str, Any]] = []
    iteration_rows: list[dict[str, Any]] = []
    for run_dir in runs:
        run_trials, run_points, run_iterations = parse_run(run_dir)
        trial_rows.extend(run_trials)
        passing_train_points.extend(run_points)
        iteration_rows.extend(run_iterations)
    if not trial_rows:
        raise SystemExit(f"No experiment summaries found in {outputs_dir}")

    aggregate = aggregate_rows(trial_rows)
    write_csv(result_dir / "results.csv", aggregate)
    write_csv(result_dir / "iterations.csv", iteration_rows, ITERATION_CSV_COLUMNS)
    grouped_bar_chart(
        trial_rows,
        metric="first_success_iteration",
        title="Iterations Until First Full Success",
        ylabel="Iterations",
        output_path=result_dir / "iterations_until_first_success.png",
        include_overall=False,
    )
    grouped_bar_chart(
        trial_rows,
        metric="fully_successful_iterations",
        title="Number Of Fully Successful Iterations",
        ylabel="Iterations",
        output_path=result_dir / "fully_successful_iterations.png",
    )
    training_score_line_chart(
        passing_train_points,
        result_dir / "test_score_by_iteration.png",
    )
    write_summary_md(
        result_dir / "summary.md",
        runs=runs,
        aggregate_rows_=aggregate,
        trial_rows=trial_rows,
        passing_train_points=passing_train_points,
    )
    manifest = {
        "runs": len(runs),
        "trial_rows": len(trial_rows),
        "iteration_rows": len(iteration_rows),
        "aggregate_rows": len(aggregate),
        "passing_training_score_points": len(passing_train_points),
        "output_dir": str(result_dir),
        "artifacts": [
            "results.csv",
            "iterations.csv",
            "iterations_until_first_success.png",
            "fully_successful_iterations.png",
            "test_score_by_iteration.png",
            "summary.md",
        ],
    }
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
