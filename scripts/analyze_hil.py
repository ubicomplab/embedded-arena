#!/usr/bin/env python3
"""Extract measurement / iteration stats from a HIL run folder and plot them.

Scans ``trial_*/iter_*`` for ``measurement_result.json`` and ``iter_result.json``,
builds a per-iteration table (CSV + optional HTML), and writes matplotlib figures
with vertical bands: green when compile + flash + measure checks all pass, red
otherwise (including missing iterations or failed compile/flash).

By default, artifacts go under ``<outputs-dir>/summary/csv`` and
``<outputs-dir>/summary/plot`` (named after each run folder).

With ``--all``, every immediate subdirectory of the given path that contains
``trial_*`` is processed, and two aggregate CSVs are written for the paper tables
(success rate % and best total energy in mJ), keyed by model and feedback mode.

Examples::

    python scripts/plot_hil_run_measurements.py outputs/hil-firmware-max78000_gpt_1
    python scripts/plot_hil_run_measurements.py outputs --all
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

os_import = __import__("os")
os_import.environ.setdefault("MPLCONFIGDIR", "/tmp/edgedl-matplotlib")
os_import.environ.setdefault("XDG_CACHE_HOME", "/tmp/edgedl-cache")

DEFAULT_CHECK_KEYS = (
    "compile_max78000.py",
    "flash_max78000.py",
    "measure_max78000.py",
)

ITER_DIR_RE = re.compile(r"^iter_(\d+)$")

# Naming mirrors scripts/run_batch_hil_experiments.sh (short tag -> display name for tables).
MODEL_TAG_ORDER: tuple[str, ...] = ("claude", "claudel", "gpt", "gptl", "gemini", "geminil")
MODEL_DISPLAY_NAME: dict[str, str] = {
    "claude": "Claude Opus 4.7",
    "claudel": "Claude Sonnet 4.6",
    "gpt": "GPT-5.4",
    "gptl": "GPT-5.4 mini",
    "gemini": "Gemini 3.1 Pro",
    "geminil": "Gemini 3 Flash",
}

# experiment yaml basename prefix -> table column (paper).
CONFIG_KEYS_ORDER: tuple[tuple[str, str], ...] = (
    ("no_feedback", "No Feedback"),
    ("doc", "Doc"),
    ("feedback", "Feedback"),
)
CONFIG_TABLE_LABELS: tuple[tuple[str, str], ...] = (
    ("no_feedback", "Sco"),
    ("doc", "Doc"),
    ("feedback", "HIL"),
)

TASK_ENERGY = "energy"
TASK_CURRENT = "current"
TASK_HEAT = "heat"
TARGET_MAX78000 = "max78000"
TARGET_ESP32 = "esp32"

def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _get_nested(d: dict[str, Any] | None, *keys: str) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _float_or_none(v: Any) -> float | None:
    if v is None:
        return None
    try:
        x = float(v)
        if math.isfinite(x):
            return x
    except (TypeError, ValueError):
        pass
    return None


def _float_or_empty(v: Any) -> float | str:
    x = _float_or_none(v)
    return "" if x is None else x


def discover_trial_iter_dirs(run_dir: Path) -> dict[int, list[int]]:
    """Return mapping trial_index -> sorted list of iter indices present on disk."""
    out: dict[int, list[int]] = {}
    if not run_dir.is_dir():
        return out
    for trial_path in sorted(run_dir.glob("trial_*")):
        m = re.match(r"trial_(\d+)$", trial_path.name)
        if not m or not trial_path.is_dir():
            continue
        tid = int(m.group(1))
        iters: list[int] = []
        for child in trial_path.iterdir():
            im = ITER_DIR_RE.match(child.name)
            if im and child.is_dir():
                iters.append(int(im.group(1)))
        if iters:
            out[tid] = sorted(iters)
    return out


def detect_check_keys(iter_data: dict[str, Any] | None) -> tuple[str, str, str]:
    if not iter_data:
        return DEFAULT_CHECK_KEYS
    checks = iter_data.get("checks")
    if not isinstance(checks, dict):
        return DEFAULT_CHECK_KEYS
    compile_key = next((k for k in checks if isinstance(k, str) and k.startswith("compile_") and k.endswith(".py")), DEFAULT_CHECK_KEYS[0])
    flash_key = next((k for k in checks if isinstance(k, str) and k.startswith("flash_") and k.endswith(".py")), DEFAULT_CHECK_KEYS[1])
    measure_key = next((k for k in checks if isinstance(k, str) and k.startswith("measure_") and k.endswith(".py")), DEFAULT_CHECK_KEYS[2])
    return (compile_key, flash_key, measure_key)


def all_three_checks_pass(iter_data: dict[str, Any] | None) -> bool | None:
    """True if compile/flash/measure checks succeeded; None if iter_result missing."""
    if not iter_data:
        return None
    checks = iter_data.get("checks")
    if not isinstance(checks, dict):
        return False
    for key in detect_check_keys(iter_data):
        c = checks.get(key)
        if not isinstance(c, dict) or not c.get("success"):
            return False
    return True


def extract_row(
    trial: int,
    iter_idx: int,
    iter_dir: Path | None,
    measurement_path: Path | None,
    iter_path: Path | None,
) -> dict[str, Any]:
    meas_raw = _read_json(measurement_path) if measurement_path else None
    iter_raw = _read_json(iter_path) if iter_path else None

    folder_exists = iter_dir is not None and iter_dir.is_dir()
    has_meas_file = meas_raw is not None
    has_iter_file = iter_raw is not None
    # Treat as "no iteration artifact" when folder missing or both JSONs unreadable
    missing = not folder_exists or (not has_meas_file and not has_iter_file)

    # Prefer top-level measurement_result; mirror keys exist under measure in iter_result
    power_src = _get_nested(meas_raw, "power") or _get_nested(iter_raw, "measure", "power")
    therm_src = _get_nested(meas_raw, "thermal") or _get_nested(iter_raw, "measure", "thermal")
    compile_key, flash_key, measure_key = detect_check_keys(iter_raw)

    checks_pass = all_three_checks_pass(iter_raw)

    compile_ok = bool(_get_nested(iter_raw, "compile", "success")) if iter_raw else False
    flash_ok = bool(_get_nested(iter_raw, "flash", "success")) if iter_raw else False
    measure_block = _get_nested(iter_raw, "measure")
    measure_ok = bool(measure_block.get("success")) if isinstance(measure_block, dict) else False

    row: dict[str, Any] = {
        "trial": trial,
        "iter": iter_idx,
        "iter_dir_exists": folder_exists,
        "has_measurement_result_json": has_meas_file,
        "has_iter_result_json": has_iter_file,
        "missing_or_empty": missing,
        "compile_success": "" if missing else compile_ok,
        "flash_success": "" if missing else flash_ok,
        "measure_success": "" if missing else measure_ok,
        "check_compile_ok": "" if missing else bool(_get_nested(iter_raw, "checks", compile_key, "success")),
        "check_flash_ok": "" if missing else bool(_get_nested(iter_raw, "checks", flash_key, "success")),
        "check_measure_ok": "" if missing else bool(_get_nested(iter_raw, "checks", measure_key, "success")),
        "all_three_checks_pass": "" if checks_pass is None else checks_pass,
        "measurement_success": "" if not meas_raw else bool(meas_raw.get("success")),
        "score": "" if missing else _get_nested(iter_raw, "measure", "score"),
        "metric": "" if missing else _get_nested(iter_raw, "measure", "metric"),
        "metric_value": "" if missing else _float_or_empty(_get_nested(iter_raw, "measure", "metric_value")),
        "checkpoint_found": "" if missing else _get_nested(iter_raw, "measure", "checkpoint", "found"),
        "avg_uA": ("" if missing else _float_or_empty(_get_nested(power_src, "avg_uA"))),
        "peak_uA": ("" if missing else _float_or_empty(_get_nested(power_src, "peak_uA"))),
        "total_energy_j": ("" if missing else _float_or_empty(_get_nested(power_src, "total_energy_j"))),
        "peak_temp_c": ("" if missing else _float_or_empty(_get_nested(therm_src, "peak_temp_c"))),
        "avg_temp_c": ("" if missing else _float_or_empty(_get_nested(therm_src, "avg_temp_c"))),
        "abs_max_temp_c": ("" if missing else _float_or_empty(_get_nested(therm_src, "abs_max_temp_c"))),
        "avg_max_temp_c": ("" if missing else _float_or_empty(_get_nested(therm_src, "avg_max_temp_c"))),
        "thermal_frames": (
            _get_nested(therm_src, "frames")
            if not missing
            else ""
        ),
    }
    if missing:
        for k in (
            "score",
            "metric",
            "metric_value",
            "checkpoint_found",
            "measurement_success",
            "compile_success",
            "flash_success",
            "measure_success",
            "check_compile_ok",
            "check_flash_ok",
            "check_measure_ok",
            "all_three_checks_pass",
        ):
            row[k] = ""
    else:
        # Normalize thermal_frames to int or empty
        tf = row["thermal_frames"]
        if tf is not None and tf != "":
            try:
                row["thermal_frames"] = int(tf)
            except (TypeError, ValueError):
                row["thermal_frames"] = ""

    return row


CSV_COLUMNS = [
    "trial",
    "iter",
    "iter_dir_exists",
    "has_measurement_result_json",
    "has_iter_result_json",
    "missing_or_empty",
    "compile_success",
    "flash_success",
    "measure_success",
    "check_compile_ok",
    "check_flash_ok",
    "check_measure_ok",
    "all_three_checks_pass",
    "measurement_success",
    "score",
    "metric",
    "metric_value",
    "checkpoint_found",
    "avg_uA",
    "peak_uA",
    "total_energy_j",
    "peak_temp_c",
    "avg_temp_c",
    "abs_max_temp_c",
    "avg_max_temp_c",
    "thermal_frames",
]


def build_rows(run_dir: Path) -> list[dict[str, Any]]:
    trial_map = discover_trial_iter_dirs(run_dir)
    if not trial_map:
        return []

    rows: list[dict[str, Any]] = []
    for trial in sorted(trial_map.keys()):
        present = set(trial_map[trial])
        max_iter = max(present)
        trial_path = run_dir / f"trial_{trial}"
        for i in range(max_iter + 1):
            iter_dir = trial_path / f"iter_{i}"
            if i not in present:
                rows.append(
                    extract_row(
                        trial,
                        i,
                        None,
                        None,
                        None,
                    )
                )
                continue
            mr = iter_dir / "measurement_result.json"
            ir = iter_dir / "iter_result.json"
            rows.append(
                extract_row(
                    trial,
                    i,
                    iter_dir,
                    mr if mr.is_file() else None,
                    ir if ir.is_file() else None,
                )
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in CSV_COLUMNS})


def write_html(path: Path, rows: list[dict[str, Any]], title: str) -> None:
    """HTML table: missing/failed-all-checks rows use red on white; green tint when all checks pass."""

    def cell_style(row: dict[str, Any]) -> str:
        if row.get("missing_or_empty"):
            return "background:#fff;color:#b00020;font-weight:600;"
        if row.get("all_three_checks_pass") is True:
            return "background:#e8f5e9;color:#1b5e20;"
        if row.get("all_three_checks_pass") is False:
            return "background:#fff;color:#b00020;"
        return ""

    display_cols = [
        "trial",
        "iter",
        "missing_or_empty",
        "all_three_checks_pass",
        "check_compile_ok",
        "check_flash_ok",
        "check_measure_ok",
        "avg_uA",
        "peak_uA",
        "total_energy_j",
        "peak_temp_c",
        "avg_temp_c",
        "abs_max_temp_c",
        "avg_max_temp_c",
        "thermal_frames",
        "metric_value",
        "score",
    ]
    lines = [
        "<!DOCTYPE html>",
        "<html><head><meta charset='utf-8'><title>HIL run measurements</title>",
        "<style>body{font-family:system-ui,sans-serif;margin:16px;}table{border-collapse:collapse;}",
        "th,td{border:1px solid #ccc;padding:6px 10px;font-size:13px;}",
        "th{background:#f5f5f5;}</style></head><body>",
        f"<h2>{title}</h2>",
        "<table><tr>",
    ]
    for c in display_cols:
        lines.append(f"<th>{c}</th>")
    lines.append("</tr>")
    for row in rows:
        lines.append("<tr>")
        for c in display_cols:
            st = cell_style(row)
            val = row.get(c, "")
            if val is True:
                val_s = "True"
            elif val is False:
                val_s = "False"
            else:
                val_s = "" if val == "" else str(val)
            lines.append(f"<td style='{st}'>{val_s}</td>")
        lines.append("</tr>")
    lines.append("</table></body></html>")
    path.write_text("\n".join(lines), encoding="utf-8")


def _band_ok(row: dict[str, Any]) -> bool:
    if row.get("missing_or_empty"):
        return False
    v = row.get("all_three_checks_pass")
    return v is True


def plot_metrics(
    rows: list[dict[str, Any]],
    out_path: Path,
    title: str,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    if not rows:
        return

    x = list(range(len(rows)))
    labels: list[str] = []
    for r in rows:
        t, i = r["trial"], r["iter"]
        labels.append(f"t{t}\ni{i}")

    def series(key: str) -> list[float]:
        vals: list[float] = []
        for r in rows:
            v = r.get(key)
            if v == "" or v is None:
                vals.append(float("nan"))
            else:
                try:
                    vals.append(float(v))
                except (TypeError, ValueError):
                    vals.append(float("nan"))
        return vals

    metrics_spec = [
        ("avg_uA", "Average current (µA)"),
        ("peak_uA", "Peak current (µA)"),
        ("total_energy_j", "Total energy (J)"),
        ("peak_temp_c", "Peak temperature (°C)"),
        ("avg_temp_c", "Average temperature (°C)"),
        ("abs_max_temp_c", "Abs max temperature (°C)"),
        ("metric_value", "Optimization metric value"),
    ]

    fig, axes = plt.subplots(len(metrics_spec), 1, figsize=(max(10, len(rows) * 0.35), 2.8 * len(metrics_spec)), sharex=True)
    if len(metrics_spec) == 1:
        axes = [axes]

    for ax in axes:
        for i, row in enumerate(rows):
            x0, x1 = i - 0.45, i + 0.45
            color = "#c8e6c9" if _band_ok(row) else "#ffcdd2"
            ax.axvspan(x0, x1, facecolor=color, alpha=0.85, zorder=0)

    for ax, (key, ylabel) in zip(axes, metrics_spec, strict=True):
        y = series(key)
        ax.plot(x, y, "o-", color="#1565c0", lw=1.2, markersize=4, zorder=2)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, alpha=0.3, zorder=1)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, fontsize=8)
    axes[0].set_title(title)
    fig.legend(
        handles=[
            Patch(facecolor="#c8e6c9", edgecolor="#81c784", label="All 3 hardware checks pass"),
            Patch(facecolor="#ffcdd2", edgecolor="#e57373", label="Missing or any check fails"),
        ],
        loc="upper right",
        fontsize=9,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def is_hil_run_directory(path: Path) -> bool:
    """True if ``path`` looks like a run folder (contains at least one ``trial_*`` dir)."""
    if not path.is_dir():
        return False
    for child in path.iterdir():
        if child.is_dir() and child.name.startswith("trial_"):
            return True
    return False


def discover_run_dirs(outputs_parent: Path) -> list[Path]:
    """Immediate subdirectories of ``outputs_parent`` that contain ``trial_*``."""
    if not outputs_parent.is_dir():
        return []
    found: list[Path] = []
    for child in sorted(outputs_parent.iterdir()):
        if child.is_dir() and is_hil_run_directory(child):
            found.append(child)
    return found


def success_iteration_percent(rows: list[dict[str, Any]]) -> float:
    """Share of iteration rows where all three hardware checks passed (0–100)."""
    if not rows:
        return 0.0
    ok = sum(1 for r in rows if r.get("all_three_checks_pass") is True)
    return round(100.0 * ok / len(rows), 2)


def best_total_energy_mj(rows: list[dict[str, Any]]) -> float | None:
    """Minimum total energy in millijoules among iterations with all checks passed."""
    best: float | None = None
    for r in rows:
        if r.get("all_three_checks_pass") is not True:
            continue
        v = _float_or_none(r.get("total_energy_j"))
        if v is None:
            continue
        mj = v * 1000.0
        if best is None or mj < best:
            best = mj
    return best


def best_metric_value(rows: list[dict[str, Any]]) -> tuple[str | None, float | None]:
    """Minimum metric_value among successful iterations, with metric name."""
    best: float | None = None
    metric_name: str | None = None
    for r in rows:
        if r.get("all_three_checks_pass") is not True:
            continue
        v = _float_or_none(r.get("metric_value"))
        if v is None:
            continue
        if best is None or v < best:
            best = v
            m = r.get("metric")
            metric_name = m if isinstance(m, str) and m else metric_name
    return metric_name, best


def detect_task_kind(run_name: str) -> str:
    if "peakCurrent" in run_name:
        return TASK_CURRENT
    return TASK_ENERGY


def detect_target_kind(run_name: str) -> str:
    if "esp32" in run_name:
        return TARGET_ESP32
    return TARGET_MAX78000


def summary_root_for_task(base_summary_root: Path, task_kind: str) -> Path:
    if base_summary_root.name == task_kind:
        return base_summary_root
    return base_summary_root / task_kind


def summary_root_for_run(base_summary_root: Path, task_kind: str, target_kind: str) -> Path:
    if target_kind == TARGET_ESP32:
        return summary_root_for_task(base_summary_root, TASK_HEAT)
    return summary_root_for_task(base_summary_root, task_kind)


def expected_run_folder_name(
    target_kind: str,
    task_kind: str,
    config_key: str,
    model_tag: str,
    run_num: int,
) -> str:
    base = f"hil-{'peakCurrent' if task_kind == TASK_CURRENT else 'firmware'}-{target_kind}"
    if config_key == "no_feedback":
        return f"{base}-no-feedback_{model_tag}_{run_num}"
    if config_key == "doc":
        return f"{base}-doc_{model_tag}_{run_num}"
    return f"{base}_{model_tag}_{run_num}"


def write_aggregate_tables(
    target_kind: str,
    task_kind: str,
    outputs_parent: Path,
    run_number: int,
    summary_csv_dir: Path,
) -> None:
    """Write two CSV matrices (models × No Feedback / Doc / Feedback) for LaTeX tables."""
    summary_csv_dir.mkdir(parents=True, exist_ok=True)

    success_path = summary_csv_dir / f"table_success_rate_percent_{target_kind}.csv"
    metric_file_label = "peak_uA" if task_kind == TASK_CURRENT else "total_energy_mj"

    success_rows: list[dict[str, str]] = []
    metric_rows: list[dict[str, str]] = []

    for tag in MODEL_TAG_ORDER:
        display = MODEL_DISPLAY_NAME[tag]
        s_out: dict[str, str] = {"Model": display}
        m_out: dict[str, str] = {"Model": display}
        for config_key, col_label in CONFIG_KEYS_ORDER:
            folder_name = expected_run_folder_name(target_kind, task_kind, config_key, tag, run_number)
            run_dir = outputs_parent / folder_name
            if not is_hil_run_directory(run_dir):
                s_out[col_label] = ""
                m_out[col_label] = ""
                continue
            rows = build_rows(run_dir)
            pct = success_iteration_percent(rows)
            s_out[col_label] = str(int(round(pct))) if pct == int(pct) else str(pct)
            if task_kind == TASK_CURRENT:
                best_peak: float | None = None
                for r in rows:
                    if r.get("all_three_checks_pass") is not True:
                        continue
                    v = _float_or_none(r.get("peak_uA"))
                    if v is None:
                        continue
                    if best_peak is None or v < best_peak:
                        best_peak = v
                m_out[col_label] = "" if best_peak is None else f"{best_peak:.2f}"
            else:
                if target_kind == TARGET_ESP32:
                    metric_name, best_metric = best_metric_value(rows)
                    if metric_name:
                        metric_file_label = metric_name
                    m_out[col_label] = "" if best_metric is None else f"{best_metric:.2f}"
                else:
                    best = best_total_energy_mj(rows)
                    m_out[col_label] = "" if best is None else f"{best:.2f}"
        success_rows.append(s_out)
        metric_rows.append(m_out)

    metric_path = summary_csv_dir / f"table_best_{metric_file_label}_{target_kind}.csv"

    fieldnames = ["Model"] + [c[1] for c in CONFIG_KEYS_ORDER]
    with success_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(success_rows)
    with metric_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(metric_rows)

    print(f"Wrote {success_path}")
    print(f"Wrote {metric_path}")


def _task_display_name(task_kind: str) -> str:
    if task_kind == TASK_CURRENT:
        return "Peak-Current Minimization"
    return "Energy Minimization"


def _format_percent_1dp(value: float | None) -> str:
    return "" if value is None else f"{value:.1f}"


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def write_task_success_tables(
    target_kind: str,
    outputs_parent: Path,
    run_number: int,
    summary_csv_dir: Path,
) -> None:
    """Write task-level success-rate CSVs averaged across models (N=6 when complete)."""
    summary_csv_dir.mkdir(parents=True, exist_ok=True)

    task_values: dict[str, dict[str, float | None]] = {}
    for task_kind in (TASK_CURRENT, TASK_ENERGY):
        by_config: dict[str, float | None] = {}
        for config_key, _ in CONFIG_TABLE_LABELS:
            per_model: list[float] = []
            for model_tag in MODEL_TAG_ORDER:
                folder_name = expected_run_folder_name(
                    target_kind,
                    task_kind,
                    config_key,
                    model_tag,
                    run_number,
                )
                run_dir = outputs_parent / folder_name
                if not is_hil_run_directory(run_dir):
                    continue
                pct = success_iteration_percent(build_rows(run_dir))
                per_model.append(pct)
            by_config[config_key] = _mean(per_model)
        task_values[task_kind] = by_config

    combined_path = summary_csv_dir / f"table_success_rate_percent_mean_by_task_{target_kind}.csv"
    combined_rows: list[dict[str, str]] = []
    if target_kind == TARGET_ESP32:
        # ESP32 table currently reports max-temp optimization only; keep the second row blank.
        combined_rows.append(
            {
                "Task": "Max-Temp Minimization",
                "Sco": _format_percent_1dp(task_values[TASK_ENERGY].get("no_feedback")),
                "Doc": _format_percent_1dp(task_values[TASK_ENERGY].get("doc")),
                "HIL": _format_percent_1dp(task_values[TASK_ENERGY].get("feedback")),
            }
        )
        combined_rows.append(
            {
                "Task": "Peak-Current Minimization",
                "Sco": "",
                "Doc": "",
                "HIL": "",
            }
        )
    else:
        for task_kind in (TASK_CURRENT, TASK_ENERGY):
            combined_rows.append(
                {
                    "Task": _task_display_name(task_kind),
                    "Sco": _format_percent_1dp(task_values[task_kind].get("no_feedback")),
                    "Doc": _format_percent_1dp(task_values[task_kind].get("doc")),
                    "HIL": _format_percent_1dp(task_values[task_kind].get("feedback")),
                }
            )
    with combined_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["Task", "Sco", "Doc", "HIL"])
        w.writeheader()
        w.writerows(combined_rows)
    print(f"Wrote {combined_path}")


def process_one_run(
    run_dir: Path,
    base_summary_root: Path,
    *,
    write_html_file: bool,
    write_plot_file: bool,
) -> bool:
    """Write CSV/HTML under ``summary_root/csv`` and PNG under ``summary_root/plot``. Returns False if no rows."""
    rows = build_rows(run_dir)
    if not rows:
        print(f"No trial_*/iter_* directories found under {run_dir}")
        return False

    stem = run_dir.name
    task_kind = detect_task_kind(stem)
    target_kind = detect_target_kind(stem)
    summary_root = summary_root_for_run(base_summary_root, task_kind, target_kind)
    csv_dir = summary_root / "csv"
    plot_dir = summary_root / "plot"
    csv_dir.mkdir(parents=True, exist_ok=True)

    csv_path = csv_dir / f"{stem}_iter_metrics.csv"
    write_csv(csv_path, rows)
    print(f"Wrote {csv_path}")

    if write_html_file:
        html_path = csv_dir / f"{stem}_iter_metrics.html"
        write_html(html_path, rows, title=f"HIL iteration metrics — {stem}")
        print(f"Wrote {html_path}")

    if write_plot_file:
        try:
            plot_path = plot_dir / f"{stem}_iter_metrics.png"
            plot_metrics(rows, plot_path, title=f"{stem}: metrics by iteration")
            print(f"Wrote {plot_path}")
        except ImportError as e:
            print(f"Skipping plot for {stem} (matplotlib not available): {e}")
    return True


def discover_targets_for_task(outputs_parent: Path, task_kind: str) -> list[str]:
    prefix = "hil-peakCurrent-" if task_kind == TASK_CURRENT else "hil-firmware-"
    targets: set[str] = set()
    for run_dir in discover_run_dirs(outputs_parent):
        name = run_dir.name
        if not name.startswith(prefix):
            continue
        rest = name[len(prefix):]
        target = rest.split("_", 1)[0].split("-doc", 1)[0].split("-no-feedback", 1)[0]
        if target:
            targets.add(target)
    return sorted(targets)


def main() -> None:
    p = argparse.ArgumentParser(description="Summarize HIL measurement_result / iter_result per iteration.")
    p.add_argument(
        "path",
        type=Path,
        help="Single run directory (e.g. outputs/hil-firmware-max78000_gpt_1) or outputs root with --all",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Process every HIL run folder directly under PATH; write aggregate table CSVs",
    )
    p.add_argument(
        "--summary-root",
        type=Path,
        default=None,
        help="Override summary root (default: <outputs-parent>/summary)",
    )
    p.add_argument(
        "--aggregate-run-number",
        type=int,
        default=1,
        metavar="N",
        help="Run suffix for aggregate tables (default: 1, i.e. *_1 folders)",
    )
    p.add_argument(
        "--no-aggregate",
        action="store_true",
        help="With --all, skip writing the two summary table CSVs",
    )
    p.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Alias for --summary-root (deprecated name)",
    )
    p.add_argument("--no-html", action="store_true", help="Skip HTML table")
    p.add_argument("--no-plot", action="store_true", help="Skip matplotlib figure")
    args = p.parse_args()

    scan_root = args.path.resolve()
    if args.summary_root or args.output_dir:
        summary_root = (args.summary_root or args.output_dir).resolve()
    elif args.all:
        summary_root = (scan_root / "summary").resolve()
    else:
        summary_root = (scan_root.parent / "summary").resolve()

    if args.all:
        if not scan_root.is_dir():
            print(f"Not a directory: {scan_root}")
            return
        runs = discover_run_dirs(scan_root)
        if not runs:
            print(f"No run folders (trial_*) found under {scan_root}")
            return
        for rd in runs:
            process_one_run(
                rd,
                summary_root,
                write_html_file=not args.no_html,
                write_plot_file=not args.no_plot,
            )
        if not args.no_aggregate:
            for task_kind in (TASK_ENERGY, TASK_CURRENT):
                targets = discover_targets_for_task(scan_root, task_kind)
                for target_kind in targets:
                    write_aggregate_tables(
                        target_kind,
                        task_kind,
                        scan_root,
                        args.aggregate_run_number,
                        summary_root_for_run(summary_root, task_kind, target_kind) / "csv",
                    )
            all_targets = sorted(
                set(discover_targets_for_task(scan_root, TASK_ENERGY))
                | set(discover_targets_for_task(scan_root, TASK_CURRENT))
            )
            for target_kind in all_targets:
                write_task_success_tables(
                    target_kind,
                    scan_root,
                    args.aggregate_run_number,
                    summary_root / "csv",
                )
        return

    run_dir = scan_root
    if not run_dir.is_dir():
        print(f"Not a directory: {run_dir}")
        return

    task_kind = detect_task_kind(run_dir.name)
    target_kind = detect_target_kind(run_dir.name)
    summary_for_single = summary_root_for_run(summary_root, task_kind, target_kind)
    process_one_run(
        run_dir,
        summary_root,
        write_html_file=not args.no_html,
        write_plot_file=not args.no_plot,
    )
    if not args.no_aggregate:
        outputs_parent = run_dir.parent
        write_aggregate_tables(
            target_kind,
            task_kind,
            outputs_parent,
            args.aggregate_run_number,
            summary_for_single / "csv",
        )
        write_task_success_tables(
            target_kind,
            outputs_parent,
            args.aggregate_run_number,
            summary_root / "csv",
        )


if __name__ == "__main__":
    main()
