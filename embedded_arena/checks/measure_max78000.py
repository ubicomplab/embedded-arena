"""Measure power, serial, and thermal from a MAX78000 DUT (firmware pre-flashed).

Modality flags (all YAML-controlled, not agent-settable):
  capture_power   — measure current via Nordic PPK2; ``ppk_sample_rate_hz`` is
                    passed to ``ppk2Monitor`` (default hardware ~100 kHz). Output
                    ``times``/``currents`` and CSVs match that effective rate.
  capture_serial  — tail the DUT UART during the capture window
  capture_thermal — record IR frames from an ESP32-S3 + MLX90640 camera

Score:
  score = normalized optimization_metric value in [0.0, 1.0], derived from
          metric_min, metric_max, and lower_is_better (all YAML-controlled).

Hardware metrics reported when the corresponding modality is enabled:
  - total_energy_j : total energy consumed over the measurement window (J)
  - peak_uA        : absolute peak current (µA)
  - avg_uA         : mean current over the window (µA)
  - peak_temp_c    : maximum pixel temperature (°C)
  - avg_temp_c     : mean pixel temperature (°C)

Hardware contract: a MAX78000 with firmware already flashed (by flash_max78000),
and optionally a Nordic PPK2 (capture_power=true) and an ESP32-S3 + MLX90640
camera (capture_thermal=true).  IR frames are saved to
<output_dir>/trial_<N>/iter_<N>/ir_measurement/.
"""

from __future__ import annotations

import json
import re
import threading
import time
import traceback as tb
from pathlib import Path
from typing import Any

import csv
import shutil

import numpy as np

from exceptions import ExperimentSetupError
from schemas import RunState, CheckResult
from pydantic import BaseModel, Field
from hardware.ppk2 import PPK2_HARDWARE_SAMPLE_RATE_HZ, PPK2_MP, ppk2Monitor  # type: ignore
from hardware.serial_monitor import SerialMonitor, detect_dut_port
from hardware.ir_camera import IRCamera, IRFrame

DEFAULT_PPK_SAMPLE_RATE_HZ: float = PPK2_HARDWARE_SAMPLE_RATE_HZ
V_SUPPLY: float = 3.3
CHECKPOINT_MSG: str = "firmware task complete checkpoint"


class Input(BaseModel):
    """Agent-settable input fields."""

    baud_rate: int = Field(
        default=115200,
        ge=9600,
        description="Baud rate for tailing the DUT UART during measurement.",
    )
    firmware_behavior_description: str = Field(
        default="",
        description=(
            "Description of what the firmware is doing (e.g. 'running YOLO inference at ~4 fps, "
            "printing confidence scores'). Fed to the LLM judge together with the task description "
            "to verify correct firmware behavior. Only evaluated after the UART checkpoint is found."
        ),
    )


class YAMLInput(BaseModel):
    """YAML-controlled configuration parameters (not agent-settable)."""

    duration_ms: int = Field(
        default=20000,
        ge=1000,
        le=120000,
        description="Capture window in milliseconds (1 000 – 120 000).",
    )
    capture_power: bool = Field(
        default=True,
        description="Record current via the Nordic PPK2 source meter.",
    )
    capture_serial: bool = Field(
        default=True,
        description="Tail the DUT UART serial output during the capture window.",
    )
    capture_thermal: bool = Field(
        default=True,
        description="Record IR frames from the ESP32-S3 + MLX90640 camera.",
    )
    ir_rate_hz: float = Field(
        default=4.0,
        description="Frame rate requested from the IR camera (Hz).",
    )
    target_voltage_v: float = Field(
        default=3.3,
        ge=1.0,
        le=5.0,
        description="Target device voltage in volts. Required when capture_power=true.",
    )
    optimization_metric: str = Field(
        default="total_energy_j",
        description=(
            "Metric to optimize. One of: total_energy_j, avg_uA, peak_uA, "
            "peak_temp_c, avg_temp_c."
        ),
    )
    metric_min: float = Field(
        default=0.0,
        description=(
            "Lower bound of the expected metric range. When lower_is_better=true "
            "this value maps to score=1.0; when lower_is_better=false it maps to score=0.0."
        ),
    )
    metric_max: float = Field(
        default=1.0,
        description=(
            "Upper bound of the expected metric range. When lower_is_better=true "
            "this value maps to score=0.0; when lower_is_better=false it maps to score=1.0."
        ),
    )
    lower_is_better: bool = Field(
        default=True,
        description="If True (default), lower metric values yield higher scores (energy, temperature).",
    )
    task_description: str = Field(
        default="",
        description=(
            "Experiment task description passed to the LLM judge as context for "
            "evaluating whether the UART output reflects correct firmware behavior."
        ),
    )
    show_ir_viewer: bool = Field(
        default=False,
        description=(
            "Open a live matplotlib window showing IR thermal frames during measurement. "
            "Requires a GUI display; set to false in headless/CI environments."
        ),
    )
    ppk_sample_rate_hz: float = Field(
        default=DEFAULT_PPK_SAMPLE_RATE_HZ,
        gt=0,
        le=DEFAULT_PPK_SAMPLE_RATE_HZ,
        description=(
            "Output sample rate (Hz) for PPK ``times``/``currents`` (passed to "
            "ppk2Monitor). Hardware streams ~100 kHz; lower values use block "
            "averaging inside the PPK helper. Omit from YAML to use default "
            f"({DEFAULT_PPK_SAMPLE_RATE_HZ:.0f} Hz)."
        ),
    )


def _power_summary(currents: np.ndarray, sample_rate_hz: float) -> dict:
    arr = np.asarray(currents, dtype=np.float64)
    sr = float(sample_rate_hz) if sample_rate_hz > 0 else DEFAULT_PPK_SAMPLE_RATE_HZ
    dt = 1.0 / sr
    return {
        "total_energy_j": float(np.sum(arr * 1e-6 * V_SUPPLY * dt)),
        "peak_uA": float(np.max(arr)),
        "avg_uA": float(np.mean(arr)),
    }


def _summarize_thermal(frames: list[np.ndarray]) -> tuple[dict | None, str]:
    if not frames:
        return None, "no thermal frames captured"
    stack = np.stack(frames)
    finite = np.isfinite(stack)
    if not np.any(finite):
        return None, "all thermal samples were non-finite"
    flat = stack[finite]
    summary = {
        "min_c": float(np.min(flat)),
        "avg_c": float(np.mean(flat)),
        "peak_c": float(np.max(flat)),
        "frames": len(frames),
    }
    return summary, (
        f"frames={summary['frames']}, "
        f"min={summary['min_c']:.2f}°C, "
        f"avg={summary['avg_c']:.2f}°C, "
        f"peak={summary['peak_c']:.2f}°C"
    )


def _summarize_thermal_cropped(frame_metas: list[dict]) -> tuple[dict | None, str]:
    """Summarize thermal stats from cropped region of IR frames using metadata."""
    if not frame_metas:
        return None, "no thermal frames captured"

    crop_mins = [m.get('crop_min') for m in frame_metas if m.get('crop_min') is not None]
    crop_maxs = [m.get('crop_max') for m in frame_metas if m.get('crop_max') is not None]
    crop_avgs = [m.get('crop_avg') for m in frame_metas if m.get('crop_avg') is not None]

    if not crop_maxs:
        return None, "no valid cropped thermal samples"

    summary = {
        "min_c": float(np.min(crop_mins)),
        "avg_c": float(np.mean(crop_avgs)),
        "peak_c": float(np.max(crop_maxs)),
        "frames": len(frame_metas),
    }
    return summary, (
        f"frames={summary['frames']}, "
        f"min={summary['min_c']:.2f}°C, "
        f"avg={summary['avg_c']:.2f}°C, "
        f"peak={summary['peak_c']:.2f}°C"
    )


def _compute_metric_score(
    value: float,
    metric_min: float,
    metric_max: float,
    lower_is_better: bool,
) -> float:
    if metric_max <= metric_min:
        return 0.5
    if lower_is_better:
        raw = (metric_max - value) / (metric_max - metric_min)
    else:
        raw = (value - metric_min) / (metric_max - metric_min)
    return max(0.0, min(1.0, raw))


def _judge_firmware_behavior(
    task_description: str,
    firmware_behavior_description: str,
    serial_output: str,
) -> tuple[bool, str]:
    """Return (passed, reason) via LLM judgment of the UART serial output."""
    raw = serial_output or "(no serial output captured)"
    head_chars, tail_chars = 2000, 6000
    if len(raw) <= head_chars + tail_chars:
        serial_preview = raw
    else:
        serial_preview = (
            raw[:head_chars]
            + f"\n...[{len(raw) - head_chars - tail_chars} chars omitted]...\n"
            + raw[-tail_chars:]
        )
    prompt = (
        "You are a lenient firmware behavior judge. Evaluate whether the UART serial output "
        "generally shows the firmware functioning as the task requires.\n\n"
        f"Task description:\n{task_description or '(not provided)'}\n\n"
        f"Expected firmware behavior (agent description):\n"
        f"{firmware_behavior_description or '(not provided)'}\n\n"
        f"UART serial output:\n{serial_preview}\n\n"
        'Respond ONLY with a JSON object: {"pass": true/false, "reason": "brief explanation"}')

    def _extract_json(text: str) -> tuple[bool, str] | None:
        m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if not m:
            return None
        try:
            data = json.loads(m.group())
            return bool(data.get("pass", False)), str(data.get("reason", ""))
        except Exception:
            return None

    try:
        import openai  # type: ignore

        client = openai.OpenAI()
        resp = client.chat.completions.create(
            model="gpt-5.4-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        text = (resp.choices[0].message.content or "").strip()
        parsed = _extract_json(text)
        if parsed is not None:
            return parsed
        return False, f"judge returned unparseable response: {text[:200]}"
    except Exception as e:
        print(f"[judge] OpenAI failed ({e}), falling back to Anthropic...")


    try:
        import anthropic  # type: ignore

        client = anthropic.Anthropic()
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text.strip()  # type: ignore
        parsed = _extract_json(text)
        if parsed is not None:
            return parsed
        return False, f"judge returned unparseable response: {text[:200]}"
    except Exception as exc:
        return True, f"judge unavailable ({exc}); skipping firmware verification"


def _init_ir_viewer() -> Any | None:
    """Open a live matplotlib window for IR thermal frames. Returns (fig, im, title) or None."""
    try:
        import matplotlib.pyplot as plt
        try:
            from hardware.ir_camera import FRAME_H, FRAME_W
        except ImportError:
            FRAME_H, FRAME_W = 24, 32
        plt.ion()
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        try:
            fig.canvas.manager.set_window_title("IR Camera — Live Measurement")
        except Exception:
            pass
        im = ax.imshow(
            np.zeros((FRAME_H, FRAME_W), dtype=np.float32),
            cmap="inferno", vmin=20.0, vmax=35.0, interpolation="nearest",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, label="°C")
        title = ax.set_title("Waiting for first IR frame…")
        fig.canvas.draw_idle()
        try:
            plt.pause(0.05)
        except Exception:
            pass
        return (fig, im, title)
    except Exception as exc:
        print(f"[measure_max78000] IR live viewer could not be opened: {exc}")
        return None


def _update_ir_viewer_frame(viewer: Any, frame: Any) -> None:
    """Update the IR viewer with a new frame (call from the main thread or via draw_idle)."""
    if viewer is None or frame is None:
        return
    try:
        _fig, im, title = viewer
        pixels = np.flipud(np.asarray(frame.pixels, dtype=np.float32))
        im.set_data(pixels)
        finite = pixels[np.isfinite(pixels)]
        if finite.size > 0:
            lo, hi = float(finite.min()), float(finite.max())
            if hi - lo < 0.5:
                hi = lo + 0.5
            im.set_clim(lo, hi)
        meta = frame.metadata or {}
        def _f(v: Any) -> str:
            try:
                return f"{float(v):.2f}"
            except (TypeError, ValueError):
                return "nan"
        title.set_text(
            f"min={_f(meta.get('min'))}°C  max={_f(meta.get('max'))}°C  avg={_f(meta.get('avg'))}°C"
        )
    except Exception:
        pass


def _run_monitor(
    duration_ms: int,
    baud_rate: int,
    capture_power: bool,
    capture_serial: bool,
    capture_thermal: bool,
    ir_rate_hz: float,
    target_voltage_v: float = 3.3,
    ppk_sample_rate_hz: float = DEFAULT_PPK_SAMPLE_RATE_HZ,
    verbose: bool = True,
    show_ir_viewer: bool = False,
) -> dict[str, Any]:
    """Run PPK2, serial, and IR capture in parallel."""
    ppk_monitor = ppk2Monitor(sample_rate_hz=ppk_sample_rate_hz)

    if capture_power and ppk_monitor.ppk2_port is None:
        raise ExperimentSetupError(
            "PPK2 not detected. Connect a Nordic PPK2 powering the MAX78000 DUT."
        )
    if verbose and capture_power:
        print(f"[measure_max78000] PPK2 detected on {ppk_monitor.ppk2_port}")

    dut_port = detect_dut_port() if capture_serial else None
    if capture_serial and dut_port is None:
        raise ExperimentSetupError(
            "DUT serial port not detected. Connect the MAX78000 UART-to-USB adapter "
            "or set capture_serial=false in the YAML config."
        )
    if verbose and capture_serial:
        print(f"[measure_max78000] DUT serial port: {dut_port} @ {baud_rate} baud")

    serial_monitor = SerialMonitor(dut_port, baud_rate) if capture_serial else None
    cam: IRCamera | None = None
    captured_pixels: list[np.ndarray] = []
    captured_frame_metas: list[dict] = []
    thermal_times_s: list[float] = []
    capture_lock = threading.Lock()
    latest_frame_lock = threading.Lock()
    latest_frame_ref: list[Any] = [None]
    capture_start_ref = [0.0]
    viewer: Any = None
    viewer_stop = threading.Event()
    viewer_thread: Any = None
    result: dict[str, Any] | None = None
    serial_text = ""

    if capture_thermal:
        cam = IRCamera()
        if cam.port is None:
            raise ExperimentSetupError(
                "IR camera not detected. Connect the ESP32-S3 + MLX90640 module "
                "or set capture_thermal=false in the YAML config."
            )
        if verbose:
            print(
                f"[measure_max78000] IR camera detected on {cam.port} @ {ir_rate_hz} Hz"
            )
        try:
            cam.open()
            cam.set_rate(ir_rate_hz)
        except (TimeoutError, AssertionError) as exc:
            cam.close()
            raise ExperimentSetupError(
                f"could not initialize IR camera: {exc}"
            ) from exc

        def _on_frame(frame: IRFrame) -> None:
            timestamp_s = max(0.0, time.monotonic() - capture_start_ref[0])
            with capture_lock:
                captured_pixels.append(frame.pixels)
                captured_frame_metas.append(frame.metadata)
                thermal_times_s.append(timestamp_s)
            with latest_frame_lock:
                latest_frame_ref[0] = frame

    try:
        ppk2 = None
        ppk_raw_data = None
        capture_start_ref[0] = time.monotonic()
        if capture_power:
            ppk2 = ppk_monitor._start_ppk2_monitor(target_voltage_v=target_voltage_v)
            capture_start_ref[0] = time.monotonic()
        if serial_monitor is not None:
            serial_monitor.start()
        if cam is not None:
            cam.start_streaming(on_frame=_on_frame, save_dir=None, save_latest=False)

        if show_ir_viewer and capture_thermal and cam is not None:
            if capture_power:
                print("[measure_max78000] IR live viewer disabled (PPK2 blocks main thread)")
            else:
                viewer = _init_ir_viewer()
                if viewer is not None:
                    print("[measure_max78000] IR live viewer opened")

        if capture_power and ppk2 is not None:
            ppk_raw_data = ppk_monitor._collect_ppk2_data_continuous(ppk2, duration_ms)
        else:
            if viewer is not None:
                import matplotlib.pyplot as plt
                t_end = time.monotonic() + duration_ms / 1000.0
                while time.monotonic() < t_end:
                    with latest_frame_lock:
                        frame = latest_frame_ref[0]
                    _update_ir_viewer_frame(viewer, frame)
                    try:
                        plt.pause(0.05)
                    except Exception:
                        remaining = t_end - time.monotonic()
                        time.sleep(min(0.05, max(0.0, remaining)))
            else:
                time.sleep(duration_ms / 1000.0)

        ppk_monitor._stop_ppk2_monitor(ppk2)

        ppk_report, ppk_times, currents = (
            ppk_monitor._process_ppk2_data(ppk2, ppk_raw_data)
            if capture_power
            else ("", np.array([]), np.array([]))
        )
        currents = np.asarray(currents, dtype=np.float64)
        ppk_times = np.asarray(ppk_times, dtype=np.float64)
        sr_out = float(ppk_monitor.output_sample_rate_hz)
        step_5hz = int(max(1, round(sr_out / 5.0)))
        result = {
            "ppk_monitor": ppk_monitor if capture_power else None,
            "ppk": (ppk_report, ppk_times, currents),
            "ppk_data": {
                "currents_5hz": (
                    currents[::step_5hz] if len(currents) > 0 else np.array([])
                ),
                "times_5hz": (
                    ppk_times[::step_5hz] if len(ppk_times) > 0 else np.array([])
                ),
                "stats": {
                    "samples": int(len(currents)),
                    "peak_uA": float(np.max(currents)) if len(currents) > 0 else 0.0,
                    "avg_uA": float(np.mean(currents)) if len(currents) > 0 else 0.0,
                    "sample_rate_hz": sr_out,
                },
            },
            "thermal_pixels": list(captured_pixels),
            "thermal_frame_metas": list(captured_frame_metas),
            "thermal_times_s": list(thermal_times_s),
        }
    except Exception:
        if ppk_monitor is not None:
            try:
                ppk_monitor.voltage_off()
            except Exception as e:
                print(
                    f"Warning: failed to turn off PPK2 voltage during error handling: {e}"
                )
        raise
    finally:
        viewer_stop.set()
        if viewer_thread is not None:
            viewer_thread.join(timeout=1.0)
        if viewer is not None:
            try:
                import matplotlib.pyplot as plt
                plt.ioff()
                plt.close(viewer[0])
                print("[measure_max78000] IR live viewer closed")
            except Exception:
                pass
        if cam is not None:
            cam.stop_streaming()
            cam.close()
        if serial_monitor is not None:
            serial_monitor.stop()

        if serial_monitor is not None:
            serial_text = serial_monitor.collect()

    if result is None:
        raise ExperimentSetupError("monitor orchestration did not produce a result")
    result["serial_text"] = serial_text
    return result


def _save_measure_to_iter_result(state: RunState, payload: dict) -> None:
    """Merge a measurement payload into iter_result.json (creating it if needed)."""
    output_dir_str = state.metadata.get("output_dir")
    if not output_dir_str:
        return
    iter_dir = (
        Path(output_dir_str)
        / f"trial_{state.trial_index}"
        / f"iter_{state.iteration_index}"
    )
    iter_dir.mkdir(parents=True, exist_ok=True)
    meas_dir = iter_dir / "measurement_data"
    meas_dir.mkdir(parents=True, exist_ok=True)

    # If ppk tuple present: (report, times, currents) -> save paired CSV and compute total energy
    ppk = payload.get("ppk")
    if ppk and isinstance(ppk, (list, tuple)) and len(ppk) >= 3:
        _, ppk_times, currents = ppk
        try:
            times = np.asarray(ppk_times, dtype=float)
            currs = np.asarray(currents, dtype=float)
            csv_path = meas_dir / "current.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["time_s", "current_uA"])
                for t, c in zip(times, currs):
                    writer.writerow([float(t), float(c)])
            payload["ppk_csv"] = str(csv_path)
            # compute total energy in J and insert explicitly
            if len(times) > 1:
                dt = float(np.mean(np.diff(times)))
            else:
                dt = 1.0 / DEFAULT_PPK_SAMPLE_RATE_HZ
            total_energy = float(np.sum(currs * 1e-6 * V_SUPPLY * dt))
            payload["total_energy_j"] = total_energy
        except Exception as e:
            print(f"[measure_max78000] Warning: failed to process PPK2 data: {e}")

    # If thermal frames and times present, compute per-frame max/avg and save timeseries CSV
    th_times = payload.get("thermal_times_s")
    th_pixels = payload.get("thermal_pixels")
    if th_times and th_pixels:
        try:
            times = np.asarray(th_times, dtype=float)
            maxs = []
            avgs = []
            for frame in th_pixels:
                arr = np.asarray(frame, dtype=float)
                finite = np.isfinite(arr)
                if np.any(finite):
                    flat = arr[finite]
                    maxs.append(float(np.max(flat)))
                    avgs.append(float(np.mean(flat)))
                else:
                    maxs.append(float("nan"))
                    avgs.append(float("nan"))
            csv_path = meas_dir / "thermal_timeseries.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["time_s", "max_temp_c", "avg_temp_c"])
                for t, mx, av in zip(times, maxs, avgs):
                    writer.writerow([float(t), mx, av])
            payload["thermal_timeseries_csv"] = str(csv_path)
            # also expose arrays as summary
            payload.setdefault("thermal_summary_series", {})
            payload["thermal_summary_series"]["time_s"] = list(times)
            payload["thermal_summary_series"]["max_temp_c"] = maxs
            payload["thermal_summary_series"]["avg_temp_c"] = avgs
        except Exception as e:
            print(f"[measure_max78000] Warning: failed to process thermal data: {e}")

    # Save any other simple array-like values as single-column CSVs
    for k in list(payload.keys()):
        if k in ("ppk", "ppk_csv", "thermal_pixels", "thermal_times_s", "thermal_summary_series"):
            continue
        v = payload.get(k)
        if isinstance(v, (list, tuple, np.ndarray)):
            try:
                arr = np.asarray(v)
                csv_path = meas_dir / f"{k}.csv"
                with open(csv_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    if arr.ndim == 1:
                        writer.writerow([k])
                        for x in arr:
                            writer.writerow([float(x)])
                    else:
                        # write column headers
                        cols = arr.shape[1]
                        header = [f"c{i}" for i in range(cols)]
                        writer.writerow(header)
                        for row in arr:
                            writer.writerow([float(x) for x in row])
                payload[k] = str(csv_path)
            except Exception as e:
                print(f"[measure_max78000] Warning: failed to save array {k} as CSV: {e}")

    # Copy any feedback images (if produced elsewhere) into iter dir and record paths
    saved_images: list[str] = []
    img_paths = payload.get("feedback_image_paths") or []
    for img in img_paths:
        try:
            src = Path(img)
            if src.exists():
                dst = iter_dir / src.name
                shutil.copy(src, dst)
                saved_images.append(str(dst))
        except Exception as e:
            print(f"[measure_max78000] Warning: failed to copy feedback image {img}: {e}")

    # Write measurement_result.json containing full measurement payload (including serial)
    measurement_json_path = iter_dir / "measurement_result.json"
    measurement_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Update iter_result.json to include only feedback (if present) and a short summary
    iter_json_path = iter_dir / "iter_result.json"
    existing: dict = {}
    if iter_json_path.exists():
        try:
            existing = json.loads(iter_json_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[measure_max78000] Warning: failed to read existing iter_result.json: {e}")
            existing = {}
    # Keep a concise measurement check record in iter_result.json.
    measure_item = {
        k: payload.get(k)
        for k in (
            "success",
            "score",
            "metric",
            "metric_value",
            "power",
            "thermal",
            "checkpoint",
            "judge",
            "stage",
            "error",
        )
        if k in payload and payload.get(k) is not None
    }
    if measure_item:
        existing["measure"] = measure_item

    existing["measure_summary"] = {
        k: payload.get(k)
        for k in ("ppk_csv", "total_energy_j", "thermal_timeseries_csv")
        if k in payload
    }
    if saved_images:
        existing.setdefault("feedback_image_paths", []).extend(saved_images)
    iter_json_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")


def check(state: RunState, agent_input: Input, yaml_input: YAMLInput) -> CheckResult:
    """Capture power, serial, and thermal from a pre-flashed MAX78000 DUT.

    Score is the normalized optimization_metric in [0.0, 1.0].

    Args:
        state: RunState containing sandbox and metadata.
        agent_input: Agent-settable parameters (baud_rate).
        yaml_input: YAML-controlled parameters (capture modes, duration, metric scaling).
    """
    fe = state.metadata.get("feedback_enabled", True)
    print(
        f"[measure_max78000] starting check — "
        f"capture_power={yaml_input.capture_power}, "
        f"capture_serial={yaml_input.capture_serial}, "
        f"capture_thermal={yaml_input.capture_thermal}, "
        f"duration_ms={yaml_input.duration_ms}, "
        f"ppk_sample_rate_hz={yaml_input.ppk_sample_rate_hz}, "
        f"metric={yaml_input.optimization_metric}"
    )
    if yaml_input.capture_power:
        assert (
            yaml_input.target_voltage_v > 0
        ), "target_voltage_v must be positive when capture_power=true"
    if yaml_input.capture_power and PPK2_MP is None:
        raise ExperimentSetupError(
            'ppk2-api is not installed; run `pip install -e ".[hardware]"`. '
            "The check refuses to fall back to simulated samples."
        )

    print("[measure_max78000] launching monitor threads...")
    try:
        monitor_result = _run_monitor(
            duration_ms=yaml_input.duration_ms,
            baud_rate=agent_input.baud_rate,
            capture_power=yaml_input.capture_power,
            capture_serial=yaml_input.capture_serial,
            capture_thermal=yaml_input.capture_thermal,
            ir_rate_hz=yaml_input.ir_rate_hz,
            target_voltage_v=yaml_input.target_voltage_v,
            ppk_sample_rate_hz=yaml_input.ppk_sample_rate_hz,
            verbose=True,
            show_ir_viewer=yaml_input.show_ir_viewer,
        )
        print("[measure_max78000] monitor threads finished.")
    except Exception as exc:
        print(f"[measure_max78000] monitor failed: {exc}")
        _save_measure_to_iter_result(
            state,
            {
                "success": False,
                "score": 0.0,
                "stage": "monitor_setup",
                "error": str(exc),
            },
        )
        return CheckResult(
            success=False,
            score=0.0,
            score_unit="measurement_error",
            feedback=(f"measurement failed: {exc}" if fe else None),
        )

    ppk_monitor = monitor_result.get("ppk_monitor")
    if ppk_monitor is not None:
        try:
            ppk_monitor.voltage_off()
        except Exception as e:
            print(f"Warning: failed to turn off PPK2 voltage after measurement: {e}")

    _, ppk_times, currents = monitor_result.get("ppk", ("", np.array([]), np.array([])))
    currents = np.asarray(currents, dtype=np.float64)
    ppk_times = np.asarray(ppk_times, dtype=np.float64)
    ppk_data = monitor_result.get("ppk_data", {})
    serial_text = monitor_result.get("serial_text", "")
    thermal_pixels = monitor_result.get("thermal_pixels", [])
    thermal_frame_metas = monitor_result.get("thermal_frame_metas", [])
    thermal_times_s = monitor_result.get("thermal_times_s", [])
    print(
        f"[measure_max78000] data collected — "
        f"power_samples={len(currents)}, "
        f"serial_chars={len(serial_text)}, "
        f"thermal_frames={len(thermal_pixels)}"
    )

    if yaml_input.capture_power and len(currents) == 0:
        _save_measure_to_iter_result(
            state,
            {
                "success": False,
                "score": 0.0,
                "stage": "ppk2_capture",
                "error": "PPK2 captured no samples",
                "serial": {"enabled": yaml_input.capture_serial, "text": serial_text},
            },
        )
        return CheckResult(
            success=False,
            score=0.0,
            score_unit="measurement_error",
            feedback=("PPK2 captured no samples" if fe else None),
        )

    if yaml_input.capture_thermal and not thermal_pixels:
        _save_measure_to_iter_result(
            state,
            {
                "success": False,
                "score": 0.0,
                "stage": "thermal_capture",
                "error": "thermal capture enabled but no IR frames received",
                "serial": {"enabled": yaml_input.capture_serial, "text": serial_text},
            },
        )
        return CheckResult(
            success=False,
            score=0.0,
            score_unit="measurement_error",
            feedback=(
                "thermal capture enabled but no IR frames received" if fe else None
            ),
        )

    try:
        # Collect metric values
        stats = ppk_data.get("stats", {})
        avg_uA = stats.get(
            "avg_uA", float(np.mean(currents)) if len(currents) > 0 else 0.0
        )
        peak_uA = stats.get(
            "peak_uA", float(np.max(currents)) if len(currents) > 0 else 0.0
        )
        sr_eff = float(
            stats.get("sample_rate_hz", yaml_input.ppk_sample_rate_hz)
            or yaml_input.ppk_sample_rate_hz
        )
        power = (
            _power_summary(currents, sample_rate_hz=sr_eff) if len(currents) > 0 else {}
        )
        total_energy_j = power.get("total_energy_j", 0.0)

        thermal_summary, _ = (
            _summarize_thermal(list(thermal_pixels)) if thermal_pixels else (None, "")
        )
        peak_temp_c = thermal_summary["peak_c"] if thermal_summary else 0.0
        avg_temp_c = thermal_summary["avg_c"] if thermal_summary else 0.0

        # Per-frame thermal stats (time-series for agent feedback and CSV)
        # Using cropped region stats from metadata when available
        thermal_per_frame: list[dict] = []
        if thermal_pixels and thermal_times_s:
            consistent_frames = [
                f
                for f in thermal_pixels
                if isinstance(f, np.ndarray) and f.shape == thermal_pixels[0].shape
            ]
            for frame, t_s in zip(
                consistent_frames, thermal_times_s[: len(consistent_frames)]
            ):
                finite = frame[np.isfinite(frame)]
                if len(finite) > 0:
                    thermal_per_frame.append(
                        {
                            "time_s": float(t_s),
                            "max_temp_c": float(np.max(finite)),
                            "avg_temp_c": float(np.mean(finite)),
                        }
                    )

        # Validate metric name
        metric_values: dict[str, float] = {
            "total_energy_j": total_energy_j,
            "avg_uA": avg_uA,
            "peak_uA": peak_uA,
            "peak_temp_c": peak_temp_c,
            "avg_temp_c": avg_temp_c,
        }
        metric = yaml_input.optimization_metric
        if metric not in metric_values:
            _save_measure_to_iter_result(
                state,
                {
                    "success": False,
                    "score": 0.0,
                    "stage": "config",
                    "error": f"unknown optimization_metric '{metric}'",
                    "valid_metrics": sorted(metric_values.keys()),
                },
            )
            return CheckResult(
                success=False,
                score=0.0,
                score_unit="config_error",
                feedback=(
                    (
                        f"optimization_metric '{metric}' is not a recognized metric. "
                        f"Valid options: {sorted(metric_values.keys())}"
                    )
                    if fe
                    else None
                ),
            )
        metric_value = metric_values[metric]
        print(
            f"[measure_max78000] metric={metric}, value={metric_value:.6g}, metric_values={metric_values}"
        )

        # Scoring: checkpoint check → LLM judge → metric score
        judge_reason = ""
        judge_pass = False
        if yaml_input.capture_serial:
            checkpoint_found = CHECKPOINT_MSG in serial_text.lower()
            print(
                f"[measure_max78000] checkpoint {'FOUND' if checkpoint_found else 'NOT FOUND'}"
            )
            if not checkpoint_found:
                score = -1.0
                success = False
                judge_reason = f"checkpoint '{CHECKPOINT_MSG}' not found in UART output"
            else:
                # Inform the judge that the checkpoint was observed and therefore
                # the judge is being triggered to validate firmware behavior.
                checkpoint_note = (
                    f"NOTE: checkpoint '{CHECKPOINT_MSG}' was found in the UART output; "
                    "triggering the LLM judge."
                )
                judge_pass, judge_reason = _judge_firmware_behavior(
                    task_description=yaml_input.task_description,
                    firmware_behavior_description=(checkpoint_note + "\n\n" + agent_input.firmware_behavior_description),
                    serial_output=serial_text,
                )
                if not judge_pass:
                    score = -1.0
                    success = False
                else:
                    score = _compute_metric_score(
                        metric_value,
                        yaml_input.metric_min,
                        yaml_input.metric_max,
                        yaml_input.lower_is_better,
                    )
                    success = True
        else:
            checkpoint_found = False
            score = _compute_metric_score(
                metric_value,
                yaml_input.metric_min,
                yaml_input.metric_max,
                yaml_input.lower_is_better,
            )
            success = True
            judge_reason = "firmware behavior not evaluated"

        print(
            f"[measure_max78000] scoring complete — success={success}, score={score:.4f}, judge_reason={judge_reason!r}"
        )

        # Build feedback
        scalar_metrics: dict[str, Any] = {}
        if yaml_input.capture_power:
            scalar_metrics["avg_uA"] = avg_uA
            scalar_metrics["peak_uA"] = peak_uA
            scalar_metrics["total_energy_j"] = total_energy_j
            scalar_metrics["power_samples"] = int(len(currents))
            scalar_metrics["ppk_sample_rate_hz"] = float(sr_eff)
        if thermal_summary:
            scalar_metrics["peak_temp_c"] = peak_temp_c
            scalar_metrics["avg_temp_c"] = avg_temp_c
        if thermal_per_frame:
            max_series = [r["max_temp_c"] for r in thermal_per_frame]
            avg_series = [r["avg_temp_c"] for r in thermal_per_frame]
            scalar_metrics["abs_max_temp_c"] = float(np.max(max_series))
            scalar_metrics["avg_max_temp_c"] = float(np.mean(max_series))
            scalar_metrics["thermal_frames"] = len(thermal_per_frame)

        thermal_ts_section = ""
        if thermal_per_frame:
            rows = ["time_s,max_temp_c,avg_temp_c"] + [
                f"{r['time_s']:.3f},{r['max_temp_c']:.2f},{r['avg_temp_c']:.2f}"
                for r in thermal_per_frame
            ]
            thermal_ts_section = (
                "## Thermal Time-series (°C per frame)\n```\n"
                + "\n".join(rows)
                + "\n```\n\n"
            )

        hw_status_lines = [
            f"capture_power={yaml_input.capture_power}, capture_serial={yaml_input.capture_serial}, capture_thermal={yaml_input.capture_thermal}",
            f"duration_ms={yaml_input.duration_ms}, optimization_metric={metric}, metric_value={metric_value:.6g}",
        ]
        hw_section = "## Hardware Status\n" + "\n".join(hw_status_lines) + "\n\n"

        if yaml_input.capture_serial:
            behavior_section = (
                f"## Firmware Behavior\n"
                f"Checkpoint: {'FOUND' if checkpoint_found else 'NOT FOUND'}\n"
                + (
                    f"Judge: {'PASS' if success else 'FAIL'}: {judge_reason}\n\n"
                    if checkpoint_found
                    else f"{judge_reason}\n\n"
                )
            )
        else:
            behavior_section = ""

        feedback = (
            f"{hw_section}"
            f"## Measurement Results\n{json.dumps(scalar_metrics, indent=2)}\n\n"
            f"{thermal_ts_section}"
            f"{behavior_section}"
            f"## UART Serial Output\n{serial_text.strip() or '(none)'}"
        )
        agent_feedback = feedback if fe else None

        print(f"[measure_max78000] saving results to iter dir")
        feedback_image_paths: list[str] = []
        output_dir_str = state.metadata.get("output_dir")
        if output_dir_str:
            iter_dir = (
                Path(output_dir_str)
                / f"trial_{state.trial_index}"
                / f"iter_{state.iteration_index}"
            )
            iter_dir.mkdir(parents=True, exist_ok=True)
            data_dir = iter_dir / "measurement_data"
            data_dir.mkdir(exist_ok=True)

            if (
                yaml_input.capture_power
                and len(ppk_times) > 0
                and len(currents) > 0
            ):
                with open(data_dir / "current.csv", "w", newline="", encoding="utf-8") as fh:
                    fh.write("time_s,current_uA\n")
                    for t, c in zip(ppk_times, currents):
                        fh.write(f"{t:.6f},{c:.6f}\n")

            times_5hz = np.asarray(ppk_data.get("times_5hz", []), dtype=np.float64)
            currents_5hz = np.asarray(
                ppk_data.get("currents_5hz", []), dtype=np.float64
            )
            if (
                yaml_input.capture_power
                and len(times_5hz) > 0
                and len(currents_5hz) > 0
            ):
                with open(
                    data_dir / "power.csv", "w", newline="", encoding="utf-8"
                ) as fh:
                    fh.write("time_s,current_uA\n")
                    for t, c in zip(times_5hz, currents_5hz):
                        fh.write(f"{t:.6f},{c:.4f}\n")

            # Thermal per-frame CSV
            if thermal_per_frame:
                with open(
                    data_dir / "thermal_per_frame.csv",
                    "w",
                    newline="",
                    encoding="utf-8",
                ) as fh:
                    fh.write("time_s,max_temp_c,avg_temp_c\n")
                    for r in thermal_per_frame:
                        fh.write(
                            f"{r['time_s']:.3f},{r['max_temp_c']:.3f},{r['avg_temp_c']:.3f}\n"
                        )

            # Plots
            try:
                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                if yaml_input.capture_power and len(times_5hz) > 0:
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(times_5hz, currents_5hz, lw=0.6, color="steelblue")
                    ax.set_xlabel("Time (s)")
                    ax.set_ylabel("Current (µA)")
                    ax.set_title("Power Consumption Over Time")
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    p = iter_dir / "power_over_time.png"
                    fig.savefig(p, dpi=100)
                    plt.close(fig)
                    feedback_image_paths.append(str(p))

                if thermal_per_frame:
                    t_vals = [r["time_s"] for r in thermal_per_frame]
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(
                        t_vals,
                        [r["max_temp_c"] for r in thermal_per_frame],
                        label="max_temp_c",
                        color="crimson",
                        lw=1.2,
                    )
                    ax.plot(
                        t_vals,
                        [r["avg_temp_c"] for r in thermal_per_frame],
                        label="avg_temp_c",
                        color="darkorange",
                        lw=1.2,
                    )
                    ax.set_xlabel("Time (s)")
                    ax.set_ylabel("Temperature (°C)")
                    ax.set_title("Chip Temperature Over Time")
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    p = iter_dir / "thermal_over_time.png"
                    fig.savefig(p, dpi=100)
                    plt.close(fig)
                    feedback_image_paths.append(str(p))
            except Exception as plot_exc:
                print(f"[measure_max78000] plot generation failed: {plot_exc}")

            # Write measurement_result.json (full, incl. serial) and update iter_result.json
            # (serial excluded from iter_result — _save_measure_to_iter_result picks
            #  only the keys listed in its measure_item block).
            power_dict: dict[str, Any] = {}
            if yaml_input.capture_power:
                power_dict = {
                    "avg_uA": avg_uA,
                    "peak_uA": peak_uA,
                    "total_energy_j": total_energy_j,
                    "power_samples": int(len(currents)),
                    "ppk_sample_rate_hz": float(sr_eff),
                    "current_csv": str(data_dir / "current.csv"),
                }
            measurement_result = {
                "power": power_dict,
                "thermal": {
                    "enabled": yaml_input.capture_thermal,
                    "frames": len(thermal_pixels),
                    "peak_temp_c": peak_temp_c,
                    "avg_temp_c": avg_temp_c,
                    "abs_max_temp_c": float(np.max([r["max_temp_c"] for r in thermal_per_frame])) if thermal_per_frame else None,
                    "avg_max_temp_c": float(np.mean([r["max_temp_c"] for r in thermal_per_frame])) if thermal_per_frame else None,
                    "thermal_timeseries_csv": str(data_dir / "thermal_per_frame.csv") if thermal_per_frame else None,
                },
                "serial": {
                    "enabled": yaml_input.capture_serial,
                    "text": serial_text,
                },
                "checkpoint": {
                    "found": checkpoint_found if yaml_input.capture_serial else None,
                    "message": CHECKPOINT_MSG,
                },
                "judge": {"reason": judge_reason},
                "score": score,
                "metric": metric,
                "metric_value": metric_value,
            }
            (iter_dir / "measurement_result.json").write_text(
                json.dumps(measurement_result, indent=2), encoding="utf-8"
            )

            # iter_result.json — no serial text, includes feedback string for traceability
            iter_measure = {
                k: v for k, v in measurement_result.items() if k != "serial"
            }
            _save_measure_to_iter_result(
                state, {"success": success, "feedback": agent_feedback, **iter_measure}
            )

        return CheckResult(
            success=success,
            score=score,
            score_unit=f"metric_score({metric})",
            feedback=agent_feedback,
            feedback_image_paths=feedback_image_paths,
        )

    except Exception as exc:
        err_trace = tb.format_exc()
        print(f"[measure_max78000] post-capture processing error:\n{err_trace}")
        _save_measure_to_iter_result(
            state,
            {
                "success": False,
                "score": 0.0,
                "stage": "processing",
                "error": str(exc),
                "traceback": err_trace,
                "serial": {"enabled": yaml_input.capture_serial, "text": serial_text},
                "power_samples": int(len(currents)),
                "thermal_frames": int(len(thermal_pixels)),
            },
        )
        return CheckResult(
            success=False,
            score=0.0,
            score_unit="processing_error",
            feedback=(
                (f"post-capture processing failed: {exc}\n\n{err_trace}")
                if fe
                else None
            ),
        )
