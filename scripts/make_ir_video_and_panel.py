#!/usr/bin/env python3
"""Create an IR video and a 10-frame SVG panel from saved .npy frames.

Supported input formats in --data-dir:
1) New combined file: raw_frames.npy
   dtype: [("ts", uint64), ("frame", float32, (24, 32))]
2) Legacy files: frame_<timestamp_ms>.npy (one frame per file)

Outputs under ``<data-dir>/plot/``:
- ir_capture_video.mp4 (full recording, all frames)
- ir_capture_video_window.mp4 (frames only within [START_OFFSET_S, START_OFFSET_S+WINDOW_SECONDS])
- ir_10frame_panel.svg
- ir_10frame_panel_hotspot.svg (same panel + red + at hottest pixel per frame)
- ir_10frame_panel.json (10 snapshots: time_s + frame as nested lists, °C)
- ir_10frame_hotspot.csv (per snapshot: time_s, max_temp_c, x, y of hottest pixel; x=col, y=row)

Usage:
    python make_ir_video_and_panel.py --data-dir /path/to/ir_capture_YYYYMMDD_HHMMSS
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# --------------------------- Constants to tweak ----------------------------
# Time window starts at (first_frame_time + START_OFFSET_S), lasts 45 seconds.
START_OFFSET_S = 16.0
WINDOW_SECONDS = 45.0
PANEL_FRAME_COUNT = 10

# Output subdirectory under --data-dir (keeps mp4/svg out of the raw .npy folder root).
PLOT_SUBDIR = "plot"

# Output filenames (written under PLOT_DIR).
VIDEO_FILENAME_FULL = "ir_capture_video.mp4"
VIDEO_FILENAME_WINDOW = "ir_capture_video_window.mp4"
SVG_FILENAME = "ir_10frame_panel.svg"
SVG_HOTSPOT_FILENAME = "ir_10frame_panel_hotspot.svg"
JSON_10FRAME_FILENAME = "ir_10frame_panel.json"
CSV_10FRAME_HOTSPOT_FILENAME = "ir_10frame_hotspot.csv"

# Video settings
VIDEO_FPS = 15
VIDEO_SCALE = 12  # upscale 32x24 thermal frame to larger video image

# Iron-style thermal palette (same as ir_camera / test_ir_camera: matplotlib inferno).
THERMAL_COLORMAP = "inferno"
# Colorbar tick spacing (°C); only multiples of this within [vmin, vmax] are labeled.
COLORBAR_TICK_STEP_C = 5
# 10-frame SVG panel layout (wider figure + right inset so colorbar labels are not clipped).
PANEL_FIGSIZE = (18.0, 7.5)
PANEL_GS_LEFT = 0.05
PANEL_GS_RIGHT = 0.94
# SVG panel only: fixed color scale and colorbar ticks (°C).
PANEL_SVG_VMIN = 20.0
PANEL_SVG_VMAX = 60.0
PANEL_SVG_TICK_STEP_C = 5   # tick marks every N °C
PANEL_SVG_LABEL_STEP_C = 10  # numeric labels every N °C (multiple of tick step)
# ---------------------------------------------------------------------------

LEGACY_RE = re.compile(r"^frame_(\d+)\.npy$")
# This file lives in IR_cam/; ir_camera.py is in the parent hardware/ directory.
_HARDWARE_DIR = Path(__file__).resolve().parent.parent
if str(_HARDWARE_DIR) not in sys.path:
    sys.path.insert(0, str(_HARDWARE_DIR))

from ir_camera import iron_color_image_with_overlay  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build IR videos and 10-frame panel plots from a capture directory.",
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Capture folder containing raw_frames.npy or frame_<timestamp_ms>.npy",
    )
    return p.parse_args()


def _load_frames_and_timestamps(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return timestamps_seconds, frames float32[frame_count, h, w]."""
    combined_path = data_dir / "raw_frames.npy"
    if combined_path.exists():
        arr = np.load(combined_path, allow_pickle=False)
        if arr.dtype.names is None or "ts" not in arr.dtype.names or "frame" not in arr.dtype.names:
            raise ValueError(f"{combined_path} is present but has unexpected dtype: {arr.dtype}")
        ts_ms = np.asarray(arr["ts"], dtype=np.float64)
        frames = np.asarray(arr["frame"], dtype=np.float32)
        return ts_ms / 1000.0, frames

    # Legacy mode: one frame per file, timestamp embedded in filename.
    ts_ms_list: list[int] = []
    frame_list: list[np.ndarray] = []
    for p in sorted(data_dir.iterdir()):
        m = LEGACY_RE.match(p.name)
        if not m:
            continue
        ts_ms_list.append(int(m.group(1)))
        frame_list.append(np.load(p, allow_pickle=False).astype(np.float32, copy=False))

    if not frame_list:
        raise FileNotFoundError(
            "No input data found. Expected raw_frames.npy or frame_<timestamp>.npy files."
        )

    ts_ms = np.asarray(ts_ms_list, dtype=np.float64)
    frames = np.stack(frame_list, axis=0).astype(np.float32, copy=False)
    return ts_ms / 1000.0, frames


def _norm_limits(frames: np.ndarray) -> tuple[float, float]:
    finite = frames[np.isfinite(frames)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    if vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax


def _window_frame_indices(ts_s: np.ndarray, start_s: float, end_s: float) -> np.ndarray:
    """Indices of all frames with start_s <= timestamp <= end_s (relative to first frame)."""
    mask = (ts_s >= start_s) & (ts_s <= end_s)
    return np.flatnonzero(mask)


def _select_equal_interval_indices(
    ts_s: np.ndarray,
    start_s: float,
    end_s: float,
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    targets = np.linspace(start_s, end_s, count, endpoint=True)
    idx = np.searchsorted(ts_s, targets, side="left")
    idx = np.clip(idx, 0, len(ts_s) - 1)
    rel_times = ts_s[idx] - start_s
    return idx, rel_times


def _frame_to_rgb(frame: np.ndarray, cmap, vmin: float, vmax: float) -> np.ndarray:
    denom = max(vmax - vmin, 1e-6)
    normalized = np.clip((frame - vmin) / denom, 0.0, 1.0)
    rgba = cmap(normalized)
    return (rgba[..., :3] * 255.0).astype(np.uint8)


def _hottest_pixel(frame: np.ndarray) -> tuple[int, int] | None:
    """Return (row, col) of the hottest finite pixel, or None if no finite data."""
    finite = np.isfinite(frame)
    if not np.any(finite):
        return None
    masked = np.where(finite, frame, -np.inf)
    idx = int(np.argmax(masked))
    r, c = np.unravel_index(idx, frame.shape)
    return int(r), int(c)


def _overlay_hot_temp_text(rgb: np.ndarray, temp_c: float) -> np.ndarray:
    """Overlay hottest-pixel temperature text on frame."""
    out = rgb.copy()
    pil = Image.fromarray(out)
    draw = ImageDraw.Draw(pil)
    font = ImageFont.load_default()
    text = f"Hot: {temp_c:.1f}C"
    x_text, y_text = 8, 8
    draw.text((x_text + 1, y_text + 1), text, fill=(0, 0, 0), font=font)
    draw.text((x_text, y_text), text, fill=(255, 255, 255), font=font)
    return np.asarray(pil)


def _ticks_multiples_in_range(
    vmin: float, vmax: float, step: float
) -> np.ndarray:
    """Values x = n * step (n integer) with vmin <= x <= vmax."""
    if (
        not math.isfinite(vmin)
        or not math.isfinite(vmax)
        or not math.isfinite(step)
        or step <= 0
        or vmax < vmin
    ):
        return np.array([], dtype=np.float64)
    lo = math.ceil(vmin / step) * step
    hi = math.floor(vmax / step) * step
    if lo > hi + 1e-12:
        return np.array([], dtype=np.float64)
    ticks = np.arange(lo, hi + step * 0.5, step, dtype=np.float64)
    return ticks[(ticks >= vmin - 1e-9) & (ticks <= vmax + 1e-9)]


def _format_temp_tick_value(v: float, span: float) -> str:
    """Format °C bar labels so endpoints match vmin/vmax (avoid .0f rounding 24.6→25)."""
    if not math.isfinite(v) or not math.isfinite(span):
        return "nan"
    if span <= 0:
        return format(v, ".2f").rstrip("0").rstrip(".")
    # Integer ticks only on very wide spans; otherwise keep ≥1 decimal for typical IR ranges.
    if span >= 50:
        decimals = 0
    elif span >= 8:
        decimals = 1
    else:
        decimals = 2
    s = format(v, f".{decimals}f")
    if decimals > 0:
        s = s.rstrip("0").rstrip(".")
    return s


def _value_to_bar_y(v: float, vmin: float, vmax: float, y0: int, y1: int) -> int:
    """Map temperature v to pixel y (top = vmax, bottom = vmin)."""
    denom = max(vmax - vmin, 1e-9)
    frac = (v - vmin) / denom
    return int(round(y1 - frac * (y1 - y0)))


def _overlay_colorbar(rgb: np.ndarray, cmap, vmin: float, vmax: float) -> np.ndarray:
    """Overlay a vertical color bar; fills full frame height; ticks at multiples of 5 °C."""
    out = rgb.copy()
    h, w = out.shape[:2]
    if h < 20 or w < 20:
        return out

    bar_w = max(12, int(round(w * 0.06)))
    # Leave room for tick marks + numeric labels on the right.
    label_reserve = max(56, int(round(w * 0.21)))
    x1 = w - label_reserve
    x0 = max(0, x1 - bar_w)
    y0 = 0
    y1 = h
    if x1 <= x0 or y1 <= y0:
        return out

    nrows = y1 - y0
    temps = np.linspace(vmax, vmin, nrows, dtype=np.float32)[:, None]
    normed = np.clip((temps - vmin) / max(vmax - vmin, 1e-9), 0.0, 1.0)
    bar_rgb = (cmap(normed)[..., :3] * 255.0).astype(np.uint8)
    bar_rgb = np.repeat(bar_rgb, x1 - x0, axis=1)
    out[y0:y1, x0:x1] = bar_rgb

    # 1 px border around the color column (does not shrink the gradient height).
    out[y0:y1, x0] = 0
    out[y0:y1, x1 - 1] = 0
    out[y0, x0:x1] = 0
    out[y1 - 1, x0:x1] = 0

    span = float(vmax - vmin)
    tick_vals = _ticks_multiples_in_range(vmin, vmax, float(COLORBAR_TICK_STEP_C))

    pil = Image.fromarray(out)
    draw = ImageDraw.Draw(pil)
    font = ImageFont.load_default()
    tick_right_x = x1 + 1
    major_len = max(6, int(round(bar_w * 0.45)))
    label_x = tick_right_x + major_len + 3

    for v in tick_vals:
        yy = _value_to_bar_y(float(v), vmin, vmax, y0, y1)
        yy = int(np.clip(yy, y0 + 1, y1 - 2))
        draw.line(
            [(tick_right_x, yy), (tick_right_x + major_len, yy)],
            fill=(0, 0, 0),
            width=2,
        )
        text = _format_temp_tick_value(float(v), span)
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            th = bbox[3] - bbox[1]
        except Exception:
            th = draw.textsize(text, font=font)[1]
        y_text = int(np.clip(yy - th // 2, 0, h - th))
        draw.text((label_x, y_text), text, fill=(0, 0, 0), font=font)

    return np.asarray(pil)


def _write_video(frames: np.ndarray, vmin: float, vmax: float, out_path: Path) -> None:
    cmap = matplotlib.colormaps[THERMAL_COLORMAP]
    try:
        import imageio.v2 as imageio
    except Exception as e:
        raise RuntimeError(
            "Video export requires imageio. Install with: pip install imageio imageio-ffmpeg"
        ) from e

    with imageio.get_writer(str(out_path), fps=VIDEO_FPS, codec="libx264") as writer:
        for frame in frames:
            hp = _hottest_pixel(frame)
            hot_temp = float("nan")
            if hp is not None:
                hot_temp = float(frame[hp[0], hp[1]])

            out_size = (frame.shape[1] * VIDEO_SCALE, frame.shape[0] * VIDEO_SCALE)
            img = iron_color_image_with_overlay(
                frame,
                max_temp_pixel=hp,
                vmin=vmin,
                vmax=vmax,
                out_size=out_size,
                cmap_name=THERMAL_COLORMAP,
            )
            rgb = np.asarray(img, dtype=np.uint8)
            if np.isfinite(hot_temp):
                rgb = _overlay_hot_temp_text(rgb, temp_c=hot_temp)
            rgb = _overlay_colorbar(rgb, cmap=cmap, vmin=vmin, vmax=vmax)
            writer.append_data(rgb)


def _make_panel_svg(
    frames: np.ndarray,
    ts_s: np.ndarray,
    idx: np.ndarray,
    rel_times: np.ndarray,
    vmin: float,
    vmax: float,
    out_path: Path,
    *,
    hotspot_cross: bool = False,
) -> None:
    # Thermal scale for SVG panels is fixed (PANEL_SVG_*); vmin/vmax only for caller compatibility.
    _ = (vmin, vmax)
    # Dedicated last column for colorbar so it never overlaps the image grid.
    fig = plt.figure(figsize=PANEL_FIGSIZE)
    gs = gridspec.GridSpec(
        2,
        6,
        figure=fig,
        width_ratios=[1, 1, 1, 1, 1, 0.11],
        wspace=0.32,
        hspace=0.72,
        left=PANEL_GS_LEFT,
        right=PANEL_GS_RIGHT,
        top=0.90,
        bottom=0.16,
    )
    cmap = matplotlib.colormaps[THERMAL_COLORMAP]
    cb_vmin = float(PANEL_SVG_VMIN)
    cb_vmax = float(PANEL_SVG_VMAX)

    for k in range(PANEL_FRAME_COUNT):
        row, col = divmod(k, 5)
        ax = fig.add_subplot(gs[row, col])
        i = int(idx[k])
        ax.imshow(
            frames[i],
            cmap=cmap,
            vmin=cb_vmin,
            vmax=cb_vmax,
            interpolation="nearest",
        )
        ax.set_title(f"{int(round(rel_times[k]))} s", fontsize=10, pad=6)
        ax.set_xticks([])
        ax.set_yticks([])

        hp = _hottest_pixel(frames[i])
        if hp is not None:
            r, c = hp
            tmax = float(frames[i][r, c])
            caption = f"max {tmax:.1f} °C\nx={c}  y={r}"
            if hotspot_cross:
                # Match test_ir_camera live view: red plus at (col, row) in image coords.
                ax.plot(
                    c,
                    r,
                    "r+",
                    markersize=8,
                    markeredgewidth=1.5,
                    clip_on=False,
                )
        else:
            caption = "max —\nx=—  y=—"
        ax.text(
            0.5,
            -0.22,
            caption,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8,
            linespacing=1.15,
        )

    cax = fig.add_subplot(gs[:, 5])
    ni = 256
    col = np.linspace(cb_vmin, cb_vmax, ni, dtype=np.float64).reshape(-1, 1)
    cax.imshow(
        col,
        aspect="auto",
        cmap=cmap,
        vmin=cb_vmin,
        vmax=cb_vmax,
        origin="lower",
        extent=[0, 1, cb_vmin, cb_vmax],
    )
    cax.set_xlim(0, 1)
    cax.set_ylim(cb_vmin, cb_vmax)
    cax.set_xticks([])
    cax.margins(0)
    ts = float(PANEL_SVG_TICK_STEP_C)
    ls = float(PANEL_SVG_LABEL_STEP_C)
    major = np.arange(cb_vmin, cb_vmax + ts * 0.5, ls, dtype=np.float64)
    minor = np.arange(cb_vmin + ts, cb_vmax, ls, dtype=np.float64)
    cax.yaxis.set_major_locator(mticker.FixedLocator(major))
    cax.yaxis.set_minor_locator(mticker.FixedLocator(minor))
    cax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _p: f"{int(round(v))}")
    )
    cax.yaxis.set_minor_formatter(mticker.NullFormatter())
    cax.yaxis.set_ticks_position("right")
    cax.yaxis.set_label_position("right")
    cax.tick_params(
        axis="y",
        which="major",
        labelsize=9,
        colors="black",
        length=6,
        width=1.0,
    )
    cax.tick_params(
        axis="y",
        which="minor",
        length=3,
        width=0.8,
        colors="black",
        labelleft=False,
        labelright=False,
    )
    for lb in cax.get_yticklabels():
        lb.set_color("black")
    cax.set_ylabel("Temp (°C)", fontsize=10, color="black")
    for spine in cax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_edgecolor("black")

    title = (
        f"IR snapshots from +{START_OFFSET_S:.1f}s to +{START_OFFSET_S + WINDOW_SECONDS:.1f}s"
    )
    if hotspot_cross:
        title += " (hottest pixel marked)"
    fig.suptitle(title, fontsize=12, y=0.98)
    fig.savefig(out_path, format="svg")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    if not data_dir.is_dir():
        raise SystemExit(f"Not a directory: {data_dir}")

    plot_dir = data_dir / PLOT_SUBDIR
    plot_dir.mkdir(parents=True, exist_ok=True)

    ts_s, frames = _load_frames_and_timestamps(data_dir)
    if len(ts_s) != len(frames):
        raise ValueError("timestamps and frames count mismatch")
    if len(ts_s) < PANEL_FRAME_COUNT:
        raise ValueError(f"Need at least {PANEL_FRAME_COUNT} frames, got {len(ts_s)}")

    # Normalize time to the first frame as requested.
    t0 = float(ts_s[0])
    ts_s = ts_s - t0

    start_s = START_OFFSET_S
    end_s = start_s + WINDOW_SECONDS
    if end_s > float(ts_s[-1]):
        raise ValueError(
            f"Requested window ends at {end_s:.2f}s but data ends at {ts_s[-1]:.2f}s. "
            "Lower START_OFFSET_S or record longer data."
        )

    win_idx = _window_frame_indices(ts_s, start_s, end_s)
    if win_idx.size == 0:
        raise ValueError(
            f"No frames in target window [{start_s:.2f}s, {end_s:.2f}s]. "
            "Check START_OFFSET_S and WINDOW_SECONDS."
        )
    window_frames = frames[win_idx]
    vmin_full, vmax_full = _norm_limits(frames)
    vmin, vmax = _norm_limits(window_frames)

    _write_video(
        frames, vmin=vmin_full, vmax=vmax_full, out_path=plot_dir / VIDEO_FILENAME_FULL
    )
    _write_video(
        window_frames,
        vmin=vmin,
        vmax=vmax,
        out_path=plot_dir / VIDEO_FILENAME_WINDOW,
    )

    idx, rel_times = _select_equal_interval_indices(
        ts_s=ts_s,
        start_s=start_s,
        end_s=end_s,
        count=PANEL_FRAME_COUNT,
    )
    _make_panel_svg(
        frames=frames,
        ts_s=ts_s,
        idx=idx,
        rel_times=rel_times,
        vmin=vmin,
        vmax=vmax,
        out_path=plot_dir / SVG_FILENAME,
        hotspot_cross=False,
    )
    _make_panel_svg(
        frames=frames,
        ts_s=ts_s,
        idx=idx,
        rel_times=rel_times,
        vmin=vmin,
        vmax=vmax,
        out_path=plot_dir / SVG_HOTSPOT_FILENAME,
        hotspot_cross=True,
    )

    hh, ww = int(frames.shape[1]), int(frames.shape[2])
    snapshots: list[dict[str, object]] = []
    for k in range(PANEL_FRAME_COUNT):
        i = int(idx[k])
        snapshots.append({
            "time_s": float(ts_s[i]),
            "frame": frames[i].astype(float).tolist(),
        })
    panel_json = {
        "time_s_note": "seconds since first frame of recording (same convention as this script)",
        "frame_note": "temperatures in °C; frame[row][col] with shape [height, width]",
        "height": hh,
        "width": ww,
        "snapshots": snapshots,
    }
    json_path = plot_dir / JSON_10FRAME_FILENAME
    with json_path.open("w", encoding="utf-8") as jf:
        json.dump(panel_json, jf, indent=2)

    csv_path = plot_dir / CSV_10FRAME_HOTSPOT_FILENAME
    with csv_path.open("w", newline="") as cf:
        writer = csv.writer(cf)
        writer.writerow(["time_s", "max_temp_c", "x", "y"])
        for k in range(PANEL_FRAME_COUNT):
            i = int(idx[k])
            fr = frames[i]
            t_snap = float(ts_s[i])
            hp = _hottest_pixel(fr)
            if hp is None:
                writer.writerow([f"{t_snap:.6f}", "", "", ""])
            else:
                row, col = hp
                writer.writerow(
                    [f"{t_snap:.6f}", f"{float(fr[row, col]):.4f}", col, row]
                )

    print(f"Loaded frames: {len(frames)}")
    print(f"Window frames used for window video: {win_idx.size} "
          f"([{start_s:.1f}s, {end_s:.1f}s] relative to first frame)")
    print(f"Data directory: {data_dir}")
    print(f"Saved video (full): {plot_dir / VIDEO_FILENAME_FULL}")
    print(f"Saved video (window): {plot_dir / VIDEO_FILENAME_WINDOW}")
    print(f"Saved panel: {plot_dir / SVG_FILENAME}")
    print(f"Saved panel (hotspot): {plot_dir / SVG_HOTSPOT_FILENAME}")
    print(f"Saved 10-frame json: {plot_dir / JSON_10FRAME_FILENAME}")
    print(f"Saved hotspot csv: {plot_dir / CSV_10FRAME_HOTSPOT_FILENAME}")


if __name__ == "__main__":
    main()
