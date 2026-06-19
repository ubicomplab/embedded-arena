"""Manual test / live viewer for the IRCamera driver.

Streams frames from the ESP32-S3 + MLX90640 until Ctrl+C, saving an mp4 and
a live-updating latest.png, and (by default) popping a matplotlib window
that refreshes as frames arrive. Timestamped per-frame PNGs are skipped by
default to save disk space when a video is being recorded.

Usage:
    python test_ir_camera.py                         # stream until Ctrl+C, save mp4, show viewer
    python test_ir_camera.py --rate 8                # 8 Hz sensor rate
    python test_ir_camera.py --port /dev/tty.usbmodem1234
    python test_ir_camera.py --duration 10           # timed run instead of Ctrl+C
    python test_ir_camera.py --no-video              # skip mp4, images only
    python test_ir_camera.py --save-images           # also keep per-frame PNGs
    python test_ir_camera.py --no-view               # run headless (no GUI window)

The viewer window uses matplotlib's default backend; closing the window
triggers a graceful shutdown the same way Ctrl+C does. As a fallback, the
driver still writes output/<run>/latest.png, which you can open in any
hot-reload viewer (macOS Preview, VS Code Image Preview, etc.).
The mp4 is finalised on clean exit and on Ctrl+C.
"""

from __future__ import annotations

import argparse
import numbers
import os
import signal
import sys
import threading
import time
from datetime import datetime

# Make the parent tools/ directory importable so we can load ir_camera.
HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS_DIR = os.path.abspath(os.path.join(HERE, ".."))
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

# All run artefacts land under IR_cam/output/ so the repo root stays clean
# (see .gitignore). The directory is created on demand by the driver.
DEFAULT_OUTPUT_ROOT = os.path.join(HERE, "output")

import numpy as np  # noqa: E402

# Pre-import matplotlib on the main thread so the driver's stream thread can
# later do `import matplotlib.cm` without racing a concurrent import here and
# hitting the "partially initialized module" circular-import error. Tolerate
# absence so the test script still runs headless on machines without mpl.
try:
    import matplotlib  # noqa: E402, F401
    import matplotlib.cm  # noqa: E402, F401
    import matplotlib.pyplot  # noqa: E402, F401
except Exception:
    pass

from ir_camera import (  # noqa: E402
    DEFAULT_BAUD,
    FRAME_H,
    FRAME_W,
    CROP_LEFT,
    CROP_RIGHT,
    CROP_TOP,
    CROP_BOTTOM,
    IRCamera,
    VALID_RATES,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="IRCamera live stream / smoke test")
    p.add_argument("--port", default=None, help="Serial port (auto-detect if omitted)")
    p.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    p.add_argument("--rate", type=float, default=32.0,
                   help=f"Refresh rate in Hz (valid: {VALID_RATES}); 16 Hz -> ~8 fps in CHESS mode")
    p.add_argument("--duration", type=float, default=None,
                   help="Stop after this many seconds (default: stream until Ctrl+C)")
    p.add_argument("--save_dir", default=None,
                   help="Output directory (default: IR_cam/output/ir_capture_<timestamp>/)")
    # Video is on by default; per-frame PNGs are opt-in to save disk space.
    p.add_argument("--no-video", dest="save_video", action="store_false",
                   help="Skip mp4 recording (video is saved by default)")
    p.add_argument("--save-images", dest="save_image", action="store_true",
                   help="Also save a timestamped PNG for every frame")
    p.add_argument("--no-latest", dest="save_latest", action="store_false",
                   help="Do not rewrite latest.png (disables file-based live-view)")
    p.add_argument("--no-view", dest="show_view", action="store_false",
                   help="Do not open the matplotlib live-view window")
    p.add_argument("--no-vflip", dest="vflip", action="store_false",
                   help="Show frames in raw sensor orientation (default: vertically flipped)")
    p.add_argument("--save_raw", action="store_true",
                   help="Also save raw_frames.npy with one (timestamp, frame) entry per frame")
    p.add_argument("--video-fps", type=float, default=None,
                   help="Video FPS (default: rate/2 for chess mode)")
    p.add_argument("--video-size", default="320x240",
                   help="Video frame size WxH (default: 320x240)")
    p.add_argument("--vmin", type=float, default=None,
                   help="Fixed colour-scale lower bound (°C)")
    p.add_argument("--vmax", type=float, default=None,
                   help="Fixed colour-scale upper bound (°C)")
    p.set_defaults(save_video=True, save_image=False, save_latest=True,
                   show_view=True, vflip=True)
    return p.parse_args()


def _parse_size(spec: str) -> tuple[int, int]:
    try:
        w_str, h_str = spec.lower().split("x", 1)
        w, h = int(w_str), int(h_str)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"invalid --video-size {spec!r}") from e
    assert w > 0 and h > 0, f"--video-size must be positive, got {spec}"
    return w, h


def _init_live_viewer(stop_event: threading.Event,
                      vmin: object, vmax: object):
    """Open a matplotlib window showing the latest thermal frame.

    Returns (fig, im, title, rect, marker) for overlays, or
    None if matplotlib couldn't be initialised (headless host, no display).
    Closing the window sets stop_event so shutdown mirrors Ctrl+C.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except Exception as e:
        print(f"[test] matplotlib unavailable, skipping live-view window: {e}")
        return None

    try:
        plt.ion()
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        try:
            fig.canvas.manager.set_window_title("IR Camera Live View")
        except Exception:
            pass
        placeholder = np.zeros((FRAME_H, FRAME_W), dtype=np.float32)
        init_vmin = float(vmin) if vmin is not None else 20.0
        init_vmax = float(vmax) if vmax is not None else 30.0
        if init_vmax - init_vmin < 0.5:
            init_vmax = init_vmin + 0.5
        im = ax.imshow(placeholder, cmap="inferno",
                       vmin=init_vmin, vmax=init_vmax,
                       interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, label="°C")

        # Add crop region rectangle (target area)
        rect = mpatches.Rectangle(
            (CROP_LEFT - 0.5, CROP_TOP - 0.5),
            CROP_RIGHT - CROP_LEFT,
            CROP_BOTTOM - CROP_TOP,
            linewidth=2,
            edgecolor='white',
            facecolor='none'
        )
        ax.add_patch(rect)

        # Add marker for hottest pixel (will be updated per frame)
        marker, = ax.plot([], [], 'r+', markersize=12, markeredgewidth=2)

        title = ax.set_title("waiting for first frame…")
        fig.canvas.mpl_connect("close_event", lambda _e: stop_event.set())
        fig.canvas.draw_idle()
        plt.pause(0.01)
    except Exception as e:
        print(f"[test] failed to open live-view window: {e}")
        return None

    return fig, im, title, rect, marker


def _update_live_viewer(viewer, frame, frame_num: int,
                        fixed_vmin: object, fixed_vmax: object,
                        vflip: bool = False) -> None:
    """Push a new frame into the viewer created by _init_live_viewer."""
    if viewer is None:
        return
    _fig, im, title, rect, marker = viewer
    pixels = np.flipud(frame.pixels) if vflip else frame.pixels
    im.set_data(pixels)
    if fixed_vmin is None or fixed_vmax is None:
        finite = pixels[np.isfinite(pixels)]
        if finite.size:
            lo = float(finite.min()) if fixed_vmin is None else float(fixed_vmin)
            hi = float(finite.max()) if fixed_vmax is None else float(fixed_vmax)
            if hi - lo < 0.5:
                hi = lo + 0.5
            im.set_clim(lo, hi)
    meta = frame.metadata

    def _fmt(value: object) -> str:
        return f"{float(value):.2f}" if isinstance(value, numbers.Real) else "nan"

    # Show cropped region stats as primary; full frame as secondary
    crop_min = meta.get('crop_min', float('nan'))
    crop_max = meta.get('crop_max', float('nan'))
    crop_avg = meta.get('crop_avg', float('nan'))

    title.set_text(
        f"frame {frame_num}  "
        f"[Target] min={_fmt(crop_min)}°C max={_fmt(crop_max)}°C avg={_fmt(crop_avg)}°C"
    )

    # Update hottest pixel marker
    max_temp_pixel = meta.get('max_temp_pixel')
    if max_temp_pixel is not None:
        row, col = max_temp_pixel
        # Adjust for vflip if needed
        if vflip:
            row = FRAME_H - 1 - row
        marker.set_data([col], [row])


def _install_sigint(stop_event: threading.Event) -> None:
    """Swap SIGINT to set an Event so the main loop can exit gracefully.

    Only the first Ctrl+C is absorbed; a second Ctrl+C restores the default
    handler so an unresponsive shutdown can still be forced.
    """
    def _handler(signum, frame):
        if not stop_event.is_set():
            print("\nInterrupt received; shutting down (press Ctrl+C again to force).")
            stop_event.set()
            # Next Ctrl+C goes straight to the default KeyboardInterrupt path.
            signal.signal(signal.SIGINT, signal.SIG_DFL)
    signal.signal(signal.SIGINT, _handler)


def main() -> int:
    args = _parse_args()

    assert args.rate in VALID_RATES, (
        f"--rate {args.rate} invalid; must be one of {VALID_RATES}")
    if args.duration is not None:
        assert args.duration > 0, "--duration must be positive"

    # Timed run if --duration given, otherwise stream indefinitely until Ctrl+C.
    bounded = args.duration is not None

    video_size = _parse_size(args.video_size) if args.save_video else (320, 240)
    # Chess-mode: effective full-frame rate is sensor_rate / 2.
    video_fps = args.video_fps if args.video_fps is not None else max(0.5, args.rate / 2.0)

    save_dir = args.save_dir or os.path.join(
        DEFAULT_OUTPUT_ROOT,
        f"ir_capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    cam = IRCamera(port=args.port, baud=args.baud)
    print(f"Port: {cam.port or '(not detected)'}")
    assert cam.port is not None, (
        "No IR camera port detected. Pass --port /dev/tty.usbmodemXXXX.")

    stop_event = threading.Event()
    _install_sigint(stop_event)

    cam.open()
    try:
        info = cam.info()
        print(f"Sensor info: {info}")
        assert cam.ping(), "Device did not respond to PING"

        cam.set_rate(args.rate)
        print(f"Rate set to {args.rate} Hz")

        cam.start_streaming(
            save_dir=save_dir,
            save_image=args.save_image,
            save_latest=args.save_latest,
            save_raw=args.save_raw,
            save_video=args.save_video,
            video_fps=video_fps,
            video_size=video_size,
            vmin=args.vmin,
            vmax=args.vmax,
        )

        outputs = []
        if args.save_latest:
            outputs.append("latest.png")
        if args.save_image:
            outputs.append("per-frame PNGs")
        if args.save_raw:
            outputs.append("raw_frames.npy")
        if args.save_video:
            outputs.append(f"stream.mp4@{video_fps:.2f}fps")
        mode_desc = f"{args.duration:.1f}s" if bounded else "∞ (Ctrl+C to stop)"
        print(f"Streaming {mode_desc} -> {save_dir} "
              f"[{', '.join(outputs) or 'in-memory only'}]")
        if args.save_latest:
            print("  (open latest.png in a hot-reloading viewer to watch live)")

        viewer = _init_live_viewer(stop_event, args.vmin, args.vmax) if args.show_view else None
        if viewer is not None:
            print("  (live-view window open; close it or press Ctrl+C to stop)")

        t_start = time.time()
        last_frame_print = -1

        def _should_continue() -> bool:
            if stop_event.is_set():
                return False
            if bounded:
                return (time.time() - t_start) < args.duration
            return True

        try:
            while _should_continue():
                f = cam.get_latest_frame()
                n = cam.frame_count()
                if f is not None and n != last_frame_print:
                    meta = f.metadata

                    def _fmt_temp(value: object) -> str:
                        return f"{float(value):.2f}" if isinstance(value, numbers.Real) else "nan"

                    print(f"  t={time.time() - t_start:6.1f}s "
                          f"frames={n} "
                          f"min={_fmt_temp(meta.get('min'))}C "
                          f"max={_fmt_temp(meta.get('max'))}C "
                          f"avg={_fmt_temp(meta.get('avg'))}C")
                    _update_live_viewer(viewer, f, n, args.vmin, args.vmax,
                                        vflip=args.vflip)
                    last_frame_print = n

                if viewer is not None:
                    # plt.pause() both flushes GUI events and sleeps, keeping
                    # the window responsive without a separate event loop.
                    import matplotlib.pyplot as plt
                    try:
                        plt.pause(0.05)
                    except Exception:
                        # A dead window (user force-killed it) should look
                        # like a graceful close to the rest of the loop.
                        stop_event.set()
                        viewer = None
                else:
                    stop_event.wait(timeout=1.0)
        finally:
            if viewer is not None:
                try:
                    import matplotlib.pyplot as plt
                    plt.ioff()
                    plt.close(viewer[0])
                except Exception:
                    pass

        cam.stop_streaming()

        total = cam.frame_count()
        elapsed = time.time() - t_start
        seq_gaps = cam.seq_gap_count()
        print(f"Total frames received: {total} in {elapsed:.1f}s "
              f"(dropped per seq counter: {seq_gaps})")

        # Throughput sanity check only applies to timed runs.
        if bounded:
            expected_min = max(1, int(args.duration * args.rate * 0.4))
            if total < expected_min:
                print(f"WARNING: expected >= {expected_min} frames at {args.rate} Hz; "
                      f"got {total}. Check USB CDC throughput and I2C pull-ups.")
                return 1

        # Verify expected output files exist.
        required = ["metadata.jsonl"]
        if args.save_latest and total > 0:
            required.append("latest.png")
        if args.save_video and total > 0:
            required.append("stream.mp4")
        if args.save_raw and total > 0:
            required.append("raw_frames.npy")
        for name in required:
            path = os.path.join(save_dir, name)
            if not os.path.exists(path):
                print(f"WARNING: expected output file missing: {path}")
                return 1
        print(f"Output saved to {save_dir}")
        return 0

    finally:
        try:
            cam.close()
        except Exception as e:
            print(f"[test] close error: {e}")


if __name__ == "__main__":
    sys.exit(main())
