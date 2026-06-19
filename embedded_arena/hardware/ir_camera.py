"""Host-side driver for the ESP32-S3 + MLX90640 thermal camera.

Mirrors the style of tools/device_monitor.py: the pipeline (or a test script)
opens the camera, issues text commands over UART, and receives binary frames
that are parsed into numpy arrays.

Streaming runs on a background thread so the main program can keep working
while frames are received. While streaming the driver can optionally save:
  - a latest.png iron-color image rewritten atomically in-place, so external
    viewers always see the most recent frame without torn writes,
  - timestamped .png copies,
  - one .npy array containing (timestamp, frame) per entry,
  - a JSONL metadata log (one frame per line).

Protocol — see IR_cam/IR_cam.ino for the device side. Streaming frames are
fixed 3088-byte binary records with CRC-16/CCITT-FALSE protection on both the
header and the payload. ASCII command/response lines (READY, OK, PONG, INFO,
ERR, OK RATE=) are '\n'-terminated and appear outside of frames; any bytes
that are neither a known ASCII line nor a valid CRC-checked frame are treated
as noise and discarded by resyncing on the 4-byte magic.

Defensive programming: every public method asserts its preconditions, frame
reads verify magic + both CRCs, and file writes use os.replace() so partial
writes can never be observed by a viewer.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

import numpy as np
import serial
import serial.tools.list_ports


# -----------------------------------------------------------------------------
# Protocol constants — must match the firmware in IR_cam/IR_cam.ino
# -----------------------------------------------------------------------------

DEFAULT_BAUD = 921600

# MLX90640 is fixed at 32x24 float32 pixels = 3072 bytes/frame.
FRAME_W = 32
FRAME_H = 24
FRAME_PIXELS = FRAME_W * FRAME_H
FRAME_BYTES = FRAME_PIXELS * 4

# Crop borders: remove 4 pixels from each edge for target area stats
CROP_BORDER = 4
CROP_LEFT = CROP_BORDER
CROP_RIGHT = FRAME_W - CROP_BORDER
CROP_TOP = CROP_BORDER
CROP_BOTTOM = FRAME_H - CROP_BORDER
CROP_W = CROP_RIGHT - CROP_LEFT
CROP_H = CROP_BOTTOM - CROP_TOP

# Binary-frame layout (kept in lockstep with IR_cam.ino).
MAGIC = b"\xAA\x55\xF0\x0D"
HEADER_SIZE = 14                         # magic(4) + seq(2) + ts(4) + len(2) + hcrc(2)
FRAME_TOTAL_BYTES = HEADER_SIZE + FRAME_BYTES + 2   # 3088

VALID_RATES: Tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)

READ_TIMEOUT_S = 0.6


# -----------------------------------------------------------------------------
# CRC-16/CCITT-FALSE  (poly 0x1021, init 0xFFFF, no refin/refout, xorout 0)
# -----------------------------------------------------------------------------

def _build_crc16_table() -> Tuple[int, ...]:
    table = []
    for b in range(256):
        crc = b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if (crc & 0x8000) else (crc << 1) & 0xFFFF
        table.append(crc)
    return tuple(table)


_CRC16_TABLE = _build_crc16_table()


def crc16_ccitt(data: bytes) -> int:
    """CRC-16/CCITT-FALSE of `data`. Byte-table implementation for speed."""
    crc = 0xFFFF
    for b in data:
        crc = ((crc << 8) & 0xFFFF) ^ _CRC16_TABLE[((crc >> 8) ^ b) & 0xFF]
    return crc


@dataclass
class IRFrame:
    """One parsed thermal frame plus its per-frame metadata."""

    metadata: dict              # {"ts": int, "seq": int, "w": 32, "h": 24, "min": ..., "max": ..., "avg": ...}
    pixels: np.ndarray          # shape (24, 32), dtype float32, units °C
    received_at: float = field(default_factory=time.time)


# -----------------------------------------------------------------------------
# Colormap utility — keeps the "latest.png" visually useful for realtime viewers
# -----------------------------------------------------------------------------

def iron_color_image(pixels: np.ndarray,
                     vmin: Optional[float] = None,
                     vmax: Optional[float] = None,
                     out_size: Tuple[int, int] = (320, 240),
                     cmap_name: str = "inferno"):
    """Render a 2D temperature array as an iron-colour RGB image.

    Args:
        pixels: 2D numpy array of temperatures (°C).
        vmin, vmax: Optional colour-scale limits. If None, uses per-frame min/max.
            Auto-widens the scale by 0.5 °C if the frame is too flat, to avoid
            divide-by-zero and blown-out contrast.
        out_size: Output image dimensions (w, h). Upscaled with NEAREST so pixels
            stay visually distinct.
        cmap_name: Matplotlib colormap name (default ``inferno``).

    Returns:
        A PIL.Image.Image in RGB mode.
    """
    assert pixels.ndim == 2, f"expected 2D array, got shape {pixels.shape}"

    # Imports are local so the driver is importable on machines without
    # matplotlib/PIL (e.g. for headless frame capture only).
    from PIL import Image
    import matplotlib
    import matplotlib.cm as mpl_cm

    finite = np.isfinite(pixels)
    if np.any(finite):
        finite_pixels = pixels[finite]
        lo = float(np.min(finite_pixels)) if vmin is None else vmin
        hi = float(np.max(finite_pixels)) if vmax is None else vmax
    else:
        lo = 0.0 if vmin is None else vmin
        hi = 1.0 if vmax is None else vmax
    if hi - lo < 0.5:
        hi = lo + 0.5

    safe_pixels = np.where(finite, pixels, lo)
    normalised = np.clip((safe_pixels - lo) / (hi - lo), 0.0, 1.0)
    try:
        cmap = matplotlib.colormaps[cmap_name]
    except (AttributeError, KeyError, ValueError):
        cmap = mpl_cm.get_cmap(cmap_name)
    rgba = (cmap(normalised) * 255).astype(np.uint8)

    img = Image.fromarray(rgba, mode="RGBA").convert("RGB")
    return img.resize(out_size, Image.NEAREST)


def iron_color_image_with_overlay(pixels: np.ndarray,
                                   max_temp_pixel: Optional[Tuple[int, int]] = None,
                                   vmin: Optional[float] = None,
                                   vmax: Optional[float] = None,
                                   out_size: Tuple[int, int] = (320, 240),
                                   cmap_name: str = "inferno"):
    """Render thermal frame with target crop bounding box and hottest pixel marker.

    Args:
        pixels: 2D numpy array of temperatures (°C).
        max_temp_pixel: (row, col) of the hottest pixel in the cropped region, or None.
        vmin, vmax: Optional colour-scale limits.
        out_size: Output image dimensions (w, h).
        cmap_name: Matplotlib colormap name (default ``inferno``).

    Returns:
        A PIL.Image.Image in RGB mode with overlays.
    """
    from PIL import Image, ImageDraw
    import matplotlib
    import matplotlib.cm as mpl_cm

    assert pixels.ndim == 2, f"expected 2D array, got shape {pixels.shape}"

    # Render base thermal image
    finite = np.isfinite(pixels)
    if np.any(finite):
        finite_pixels = pixels[finite]
        lo = float(np.min(finite_pixels)) if vmin is None else vmin
        hi = float(np.max(finite_pixels)) if vmax is None else vmax
    else:
        lo = 0.0 if vmin is None else vmin
        hi = 1.0 if vmax is None else vmax
    if hi - lo < 0.5:
        hi = lo + 0.5

    safe_pixels = np.where(finite, pixels, lo)
    normalised = np.clip((safe_pixels - lo) / (hi - lo), 0.0, 1.0)
    try:
        cmap = matplotlib.colormaps[cmap_name]
    except (AttributeError, KeyError, ValueError):
        cmap = mpl_cm.get_cmap(cmap_name)
    rgba = (cmap(normalised) * 255).astype(np.uint8)

    img = Image.fromarray(rgba, mode="RGBA").convert("RGB")
    img = img.resize(out_size, Image.NEAREST)

    # Draw overlays: target crop region and max temp pixel
    draw = ImageDraw.Draw(img)
    w_out, h_out = out_size

    # Scale factors from sensor to output image
    scale_w = w_out / FRAME_W
    scale_h = h_out / FRAME_H

    # Draw crop bounding box (white rectangle)
    crop_left_px = int(CROP_LEFT * scale_w)
    crop_top_px = int(CROP_TOP * scale_h)
    crop_right_px = int(CROP_RIGHT * scale_w)
    crop_bottom_px = int(CROP_BOTTOM * scale_h)
    draw.rectangle(
        [(crop_left_px, crop_top_px), (crop_right_px - 1, crop_bottom_px - 1)],
        outline=(255, 255, 255),
        width=2
    )

    # Draw marker for hottest pixel (red crosshair)
    if max_temp_pixel is not None:
        row, col = max_temp_pixel
        px_x = int(col * scale_w + scale_w / 2)
        px_y = int(row * scale_h + scale_h / 2)
        radius = 6
        # Crosshair: horizontal and vertical lines
        draw.line([(px_x - radius, px_y), (px_x + radius, px_y)], fill=(255, 0, 0), width=2)
        draw.line([(px_x, px_y - radius), (px_x, px_y + radius)], fill=(255, 0, 0), width=2)

    return img


# -----------------------------------------------------------------------------
# ffmpeg-pipe video writer — no extra Python dependency
# -----------------------------------------------------------------------------

class _FFmpegVideoWriter:
    """Stream raw RGB frames into ffmpeg over a pipe to produce an mp4.

    Using ffmpeg as a subprocess avoids a cv2/imageio dependency. Frames are
    expected as contiguous uint8 RGB arrays with the exact configured shape.
    Closing is idempotent and bounded so KeyboardInterrupt can still finalise
    the file.
    """

    def __init__(self, path: str, width: int, height: int, fps: float):
        assert shutil.which("ffmpeg") is not None, (
            "ffmpeg not found on PATH — install ffmpeg or disable --video"
        )
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel", "error",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}",
            "-r", f"{max(0.5, float(fps)):.3f}",
            "-i", "-",
            "-an",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "veryfast",
            "-movflags", "+faststart",
            path,
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        self._width = width
        self._height = height
        self._path = path
        self._closed = False
        self._frames_written = 0

    def write(self, rgb: np.ndarray) -> None:
        assert rgb.ndim == 3 and rgb.shape[2] == 3, f"expected HxWx3 RGB, got {rgb.shape}"
        assert rgb.shape[0] == self._height and rgb.shape[1] == self._width, (
            f"frame shape {rgb.shape[:2]} does not match writer {self._height}x{self._width}"
        )
        if self._closed or self._proc.stdin is None:
            return
        data = rgb.astype(np.uint8, copy=False).tobytes()
        try:
            self._proc.stdin.write(data)
            self._frames_written += 1
        except (BrokenPipeError, ValueError):
            # ffmpeg died; finalise so a partial file is still readable.
            self._closed = True

    def close(self, timeout: float = 5.0) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._proc.stdin is not None:
                self._proc.stdin.close()
        except Exception:
            pass
        try:
            self._proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            try:
                self._proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                pass

    @property
    def frames_written(self) -> int:
        return self._frames_written


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------

class IRCamera:
    """ESP32-S3 + MLX90640 IR camera driver (host side).

    Typical usage:

        cam = IRCamera()            # auto-detects port
        cam.open()
        cam.ping()
        cam.set_rate(4)
        cam.start_streaming(save_dir="./ir_out")
        time.sleep(10)
        cam.stop_streaming()
        cam.close()

    The background streaming thread is the only place that reads from the
    serial port once start_streaming() has been called; do NOT call
    _read_frame() directly while streaming is active.
    """

    def __init__(self, port: Optional[str] = None,
                 baud: int = DEFAULT_BAUD,
                 save_dir: Optional[str] = None):
        self.port = "/dev/tty.usbmodem1201"
        self.baud = baud
        self.save_dir = save_dir
        self.ser: Optional[serial.Serial] = None

        # Shared streaming state (all accessed under _lock from outside _stream_loop).
        self._lock = threading.Lock()
        self._latest_frame: Optional[IRFrame] = None
        self._frame_count = 0
        self._stream_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Persistent receive buffer for the magic-scanning framer. Bytes that
        # arrive mid-frame or between frames are accumulated here; the framer
        # consumes from the front when a complete, CRC-valid frame is decoded.
        self._rx_buf: bytearray = bytearray()

        # Dropped-frame accounting (observed by inspecting the seq field).
        self._last_seq: Optional[int] = None
        self._seq_gaps: int = 0

        # Optional video sink and per-frame callback, configured per session.
        self._video_writer: Optional[_FFmpegVideoWriter] = None
        self._video_path: Optional[str] = None
        self._video_fps: float = 2.0
        self._video_size: Tuple[int, int] = (320, 240)   # (w, h)
        self._on_frame: Optional[Callable[[IRFrame], None]] = None
        self._save_raw_enabled: bool = False
        self._raw_frame_records: list[tuple[int, np.ndarray]] = []

    # ------------------------------------------------------------------
    # Port discovery
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_port() -> Optional[str]:
        """Heuristic scan for an ESP32 USB-serial port.

        Skips PPK2 (identified by its D8E serial prefix) so the IR camera and
        the power profiler can coexist on the same host without a conflict.
        """
        ports = serial.tools.list_ports.comports()

        # Prefer ports whose description clearly identifies them as ESP32.
        for p in ports:
            desc = (p.description or "").lower()
            manu = (p.manufacturer or "").lower()
            if any(k in desc for k in ("esp32", "usb jtag", "cp210", "ch340", "ch9102")):
                return p.device
            if "espressif" in manu:
                return p.device

        # Fallback: first usbmodem/usbserial that isn't a PPK2.
        for p in ports:
            dev = p.device
            if ("usbmodem" in dev or "usbserial" in dev) and "D8E" not in dev:
                return dev

        return None

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def open(self, ready_timeout_s: float = 12.0, sensor_timeout_s: float = 15.0) -> None:
        """Open the serial port and sync with the device.

        Two-phase sync:
          1. PING loop — waits until commsTask is alive (fast; typically < 1 s
             because commsTask now starts before MLX init).
          2. Sensor-ready poll — issues INFO and checks `ready: true` in the
             JSON response, waiting up to sensor_timeout_s for the MLX90640
             to finish initialising on senseTask.

        If the firmware is old and never emits `ready` in INFO, phase 2 is
        skipped after one successful INFO parse and we proceed immediately.
        """
        assert self.port is not None, "No IR camera port found — specify port= explicitly"
        assert self.ser is None, "Camera is already open"

        self.ser = serial.Serial(self.port, self.baud, timeout=READ_TIMEOUT_S)
        # Give a just-reset bootloader time to jump into the sketch. Native
        # USB CDC doesn't always reset on open, but if it does this is enough
        # for the sketch's Serial.begin() to re-enumerate.
        time.sleep(0.5)

        # Drain any pending output (boot banners, stale replies).
        drain_deadline = time.time() + 0.5
        while time.time() < drain_deadline:
            if not self.ser.in_waiting:
                time.sleep(0.05)
                if not self.ser.in_waiting:
                    break
            _ = self.ser.readline()

        # If the last session ended without STOP (killed process, unplug, etc.)
        # the firmware may still be streaming binary frames — clear that before
        # line-based PING/INFO sync.
        try:
            for _ in range(3):
                self._send("STOP")
                time.sleep(0.08)
            self.ser.reset_input_buffer()
        except Exception:
            pass

        # ---- Phase 1: PING loop ----
        deadline = time.time() + ready_timeout_s
        last_err: Optional[str] = None
        while time.time() < deadline:
            try:
                self.ser.reset_input_buffer()
            except Exception:
                pass
            try:
                self._send("PING")
            except Exception as e:
                last_err = f"send failed: {e}"
                time.sleep(0.3)
                continue
            try:
                self._await_line(b"PONG", timeout_s=1.0, exact=True)
                break
            except TimeoutError:
                last_err = "no PONG"
                time.sleep(0.3)
        else:
            try:
                self.ser.close()
            except Exception:
                pass
            self.ser = None
            raise TimeoutError(
                f"unable to synchronize with camera on {self.port} within "
                f"{ready_timeout_s:.1f}s ({last_err or 'unknown'})"
            )

        # ---- Phase 2: wait for sensor ready ----
        # Poll INFO until ready:true appears or sensor_timeout_s elapses.
        sense_deadline = time.time() + sensor_timeout_s
        sensor_ready_known = False
        while time.time() < sense_deadline:
            try:
                self.ser.reset_input_buffer()
                time.sleep(0.05)
                self._send("INFO")
                info_deadline = time.time() + 1.5
                while time.time() < info_deadline:
                    line = self.ser.readline()
                    if not line:
                        continue
                    probe = line.strip()
                    if not probe.startswith(b"INFO "):
                        continue
                    brace = probe.find(b"{")
                    end = probe.rfind(b"}")
                    if brace < 0 or end <= brace:
                        continue
                    try:
                        import json as _json
                        parsed = _json.loads(probe[brace:end + 1])
                    except Exception:
                        continue
                    if not isinstance(parsed, dict):
                        continue
                    # Old firmware without the `ready` field: proceed immediately.
                    if "ready" not in parsed:
                        return
                    if parsed["ready"]:
                        return
                    # ready=false: sensor still initialising — keep polling.
                    sensor_ready_known = True
                    break
            except Exception:
                pass
            time.sleep(0.5)

        # Timed out waiting for sensor ready.
        hint = "sensor initialisation timed out" if sensor_ready_known else "INFO not received"
        try:
            self.ser.close()
        except Exception:
            pass
        self.ser = None
        raise TimeoutError(
            f"IR camera on {self.port} did not become ready within "
            f"{sensor_timeout_s:.1f}s ({hint})"
        )

    def close(self) -> None:
        """Stop streaming (if active) and close the serial port."""
        if self._stream_thread is not None:
            self.stop_streaming()
        if self.ser is not None:
            try:
                self._send("STOP")
            except Exception:
                pass
            self.ser.close()
            self.ser = None

    # ------------------------------------------------------------------
    # Low-level I/O
    # ------------------------------------------------------------------

    def _send(self, cmd: str) -> None:
        """Send a newline-terminated ASCII command."""
        assert self.ser is not None, "Camera not opened"
        self.ser.write((cmd + "\n").encode("ascii"))
        self.ser.flush()

    def _await_line(self, needle: bytes, timeout_s: float = 2.0,
                    exact: bool = False, startswith: bool = False) -> bytes:
        """Read lines until one matches `needle`, or TimeoutError."""
        assert self.ser is not None, "Camera not opened"
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            line = self.ser.readline()
            if not line:
                continue
            probe = line.strip()
            if exact:
                if probe == needle:
                    return line
                continue
            if startswith:
                if probe.startswith(needle):
                    return line
                continue
            if needle in line:
                return line
        raise TimeoutError(f"expected {needle!r} within {timeout_s}s")

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    def ping(self) -> bool:
        """Send PING, return True if PONG is received within 2 s."""
        assert self.ser is not None, "Camera not opened"
        assert self._stream_thread is None, "cannot ping while streaming"
        self._send("PING")
        try:
            self._await_line(b"PONG", timeout_s=2.0)
            return True
        except TimeoutError:
            return False

    def info(self, timeout_s: float = 3.0, retries: int = 3) -> dict:
        """Query static sensor info (size, pixel count, format).

        With the CRC-protected binary framing in v2, ASCII and binary bytes
        never interleave, so this parser no longer needs the salvage path
        that v1 required. First-connection calls can still race with
        residual boot output, so retries default to 3.
        """
        assert self.ser is not None, "Camera not opened"
        assert self._stream_thread is None, "cannot query INFO while streaming"

        retries = max(1, int(retries))
        for attempt in range(retries):
            # Drop any leftover boot chatter / responses from a prior command.
            self.ser.reset_input_buffer()
            time.sleep(0.05)
            self._send("INFO")

            deadline = time.time() + timeout_s
            while time.time() < deadline:
                line = self.ser.readline()
                if not line:
                    continue
                probe = line.strip()
                if not probe.startswith(b"INFO "):
                    continue
                brace = probe.find(b"{")
                end = probe.rfind(b"}")
                if brace < 0 or end <= brace:
                    continue
                try:
                    parsed = json.loads(probe[brace:end + 1])
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    return parsed

            if attempt < retries - 1:
                time.sleep(0.1)

        raise TimeoutError(f"expected INFO response within {timeout_s:.1f}s (retries={retries})")

    def set_rate(self, hz: float) -> None:
        """Set the MLX90640 refresh rate. Must be one of VALID_RATES."""
        assert self.ser is not None, "Camera not opened"
        assert hz in VALID_RATES, f"invalid rate {hz}; valid: {VALID_RATES}"
        assert self._stream_thread is None, "stop streaming before changing rate"
        for _ in range(4):
            self._send(f"RATE {hz}")
            try:
                self._await_line(b"OK RATE=", timeout_s=2.0, startswith=True)
                return
            except TimeoutError:
                # Re-check liveness and retry; command replies can be lost on noisy links.
                if not self.ping():
                    raise TimeoutError(f"camera unresponsive while setting RATE {hz} at baud {self.baud}")
                time.sleep(0.1)
        raise TimeoutError(f"expected b'OK RATE=' while setting RATE {hz} at baud {self.baud}")

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def start_streaming(self,
                        save_dir: Optional[str] = None,
                        save_image: bool = False,
                        save_latest: bool = True,
                        save_raw: bool = False,
                        save_video: bool = False,
                        video_fps: float = 2.0,
                        video_size: Tuple[int, int] = (320, 240),
                        video_filename: str = "stream.mp4",
                        vmin: Optional[float] = None,
                        vmax: Optional[float] = None,
                        on_frame: Optional[Callable[[IRFrame], None]] = None) -> None:
        """Enable streaming and spawn the background reader thread.

        Args:
            save_dir: Directory for latest.png, timestamped .png, raw_frames.npy, metadata.jsonl,
                stream.mp4. Created if missing. None = receive frames into memory only.
            save_image: Save a timestamped iron-colour PNG for every frame. Usually
                left False when save_video is True (latest.png is sufficient for live
                preview).
            save_latest: Atomically rewrite latest.png on every frame so an external
                image viewer sees a live feed. Enabled by default.
            save_raw: Also save one raw_frames.npy array with per-frame
                timestamp+pixels entries.
            save_video: Also append each iron-colour frame to an mp4 via ffmpeg.
                Requires ffmpeg on PATH and `save_dir`.
            video_fps: Target frame rate for the output video. For MLX90640 in
                chess mode this is typically sensor_rate / 2.
            video_size: (width, height) of the rendered video frames.
            video_filename: Filename written under `save_dir` (default "stream.mp4").
            vmin, vmax: Fixed colour-scale limits; None = per-frame auto-scale.
            on_frame: Optional callback invoked in the stream thread for every
                successfully decoded frame. Exceptions are caught and logged so
                a buggy callback cannot kill the streaming loop.
        """
        assert self.ser is not None, "Camera not opened"
        assert self._stream_thread is None, "already streaming"
        if save_video:
            assert save_dir is not None or self.save_dir is not None, (
                "save_video requires save_dir"
            )

        if save_dir is not None:
            self.save_dir = save_dir
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)

        # Reset framer state before a new session so stale bytes and stale
        # sequence counters from a previous run don't pollute this one.
        self._rx_buf.clear()
        self._last_seq = None
        self._seq_gaps = 0
        self.ser.reset_input_buffer()

        # Record video/callback config; the actual ffmpeg process is opened
        # lazily on the first frame so a stream that never produces anything
        # doesn't leave a zero-byte mp4 behind.
        self._video_path = (
            os.path.join(self.save_dir, video_filename)
            if (save_video and self.save_dir) else None
        )
        self._video_fps = float(video_fps)
        self._video_size = (int(video_size[0]), int(video_size[1]))
        self._on_frame = on_frame
        self._save_raw_enabled = bool(save_raw and self.save_dir)
        self._raw_frame_records = []

        self._send("START")
        try:
            self._await_line(b"OK", timeout_s=2.0)
        except TimeoutError:
            # Some hosts/links can lose the small ASCII reply while the
            # device immediately begins streaming binary frames. As a
            # fallback, probe the incoming bytes briefly for the magic
            # header and proceed if we see it (tolerant behaviour).
            fallback_deadline = time.time() + 2.0
            found_magic = False
            while time.time() < fallback_deadline:
                try:
                    pending = self.ser.in_waiting
                except Exception:
                    pending = 0
                if pending:
                    data = self.ser.read(pending)
                    if MAGIC in data:
                        # Seed the receiver buffer with what we've read so the
                        # streaming thread can continue decoding frames.
                        self._rx_buf.extend(data)
                        found_magic = True
                        break
                time.sleep(0.05)
            if not found_magic:
                raise

        self._stop_event.clear()
        self._stream_thread = threading.Thread(
            target=self._stream_loop,
            args=(save_image, save_latest, save_raw, save_video, vmin, vmax),
            name="IRCamera-stream",
            daemon=True,
        )
        self._stream_thread.start()

    def stop_streaming(self, join_timeout_s: float = 3.0) -> None:
        """Signal the reader thread to stop, wait for it, finalise video."""
        if self._stream_thread is None:
            # Still release the video writer in case of a broken start.
            self._close_video_writer()
            return

        self._stop_event.set()
        # Ask the device to stop so we don't keep draining frames during shutdown.
        try:
            self._send("STOP")
        except Exception:
            pass

        self._stream_thread.join(timeout=join_timeout_s)
        self._stream_thread = None
        self._flush_raw_records()
        # Finalise after the thread is gone so no pending write can race it.
        self._close_video_writer()
        self._on_frame = None

    def _close_video_writer(self) -> None:
        if self._video_writer is not None:
            try:
                self._video_writer.close()
            except Exception as e:
                print(f"[IRCamera] video close error: {e}")
            self._video_writer = None

    def _flush_raw_records(self) -> None:
        """Persist buffered raw records into one .npy array."""
        if not self._save_raw_enabled or self.save_dir is None:
            return
        if not self._raw_frame_records:
            return

        records = np.empty(
            len(self._raw_frame_records),
            dtype=[("ts", np.uint64), ("frame", np.float32, (FRAME_H, FRAME_W))],
        )
        for idx, (ts, pixels) in enumerate(self._raw_frame_records):
            records["ts"][idx] = np.uint64(ts)
            records["frame"][idx] = pixels.astype(np.float32, copy=False)

        path = os.path.join(self.save_dir, "raw_frames.npy")
        np.save(path, records)
        self._raw_frame_records = []

    def get_latest_frame(self) -> Optional[IRFrame]:
        """Return the most recently received frame, or None if no frame yet."""
        with self._lock:
            return self._latest_frame

    def frame_count(self) -> int:
        """Total number of frames received since open()."""
        with self._lock:
            return self._frame_count

    def seq_gap_count(self) -> int:
        """Number of missing frames detected from the device seq counter.

        Incremented whenever two consecutive decoded frames have a seq delta
        greater than 1 (with 16-bit wrap accounted for). Useful as a
        dropped-frame indicator orthogonal to CRC failures.
        """
        with self._lock:
            return self._seq_gaps

    # ------------------------------------------------------------------
    # Background reader
    # ------------------------------------------------------------------

    def _stream_loop(self, save_image: bool, save_latest: bool, save_raw: bool,
                     save_video: bool, vmin: Optional[float],
                     vmax: Optional[float]) -> None:
        while not self._stop_event.is_set():
            try:
                frame = self._read_frame()
            except Exception as e:
                print(f"[IRCamera] stream read error: {e}")
                time.sleep(0.05)
                continue

            if frame is None:
                continue

            with self._lock:
                seq = frame.metadata.get("seq")
                if isinstance(seq, int) and self._last_seq is not None:
                    # 16-bit wrap-safe delta: expected delta is 1.
                    delta = (seq - self._last_seq) & 0xFFFF
                    if delta > 1:
                        self._seq_gaps += delta - 1
                if isinstance(seq, int):
                    self._last_seq = seq

                self._latest_frame = frame
                self._frame_count += 1

            if self.save_dir:
                try:
                    self._save_frame(frame, save_image=save_image,
                                     save_latest=save_latest,
                                     save_raw=save_raw, save_video=save_video,
                                     vmin=vmin, vmax=vmax)
                except Exception as e:
                    # Disk errors must not crash the streaming loop.
                    print(f"[IRCamera] save error: {e}")

            cb = self._on_frame
            if cb is not None:
                try:
                    cb(frame)
                except Exception as e:
                    # Isolate user callback failures from the streaming loop.
                    print(f"[IRCamera] on_frame callback error: {e}")

    def _read_frame(self) -> Optional[IRFrame]:
        """Magic-scanning, CRC-validated frame reader.

        Returns one IRFrame if a complete, CRC-valid 3088-byte frame is
        decoded this call, otherwise None. Bytes are accumulated in
        self._rx_buf across calls so a partial frame at the tail is preserved
        for the next invocation.
        """
        assert self.ser is not None, "Camera not opened"

        # Pull whatever is already sitting in the OS buffer first (cheap).
        # Only do a timed blocking read if we don't yet have enough bytes to
        # possibly contain a full frame — this way back-to-back buffered
        # frames are delivered without an extra timeout round-trip.
        pending = self.ser.in_waiting
        if pending:
            self._rx_buf.extend(self.ser.read(pending))
        if len(self._rx_buf) < FRAME_TOTAL_BYTES:
            # One small timed read keeps CPU low while streaming is idle.
            chunk = self.ser.read(1)
            if chunk:
                self._rx_buf.extend(chunk)

        while True:
            idx = self._rx_buf.find(MAGIC)
            if idx < 0:
                # No magic yet. Anything we have so far is either text
                # (boot/ERR/…) or corruption — keep just the last 3 bytes so
                # a magic split across reads can still be found.
                if len(self._rx_buf) > 3:
                    # Surface any clean ASCII preamble for debugging before
                    # discarding it; noisy binary runs are dropped silently.
                    preamble = bytes(self._rx_buf[:-3])
                    self._log_preamble(preamble)
                    del self._rx_buf[:-3]
                return None

            if idx > 0:
                preamble = bytes(self._rx_buf[:idx])
                self._log_preamble(preamble)
                del self._rx_buf[:idx]

            if len(self._rx_buf) < FRAME_TOTAL_BYTES:
                # Wait for the rest of the frame to arrive on the next call.
                return None

            # Header CRC check. A bad header CRC almost always means the
            # magic pattern showed up inside random data (1-in-2^32 odds);
            # advance past this magic and keep scanning.
            header = bytes(self._rx_buf[:HEADER_SIZE])
            h_crc_expected = int.from_bytes(header[12:14], "little")
            if crc16_ccitt(header[:12]) != h_crc_expected:
                # Skip the 4-byte magic and try to find another.
                del self._rx_buf[:4]
                continue

            seq = int.from_bytes(header[4:6], "little")
            ts_ms = int.from_bytes(header[6:10], "little")
            payload_len = int.from_bytes(header[10:12], "little")
            if payload_len != FRAME_BYTES:
                # Protocol mismatch. Can't trust anything after this header.
                print(f"[IRCamera] unexpected payload_len={payload_len}; dropping frame")
                del self._rx_buf[:4]
                continue

            payload = bytes(self._rx_buf[HEADER_SIZE:HEADER_SIZE + FRAME_BYTES])
            p_crc_expected = int.from_bytes(
                self._rx_buf[HEADER_SIZE + FRAME_BYTES:HEADER_SIZE + FRAME_BYTES + 2],
                "little",
            )
            if crc16_ccitt(payload) != p_crc_expected:
                print(f"[IRCamera] payload CRC mismatch (seq={seq}); dropping frame")
                del self._rx_buf[:4]
                continue

            # Consume the complete frame.
            del self._rx_buf[:FRAME_TOTAL_BYTES]

            pixels = np.frombuffer(payload, dtype="<f4").reshape(FRAME_H, FRAME_W).copy()

            # Compute stats host-side; firmware no longer ships per-frame stats.
            finite = np.isfinite(pixels)
            if np.any(finite):
                fp = pixels[finite]
                stats = {
                    "min": float(np.min(fp)),
                    "max": float(np.max(fp)),
                    "avg": float(np.mean(fp)),
                }
            else:
                stats = {"min": float("nan"), "max": float("nan"), "avg": float("nan")}

            # Compute stats for cropped region (target area)
            cropped = pixels[CROP_TOP:CROP_BOTTOM, CROP_LEFT:CROP_RIGHT]
            finite_crop = np.isfinite(cropped)
            if np.any(finite_crop):
                fc = cropped[finite_crop]
                crop_stats = {
                    "crop_min": float(np.min(fc)),
                    "crop_max": float(np.max(fc)),
                    "crop_avg": float(np.mean(fc)),
                }
                # Find the pixel with max temperature in cropped region
                max_idx_flat = np.argmax(cropped)
                max_row_rel, max_col_rel = np.unravel_index(max_idx_flat, cropped.shape)
                max_row = CROP_TOP + max_row_rel
                max_col = CROP_LEFT + max_col_rel
                crop_stats["max_temp_pixel"] = (int(max_row), int(max_col))
            else:
                crop_stats = {
                    "crop_min": float("nan"),
                    "crop_max": float("nan"),
                    "crop_avg": float("nan"),
                    "max_temp_pixel": None,
                }

            meta = {
                "ts": ts_ms,
                "seq": seq,
                "w": FRAME_W,
                "h": FRAME_H,
                "bytes": FRAME_BYTES,
                **stats,
                **crop_stats,
            }
            return IRFrame(metadata=meta, pixels=pixels)

    @staticmethod
    def _log_preamble(preamble: bytes) -> None:
        """Surface ASCII chatter (e.g. ERR lines) that appears between frames.

        Binary junk is dropped silently since it's almost always the tail of a
        corrupted frame or noise that's already been accounted for elsewhere.
        """
        if not preamble:
            return
        # Heuristic: treat as ASCII iff >=80% of the bytes are printable/\n/\r.
        printable = sum(1 for b in preamble if 32 <= b < 127 or b in (9, 10, 13))
        if printable * 5 >= len(preamble) * 4:
            text = preamble.decode("ascii", errors="replace").strip()
            if text:
                print(f"[IRCamera] device says: {text}")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_frame(self, frame: IRFrame, save_image: bool, save_latest: bool,
                    save_raw: bool, save_video: bool,
                    vmin: Optional[float], vmax: Optional[float]) -> None:
        assert self.save_dir is not None

        # Metadata log (one JSON per line; easy to tail -f).
        meta_record = {**frame.metadata, "received_at": frame.received_at}
        with open(os.path.join(self.save_dir, "metadata.jsonl"), "a") as f:
            f.write(json.dumps(meta_record) + "\n")

        ts = int(frame.metadata.get("ts", int(frame.received_at * 1000)))

        if save_raw:
            # Keep one entry per frame: (timestamp, raw pixels), then flush once.
            self._raw_frame_records.append((ts, frame.pixels.copy()))

        # Render once if any downstream consumer needs a coloured image. The
        # video size takes precedence so the frame written to both latest.png
        # and the mp4 have the same resolution.
        rendered: Optional[np.ndarray] = None
        if save_image or save_latest or save_video:
            out_size = self._video_size if save_video else (320, 240)
            max_temp_pixel = frame.metadata.get("max_temp_pixel")
            img = iron_color_image_with_overlay(
                frame.pixels,
                max_temp_pixel=max_temp_pixel,
                vmin=vmin,
                vmax=vmax,
                out_size=out_size
            )

            if save_image:
                # Timestamped per-frame copy (only when explicitly requested).
                stamped = os.path.join(self.save_dir, f"ir_{ts:012d}.png")
                img.save(stamped)

            if save_latest:
                # Atomic rewrite so a viewer never sees a partial PNG.
                latest = os.path.join(self.save_dir, "latest.png")
                tmp = latest + ".tmp"
                img.save(tmp, format="PNG")
                os.replace(tmp, latest)

            if save_video:
                rendered = np.asarray(img)

        if save_video and rendered is not None and self._video_path is not None:
            # Lazy-open the writer on the first frame so we know the exact
            # output dimensions from the rendered array and don't create a
            # 0-byte file if streaming never produces anything.
            if self._video_writer is None:
                h, w = rendered.shape[:2]
                try:
                    self._video_writer = _FFmpegVideoWriter(
                        self._video_path, w, h, self._video_fps)
                except Exception as e:
                    print(f"[IRCamera] video open error: {e}")
                    self._video_path = None
                    return
            try:
                self._video_writer.write(rendered)
            except Exception as e:
                print(f"[IRCamera] video write error: {e}")


if __name__ == "__main__":
    # Minimal smoke test. For a proper scripted test see IR_cam/test_ir_camera.py.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=None)
    parser.add_argument("--rate", type=float, default=64.0)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--save_dir", default="./ir_out")
    args = parser.parse_args()

    cam = IRCamera(port=args.port)
    cam.open()
    print("INFO:", cam.info())
    print("PING ok:", cam.ping())
    cam.set_rate(args.rate)
    cam.start_streaming(save_dir=args.save_dir)

    end = time.time() + args.duration
    while time.time() < end:
        time.sleep(0.5)
        f = cam.get_latest_frame()
        if f:
            print(f"frame#{cam.frame_count()} seq={f.metadata.get('seq')} "
                  f"ts={f.metadata.get('ts')} "
                  f"min={f.metadata.get('min')} max={f.metadata.get('max')}")

    cam.stop_streaming()
    cam.close()
    print(f"Saved {cam.frame_count()} frames to {args.save_dir} "
          f"(seq gaps: {cam.seq_gap_count()})")
