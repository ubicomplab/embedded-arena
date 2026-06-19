"""PPK2-only helper for source-meter power capture."""

from __future__ import annotations

import time
import traceback

import numpy as np
import serial
import serial.tools.list_ports

try:
    from ppk2_api.ppk2_api import PPK2_MP, PPK2_API
except ImportError:
    print("Warning: ppk2_api not found. ppk2Monitor will use simulation mode.")
    PPK2_MP = None
    PPK2_API = None


PPK2_HARDWARE_SAMPLE_RATE_HZ: float = 100_000.0


class ppk2Monitor:
    """PPK2-only power capture helper.

    Hardware streams ~100 kHz. ``sample_rate_hz`` sets the output ``times`` /
    ``currents`` series (non-overlapping block means when below hardware rate).
    """

    SAMPLE_RATE_HZ: float = PPK2_HARDWARE_SAMPLE_RATE_HZ

    def __init__(
        self,
        ppk2_port: str | None = None,
        sample_rate_hz: float = PPK2_HARDWARE_SAMPLE_RATE_HZ,
    ):
        self.ppk2 = None
        self.ppk2_port = ppk2_port
        self.ppk2_candidates: list[str] = []
        if sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        self.sample_rate_hz = float(min(sample_rate_hz, PPK2_HARDWARE_SAMPLE_RATE_HZ))
        self.output_sample_rate_hz: float = PPK2_HARDWARE_SAMPLE_RATE_HZ
        self._detect_port()

    def _hardware_to_output_series(
        self, currents_hw: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Map hardware-rate samples to output times/currents at ``self.sample_rate_hz``."""
        arr = np.asarray(currents_hw, dtype=np.float64)
        hw = PPK2_HARDWARE_SAMPLE_RATE_HZ
        req = self.sample_rate_hz
        if len(arr) == 0:
            self.output_sample_rate_hz = hw if req >= hw - 1e-9 else req
            return np.array([]), np.array([]), self.output_sample_rate_hz
        if req >= hw - 1e-9:
            times = np.arange(len(arr), dtype=np.float64) / hw
            self.output_sample_rate_hz = hw
            return times, arr, hw
        n = max(1, int(round(hw / req)))
        eff = hw / float(n)
        m = len(arr) // n
        if m == 0:
            times = np.arange(len(arr), dtype=np.float64) / hw
            self.output_sample_rate_hz = hw
            return times, arr, hw
        trimmed = arr[: m * n].reshape(m, n).mean(axis=1)
        times = np.arange(m, dtype=np.float64) / eff
        self.output_sample_rate_hz = eff
        return times, trimmed, eff

    def _detect_port(self) -> None:
        if self.ppk2_port:
            self.ppk2_candidates = [self.ppk2_port]
            return

        print("Scanning for connected devices...")
        ports = serial.tools.list_ports.comports()
        candidates = [p.device for p in ports]
        print(f"Available ports: {candidates}")

        # PPK2 on macOS exposes two CDC ACM interfaces (suffix A2 vs A4).
        # One is the ppk2_api data/control interface (LED responds to mode
        # commands); the other is an MCU UART passthrough where control
        # bytes are silently dropped. We can't tell them apart from the
        # device name alone, so collect both and let _init_ppk2 probe in
        # order until one responds to get_modifiers().
        ppk2_candidates = []
        for dev in candidates:
            if "usbmodem" in dev and "D8E" in dev:
                ppk2_candidates.append(dev)

        # Prefer /dev/cu.* over /dev/tty.* on macOS (same interface, /dev/cu
        # is the call-out variant used for outgoing connections).
        cu_candidates = [d for d in ppk2_candidates if d.startswith("/dev/cu.")]
        if cu_candidates:
            ppk2_candidates = cu_candidates

        # Try lower suffix first, then higher; _init_ppk2 walks the list
        # and validates each by reading modifiers from the device.
        self.ppk2_candidates = sorted(ppk2_candidates)

        if not self.ppk2_candidates and PPK2_API:
            try:
                devices = PPK2_API.list_devices() or []
                cu_devices = [d for d in devices if d.startswith("/dev/cu.")]
                self.ppk2_candidates = sorted(cu_devices or devices)
                if self.ppk2_candidates:
                    print(f"PPK2_API.list_devices(): {self.ppk2_candidates}")
            except Exception:
                pass

        if self.ppk2_candidates:
            self.ppk2_port = self.ppk2_candidates[0]
            print(f"PPK2 candidates (will probe in order): {self.ppk2_candidates}")
        else:
            print("Warning: PPK2 port not detected (simulation mode will be used).")

    def _init_ppk2(self, target_voltage_v: float = 3.3) -> bool:
        if self.ppk2:
            return True
        if not PPK2_MP or not self.ppk2_candidates:
            print("PPK2 not available.")
            return False
        if target_voltage_v < 1.0 or target_voltage_v > 5.0:
            print(f"Error: target_voltage_v must be between 1.0V and 5.0V, got {target_voltage_v}V")
            return False

        last_error: Exception | None = None
        for port in self.ppk2_candidates:
            ppk2 = None
            try:
                print(f"Probing PPK2 on {port}...")
                ppk2 = PPK2_MP(port=port)
                # The data/control interface answers get_modifiers() with a
                # parseable response; the UART passthrough does not.
                ppk2.get_modifiers()
                ppk2.use_source_meter()
                ppk2.set_source_voltage(int(target_voltage_v * 1000))
                self.ppk2 = ppk2
                self.ppk2_port = port
                print(f"PPK2 initialized on {port} with {target_voltage_v}V output.")
                return True
            except Exception as exc:
                last_error = exc
                print(f"  candidate {port} failed: {exc}")
                # Explicitly clean up the partially-initialized object.
                # PPK2_MP.__del__ accesses _quit_evt unconditionally; if the
                # constructor raised before setting it, patch a dummy in so
                # __del__ doesn't print a spurious AttributeError traceback.
                try:
                    if ppk2 is not None:
                        if not hasattr(ppk2, "_quit_evt"):
                            import threading
                            ppk2._quit_evt = threading.Event()
                        del ppk2
                except Exception:
                    pass
                continue

        print(
            f"Error initializing PPK2: tried {self.ppk2_candidates}, "
            f"last error: {last_error}"
        )
        self.ppk2 = None
        return False

    def voltage_on(self, target_voltage_v: float = 3.3) -> bool:
        if not self._init_ppk2(target_voltage_v=target_voltage_v):
            return False
        self.ppk2.toggle_DUT_power("ON")
        print("DUT power ON.")
        return True

    def voltage_off(self) -> None:
        if self.ppk2 is None:
            return
        try:
            self.ppk2.toggle_DUT_power("OFF")
            print("DUT power OFF.")
        except Exception as exc:
            print(f"Warning: voltage_off failed: {exc}")

    def close(self) -> None:
        if self.ppk2:
            try:
                self.ppk2.toggle_DUT_power("OFF")
                print("PPK2 DUT power OFF.")
            except Exception as exc:
                print(f"Warning: failed to toggle DUT power off: {exc}")
            finally:
                self.ppk2 = None

    def _start_ppk2_monitor(self, target_voltage_v: float = 3.3):
        if not self._init_ppk2(target_voltage_v=target_voltage_v):
            return None
        self.ppk2.toggle_DUT_power("ON")
        self.ppk2.start_measuring()
        print("PPK2 measurement started.")
        return self.ppk2

    def _stop_ppk2_monitor(self, ppk2) -> None:
        if ppk2:
            ppk2.stop_measuring()
            ppk2.toggle_DUT_power("OFF")
            print("PPK2 measurement stopped.")

    def _collect_ppk2_data_continuous(self, ppk2, duration_ms: int):
        if not ppk2:
            return None

        print(f"Collecting PPK2 data for {duration_ms} ms...")
        all_raw = bytearray()
        deadline = time.time() + duration_ms / 1000.0

        try:
            while time.time() < deadline:
                chunk = ppk2.get_data()
                if chunk:
                    all_raw.extend(chunk)
                time.sleep(0.001)
            return bytes(all_raw)
        except Exception as exc:
            print(f"Error collecting PPK2 data: {exc}")
            return None

    def _process_ppk2_data(self, ppk2, ppk_raw_data):
        report = ""
        currents = np.array([])
        times = np.array([])

        if not ppk2:
            return report, times, currents
        if ppk_raw_data is None:
            return report, times, currents

        try:
            if ppk_raw_data == b"":
                return report, times, currents

            samples, _ = ppk2.get_samples(ppk_raw_data)
            if samples is None or len(samples) == 0:
                return report, times, currents

            currents_hw = np.asarray(samples, dtype=np.float64)
            times, currents, _eff = self._hardware_to_output_series(currents_hw)
        except Exception:
            traceback.print_exc()

        return report, times, currents

    def run(self, duration_ms: int, mode: str = "ppk", target_voltage_v: float | None = None):
        if target_voltage_v is None:
            target_voltage_v = 3.3

        if not PPK2_MP or not self.ppk2_port:
            return self._run_simulation(duration_ms, mode)

        ppk2 = None
        try:
            if mode in ("all", "ppk"):
                ppk2 = self._start_ppk2_monitor(target_voltage_v=target_voltage_v)
                ppk_raw_data = self._collect_ppk2_data_continuous(ppk2, duration_ms)
            else:
                time.sleep(duration_ms / 1000.0)
                ppk_raw_data = None
        finally:
            if ppk2:
                self._stop_ppk2_monitor(ppk2)

        _, times, currents = self._process_ppk2_data(ppk2, ppk_raw_data)
        sr_out = self.output_sample_rate_hz
        data = {"times": times, "currents": currents}
        if len(currents) > 0:
            step = int(max(1, round(sr_out / 5.0)))
            data["times_5hz"] = times[::step]
            data["currents_5hz"] = currents[::step]
            data["stats"] = {
                "samples": int(len(currents)),
                "peak_uA": float(np.max(currents)),
                "avg_uA": float(np.mean(currents)),
                "sample_rate_hz": float(sr_out),
            }
        else:
            data["times_5hz"] = np.array([])
            data["currents_5hz"] = np.array([])
            data["stats"] = {
                "samples": 0,
                "peak_uA": 0.0,
                "avg_uA": 0.0,
                "sample_rate_hz": float(sr_out),
            }

        return {"text": "", "data": data}

    def _run_simulation(self, duration_ms: int, mode: str = "ppk"):
        print(f"Simulating {duration_ms} ms monitoring (mode={mode}).")
        time.sleep(duration_ms / 1000.0)
        hw = PPK2_HARDWARE_SAMPLE_RATE_HZ
        req = self.sample_rate_hz
        if req >= hw - 1e-9:
            eff_hz = hw
        else:
            eff_hz = hw / float(max(1, int(round(hw / req))))
        self.output_sample_rate_hz = eff_hz
        num_samples = max(1, int(round(duration_ms / 1000.0 * eff_hz)))
        currents = np.random.normal(500, 50, num_samples)
        times = np.arange(num_samples, dtype=np.float64) / eff_hz
        step = int(max(1, round(eff_hz / 5.0)))
        return {
            "text": "",
            "data": {
                "times": times,
                "currents": currents,
                "times_5hz": times[::step],
                "currents_5hz": currents[::step],
                "stats": {
                    "samples": num_samples,
                    "peak_uA": float(np.max(currents)),
                    "avg_uA": float(np.mean(currents)),
                    "sample_rate_hz": float(eff_hz),
                },
            },
        }
