"""Firmware flash tool for the ESP32 target using ESP-IDF.

Programs the ESP32 over USB-CDC/JTAG using `idf.py flash`. The device
serial port is discovered from the ESP32_PORT environment variable (set by
scripts/setup_esp32.sh) or passed explicitly.

`idf.py flash` reads build/flasher_args.json produced by the build step so
all partition images (bootloader, partition table, app) are flashed correctly
in one shot without needing to know each image's address.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import time

from exceptions import ExperimentSetupError


def verify_esp32_serial_port_ready(
    port: str,
    *,
    probe_baud: int = 115200,
    timeout_s: float = 2.0,
    post_power_settle_s: float = 0.0,
) -> None:
    """Sanity-check the ESP32 serial device before flashing.

    Opens the port briefly with pyserial. Raises ExperimentSetupError if the
    device path does not exist or the port cannot be opened (missing board,
    wrong port, or already in use by another process).

    Args:
        port: Serial device (e.g. /dev/tty.usbmodem*, COM3).
        probe_baud: Baud used only for the probe open (115200 is safe for CDC).
        timeout_s: Serial read timeout for the probe open.
        post_power_settle_s: Optional delay after enabling external power (PPK2)
            so USB enumeration can finish before probing.
    """
    if not port or not str(port).strip():
        raise ExperimentSetupError("ESP32 flash port is empty; set port in experiment YAML or ESP32_PORT.")
    port = str(port).strip()
    if post_power_settle_s > 0:
        time.sleep(post_power_settle_s)
    # Non-Windows: fail fast if the device node is absent.
    if os.name != "nt" and port.startswith("/") and not os.path.exists(port):
        raise ExperimentSetupError(
            f"ESP32 serial device does not exist: {port!r}. "
            "Plug in the board, confirm the port in your OS, and update the YAML `port`."
        )
    try:
        import serial
    except ImportError as exc:
        raise ExperimentSetupError(
            "pyserial is required for the ESP32 pre-flash serial sanity check. "
            "Install with: pip install pyserial"
        ) from exc
    try:
        ser = serial.Serial(
            port,
            probe_baud,
            timeout=timeout_s,
            write_timeout=timeout_s,
        )
        try:
            ser.reset_input_buffer()
        except Exception:
            pass
        ser.close()
    except Exception as exc:
        raise ExperimentSetupError(
            f"Cannot open ESP32 serial port {port!r} before flash "
            f"(board unplugged, wrong port, permissions, or port in use): {exc}"
        ) from exc


class ESP32Flasher:
    """Programs an ESP32 over its USB-CDC serial port using idf.py flash.

    Requires:
      - ESP32_PORT (env): serial device, e.g. /dev/tty.usbmodem... or COM3
      - IDF_PATH (env): ESP-IDF root, if idf.py is not on PATH
    """

    def __init__(
        self,
        project_root: str,
        port: str | None = None,
        baud: int = 921600,
        idf_path: str | None = None,
        *,
        post_power_settle_s: float = 0.0,
    ):
        self.project_root = os.path.abspath(project_root)
        self.port = port or os.environ.get("ESP32_PORT")
        self.baud = int(os.environ.get("ESP32_BAUD", baud))
        self.idf_path = idf_path or os.environ.get("IDF_PATH", "")
        self.post_power_settle_s = float(post_power_settle_s)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def flash(self) -> tuple[str, bool]:
        """Flash all partition images to the device.

        Returns:
            (output, success)
        """
        if not self.port:
            return (
                "ESP32_PORT is not set. Connect the ESP32, find its serial port, "
                "and set ESP32_PORT in .env (or run scripts/setup_esp32.sh).",
                False,
            )

        idf_py = self._find_idf_py()
        if idf_py is None:
            return (
                "idf.py not found. Source $IDF_PATH/export.sh or run "
                "scripts/setup_esp32.sh, then source .env.",
                False,
            )
        env = self._make_idf_env(idf_py)
        # Prefer invoking idf.py with the ESP-IDF Python venv if available so
        # idf.py imports (esp_idf_monitor, etc.) resolve correctly even when
        # the current shell hasn't sourced export.sh.
        # Use ESP-IDF's Python venv for idf.py so imports (esp_idf_monitor, etc.)
        # resolve correctly and don't depend on the script's Python environment.
        idf_python = env.get("IDF_PYTHON_ENV_PATH")
        if not idf_python:
            pyenv_dir = os.path.expanduser("~/.espressif/python_env")
            if os.path.isdir(pyenv_dir):
                for name in sorted(os.listdir(pyenv_dir)):
                    candidate = os.path.join(pyenv_dir, name, "bin", "python")
                    if os.path.isfile(candidate):
                        idf_python = candidate
                        break

        if not idf_python or not os.path.isfile(idf_python):
            return (
                "ESP-IDF Python environment not found. Set IDF_PYTHON_ENV_PATH or "
                "ensure ~/.espressif/python_env contains a Python installation.",
                False,
            )

        # Pin the venv for idf.py's self-bootstrap: it re-detects `python3` from PATH
        # and computes the venv name from that Python's version, so a system python3
        # of a different minor version makes IDF look up a venv that doesn't exist.
        venv_root = os.path.dirname(os.path.dirname(idf_python))
        venv_bin = os.path.join(venv_root, "bin")
        env["IDF_PYTHON_ENV_PATH"] = venv_root
        env["PATH"] = venv_bin + os.pathsep + env.get("PATH", "")

        # Ensure the board is reachable on the configured serial port before idf.py flash.
        verify_esp32_serial_port_ready(
            self.port,
            post_power_settle_s=self.post_power_settle_s,
        )

        command = [idf_python, idf_py, "-p", self.port, "-b", str(self.baud), "flash"]

        print(f"ESP32 flash command: {' '.join(command)}")

        try:
            process = self._run_idf_command(
                command,
                env=env,
                timeout=120,
            )
            ok = process.returncode == 0
            process_output = self._process_output(process)
            if ok:
                out = "flash finished (full log streamed to terminal)"
            else:
                out = (
                    f"idf.py flash failed with exit code {process.returncode}. "
                    f"\n{process_output}".strip()
                )
            return out, ok
        except subprocess.TimeoutExpired as exc:
            tail = ""
            if exc.stdout:
                tail += exc.stdout if isinstance(exc.stdout, str) else exc.stdout.decode(
                    "utf-8", errors="replace"
                )
            if exc.stderr:
                tail += exc.stderr if isinstance(exc.stderr, str) else exc.stderr.decode(
                    "utf-8", errors="replace"
                )
            return (
                f"Flash timed out after 120s (partial output may appear above).\n{tail}"
            ).strip(),
            False,
        except FileNotFoundError as exc:
            return f"idf.py not executable: {exc}", False

    def _run_idf_command(
        self,
        command: list[str],
        env: dict[str, str],
        timeout: int,
    ) -> subprocess.CompletedProcess:
        """Flash via idf; inherit stdout/stderr so progress appears in the host terminal."""
        export_sh = os.path.join(env.get("IDF_PATH", ""), "export.sh")
        if os.path.isfile(export_sh):
            shell_parts = [f"source {shlex.quote(export_sh)} 1>&2", "&&"]
            shell_parts.extend(shlex.quote(part) for part in command)
            shell_cmd = " ".join(shell_parts)
            process = subprocess.run(
                ["bash", "-lc", shell_cmd],
                cwd=self.project_root,
                timeout=timeout,
                env=env,
                capture_output=True,
                text=True,
            )
            self._emit_output(process)
            return process

        process = subprocess.run(
            command,
            cwd=self.project_root,
            timeout=timeout,
            env=env,
            capture_output=True,
            text=True,
        )
        self._emit_output(process)
        return process

    def _process_output(self, process: subprocess.CompletedProcess) -> str:
        stdout = process.stdout if isinstance(process.stdout, str) else ""
        stderr = process.stderr if isinstance(process.stderr, str) else ""
        return (stdout + ("\n" + stderr if stderr else "")).strip()

    def _emit_output(self, process: subprocess.CompletedProcess) -> None:
        if process.stdout:
            print(process.stdout, end="" if process.stdout.endswith("\n") else "\n")
        if process.stderr:
            print(process.stderr, end="" if process.stderr.endswith("\n") else "\n")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_idf_py(self) -> str | None:
        found = shutil.which("idf.py")
        if found:
            return found
        if self.idf_path:
            candidate = os.path.join(self.idf_path, "tools", "idf.py")
            if os.path.isfile(candidate):
                return candidate
        return None

    def _make_idf_env(self, idf_py: str) -> dict[str, str]:
        env = dict(os.environ)
        idf_path = self.idf_path
        if not idf_path:
            # idf.py is typically at <IDF_PATH>/tools/idf.py
            idf_path = os.path.dirname(os.path.dirname(os.path.abspath(idf_py)))
        if idf_path:
            env["IDF_PATH"] = idf_path

        export_sh = os.path.join(idf_path, "export.sh") if idf_path else ""
        if not export_sh or not os.path.isfile(export_sh):
            return env

        shell_cmd = f'source "{export_sh}" >/dev/null 2>&1; env -0'
        try:
            result = subprocess.run(
                ["bash", "-lc", shell_cmd],
                capture_output=True,
                text=False,
                timeout=60,
                check=False,
            )
            if result.stdout:
                for chunk in result.stdout.split(b"\0"):
                    if not chunk:
                        continue
                    key, _, value = chunk.partition(b"=")
                    if not key:
                        continue
                    env[key.decode("utf-8", errors="ignore")] = value.decode(
                        "utf-8", errors="ignore"
                    )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            # Continue with the inherited env if shell/bootstrap is unavailable.
            pass

        return env


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ESP32 flash utility")
    parser.add_argument("project_root", nargs="?", default=".", help="Firmware project root")
    parser.add_argument("--port", default=None, help="Override ESP32_PORT env var")
    parser.add_argument("--baud", type=int, default=921600)
    args = parser.parse_args()

    flasher = ESP32Flasher(project_root=args.project_root, port=args.port, baud=args.baud)
    output, success = flasher.flash()
    print(output)
    print("Success:", success)
