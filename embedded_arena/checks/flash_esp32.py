"""Flash an ESP32 firmware project via ESP-IDF `idf.py flash`.

Reads build/flasher_args.json produced by `compile_esp32` (or a prior
`idf.py build`) to flash all partition images (bootloader, partition table,
app) in one shot.

Required host environment:
  - `idf.py` on PATH (source $IDF_PATH/export.sh) OR IDF_PATH set in .env
The serial port is read from the YAML `port` param when present, otherwise from `ESP32_PORT`.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from exceptions import ExperimentSetupError
from schemas import RunState, CheckResult
from pydantic import BaseModel, Field
from checks.common import binary_result
from hardware.esp32_flasher import ESP32Flasher
from hardware.ppk2 import ppk2Monitor


def _binary_with_feedback(state: RunState, checks: list[tuple[str, bool, float, str]]) -> CheckResult:
    r = binary_result(checks=checks)
    return r if state.metadata.get("feedback_enabled", True) else r.model_copy(update={"feedback": None})


class Input(BaseModel):
    """Agent-settable input fields."""

    project_dir: str = Field(
        description="Sandbox-relative ESP-IDF project root. Must contain "
        "build/flasher_args.json produced by compile_esp32."
    )
    baud: int = Field(
        default=921600,
        ge=9600,
        description="Baud rate used when calling idf.py flash.",
    )


class YAMLInput(BaseModel):
    """YAML-controlled configuration parameters (not agent-settable)."""

    port: str = Field(
        default="",
        description="Serial port for the ESP32 (e.g. /dev/tty.usbmodem1101). "
        "If omitted, the check uses ESP32_PORT from the environment.",
    )
    capture_power: bool = Field(
        default=False,
        description="If true, device is powered by PPK2 only; voltage will be enabled before flashing.",
    )
    target_voltage_v: float = Field(
        default=3.3,
        ge=1.0,
        le=5.0,
        description="Target device voltage in volts (used when capture_power=true).",
    )


def check(state: RunState, agent_input: Input, yaml_input: YAMLInput) -> CheckResult:
    """Flash all partition images to the ESP32 and verify success.

    Args:
        state: RunState containing sandbox and metadata.
        agent_input: Agent-settable parameters (project_dir, port, baud).
        yaml_input: YAML-controlled parameters (capture_power, target_voltage_v).
    """
    port = yaml_input.port or os.environ.get("ESP32_PORT", "")
    if not port:
        raise ExperimentSetupError(
            "ESP32 serial port is not configured. Run scripts/setup_esp32.sh, "
            "set ESP32_PORT in .env, or set the YAML port param for this check."
        )
    idf_path = os.environ.get("IDF_PATH")
    if not idf_path and shutil.which("idf.py") is None:
        raise ExperimentSetupError(
            "IDF_PATH is not set and idf.py is not on PATH. "
            "Run scripts/setup_esp32.sh, then source .env."
        )

    checks: list[tuple[str, bool, float, str]] = []

    try:
        project_root = Path(
            state.sandbox._resolve_relative_path(agent_input.project_dir)
        )
    except ValueError as exc:
        checks.append(("project_dir inside sandbox", False, 1.0, str(exc)))
        return _binary_with_feedback(state, checks)

    if not project_root.is_dir():
        checks.append(
            (
                "project_dir exists",
                False,
                1.0,
                f"{agent_input.project_dir} is not a directory",
            )
        )
        return _binary_with_feedback(state, checks)

    flasher_args = project_root / "build" / "flasher_args.json"
    if not flasher_args.exists():
        checks.append(
            (
                "flasher_args.json present",
                False,
                1.0,
                f"missing {flasher_args} — did compile_esp32 run successfully?",
            )
        )
        return _binary_with_feedback(state, checks)
    checks.append(("flasher_args.json present", True, 0.2, str(flasher_args)))

    # Enable PPK2 voltage output if device is powered by PPK2
    monitor = None
    try:
        if yaml_input.capture_power:
            monitor = ppk2Monitor()
            if monitor.ppk2_port is None:
                checks.append(
                    (
                        "PPK2 voltage enabled",
                        False,
                        0.5,
                        "PPK2 not detected but capture_power=true",
                    )
                )
                return _binary_with_feedback(state, checks)
            monitor.voltage_on(target_voltage_v=yaml_input.target_voltage_v)
            checks.append(
                (
                    "PPK2 voltage enabled",
                    True,
                    0.1,
                    f"PPK2 output {yaml_input.target_voltage_v}V",
                )
            )

        flasher = ESP32Flasher(
            project_root=str(project_root),
            port=port,
            baud=agent_input.baud,
            post_power_settle_s=0.5 if yaml_input.capture_power else 0.0,
        )
        output, success = flasher.flash()
        flash_detail = (
            "flash finished"
            if success
            else (output[-6000:] if output else "idf.py flash failed (no flasher output)")
        )
        checks.append(("idf.py flash", success, 0.8, flash_detail))
    finally:
        # Turn off PPK2 voltage and close connection after flashing
        if monitor is not None:
            try:
                monitor.voltage_off()
            except Exception as e:
                print(f"Warning: failed to turn off PPK2 voltage: {e}")
            try:
                monitor.close()
            except Exception as e:
                print(f"Warning: failed to close PPK2 connection: {e}")

    return _binary_with_feedback(state, checks)
