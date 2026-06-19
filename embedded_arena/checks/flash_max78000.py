"""Flash a MAX78000 ELF onto the connected target.

Two flavors, picked by which Input field the agent populates:

  - `project_dir` (sandbox-relative): use after a `compile_max78000` check
    has built the agent's edited firmware in the sandbox. The ELF is
    expected at `<project_dir>/build/<project_name>.elf`.

  - `prefix`: use after a `synthesis_max78000` check has produced a
    synthesized project under
    `${AI8X_SYNTHESIS_DIR}/${AI8X_TEST_DIR}/<prefix>`. The ELF is
    expected at `build/<prefix>.elf` inside that project.

Exactly one of the two must be supplied. Mass-erases first by default
(set `erase_first=False` to skip).
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

from exceptions import ExperimentSetupError
from schemas import RunState, CheckResult
from pydantic import BaseModel, Field, model_validator
from checks.common import binary_result
from hardware.max78000_flasher import MAX78000Flasher
from hardware.ppk2 import ppk2Monitor


def _binary_with_feedback(state: RunState, checks: list[tuple[str, bool, float, str]]) -> CheckResult:
    r = binary_result(checks=checks)
    return r if state.metadata.get("feedback_enabled", True) else r.model_copy(update={"feedback": None})


class Input(BaseModel):
    """Agent-settable input fields."""

    project_dir: str | None = Field(
        default=None,
        description="Sandbox-relative path to a firmware project containing build/<project_name>.elf. Use this after compile_max78000.",
    )
    project_name: str = Field(
        default="yolo-pico_llm",
        description="Project name passed to make; the ELF is expected at <project_dir>/build/<project_name>.elf.",
    )
    prefix: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z0-9_.-]+$",
        description="Synthesis prefix; resolves to ${AI8X_SYNTHESIS_DIR}/${AI8X_TEST_DIR}/<prefix>/build/<prefix>.elf. Use this after synthesis_max78000.",
    )
    erase_first: bool = True

    @model_validator(mode="after")
    def _exactly_one_source(self):
        if (self.project_dir is None) == (self.prefix is None):
            raise ValueError(
                "flash_max78000 requires exactly one of `project_dir` or `prefix`."
            )
        return self


class YAMLInput(BaseModel):
    """YAML-controlled configuration parameters (not agent-settable)."""

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


def resolve_elf(state: RunState, agent_input: Input) -> tuple[Path | None, str]:
    """Resolve the host-absolute ELF path from the Input. Returns (path, error)."""
    if agent_input.project_dir is not None:
        try:
            project_root = Path(
                state.sandbox._resolve_relative_path(agent_input.project_dir)
            )
        except ValueError as exc:
            return None, str(exc)
        if not project_root.exists():
            return None, f"sandbox project_dir {agent_input.project_dir} does not exist"
        elf = project_root / "build" / f"{agent_input.project_name}.elf"
        if not elf.exists():
            return None, (
                f"expected ELF at sandbox path {agent_input.project_dir}/build/{agent_input.project_name}.elf "
                "— did compile_max78000 run successfully?"
            )
        return elf, ""

    ai8x_synthesis_dir = os.environ.get("AI8X_SYNTHESIS_DIR")
    if not ai8x_synthesis_dir:
        raise ExperimentSetupError(
            "AI8X_SYNTHESIS_DIR is not set. Run scripts/setup_max78000.sh, then source .env."
        )
    test_dir = os.environ.get("AI8X_TEST_DIR", "sdk/Examples/MAX78000/CNN")
    project_dir = Path(ai8x_synthesis_dir) / test_dir / agent_input.prefix  # type: ignore
    elf = project_dir / "build" / f"{agent_input.prefix}.elf"
    if not elf.exists():
        return None, (
            f"expected ELF at {elf} — did synthesis_max78000 run for the same prefix?"
        )
    return elf, ""


def check(state: RunState, agent_input: Input, yaml_input: YAMLInput) -> CheckResult:
    """Erase + flash the resolved ELF and report success.

    Args:
        state: RunState containing sandbox and metadata.
        agent_input: Agent-settable parameters (project_dir/prefix, erase_first, etc.).
        yaml_input: YAML-controlled parameters (capture_power, target_voltage_v).
    """
    if shutil.which("arm-none-eabi-gdb") is None:
        raise ExperimentSetupError(
            "arm-none-eabi-gdb is not on PATH. Run scripts/setup_max78000.sh."
        )
    if shutil.which("openocd") is None:
        raise ExperimentSetupError(
            "openocd is not on PATH. Install OpenOCD (Maxim SDK ships one in Tools/OpenOCD)."
        )

    checks: list[tuple[str, bool, float, str]] = []
    erase_output: str = ""
    flash_output: str = ""

    elf, elf_error = resolve_elf(state, agent_input)
    checks.append(("ELF located", elf is not None, 0.20, "ok" if elf is not None else elf_error))
    if elf is None:
        return _binary_with_feedback(state, checks)

    # Enable PPK2 voltage output if device is powered by PPK2
    monitor = None
    try:
        if yaml_input.capture_power:
            monitor = ppk2Monitor()
            if not monitor.ppk2_candidates:
                checks.append(
                    (
                        "PPK2 voltage enabled",
                        False,
                        0.15,
                        "PPK2 not detected but capture_power=true",
                    )
                )
                return _binary_with_feedback(state, checks)
            voltage_ok = monitor.voltage_on(
                target_voltage_v=yaml_input.target_voltage_v
            )
            if not voltage_ok:
                checks.append(
                    (
                        "PPK2 voltage enabled",
                        False,
                        0.15,
                        f"PPK2 candidates {monitor.ppk2_candidates} did not respond to control commands",
                    )
                )
                return _binary_with_feedback(state, checks)
            time.sleep(2.0)  # allow target to boot before SWD access
            checks.append(
                (
                    "PPK2 voltage enabled",
                    True,
                    0.05,
                    f"PPK2 output {yaml_input.target_voltage_v}V on {monitor.ppk2_port}",
                )
            )

        project_dir = elf.parent.parent
        flasher = MAX78000Flasher(
            project_dir=str(project_dir),
            elf_file=os.path.relpath(elf, project_dir),
        )

        _MAX_FLASH_ATTEMPTS = 3
        _FLASH_RETRY_DELAY_S = 2.0

        if agent_input.erase_first:
            erase_ok = False
            erase_output = ""
            for attempt in range(_MAX_FLASH_ATTEMPTS):
                erase_output, erase_ok = flasher.erase()
                if erase_ok:
                    break
                if attempt < _MAX_FLASH_ATTEMPTS - 1:
                    print(f"[flash_max78000] erase attempt {attempt + 1} failed, retrying in {_FLASH_RETRY_DELAY_S}s...")
                    time.sleep(_FLASH_RETRY_DELAY_S)
            erase_detail = erase_output[-2000:] if not erase_ok else "ok"
            checks.append(("mass erase", erase_ok, 0.30, erase_detail or "ok"))
            if not erase_ok:
                return _binary_with_feedback(state, checks)
        else:
            checks.append(("mass erase", True, 0.30, "ok"))

        flash_ok = False
        flash_output = ""
        for attempt in range(_MAX_FLASH_ATTEMPTS):
            flash_output, flash_ok = flasher.flash(run_after=True)
            if flash_ok:
                break
            if attempt < _MAX_FLASH_ATTEMPTS - 1:
                print(f"[flash_max78000] flash attempt {attempt + 1} failed, retrying in {_FLASH_RETRY_DELAY_S}s...")
                time.sleep(_FLASH_RETRY_DELAY_S)
        flash_detail = flash_output[-4000:] if not flash_ok else "ok"
        checks.append(("flash + run", flash_ok, 0.50, flash_detail or "ok"))
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

    result = _binary_with_feedback(state, checks)
    _save_iter_json(
        state,
        {
            "flash": {
                "success": result.success,
                "erase_output": erase_output if agent_input.erase_first else "skipped",
                "flash_output": flash_output,
            }
        },
    )
    return result


def _save_iter_json(state: RunState, data: dict) -> None:
    output_dir_str = state.metadata.get("output_dir")
    if not output_dir_str:
        return
    iter_dir = (
        Path(output_dir_str)
        / f"trial_{state.trial_index}"
        / f"iter_{state.iteration_index}"
    )
    iter_dir.mkdir(parents=True, exist_ok=True)
    existing: dict = {}
    path = iter_dir / "iter_result.json"
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except Exception:
            pass
    existing.update(data)
    path.write_text(json.dumps(existing, indent=2))
