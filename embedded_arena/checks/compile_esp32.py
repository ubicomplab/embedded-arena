"""Compile a sandbox ESP32 firmware project via ESP-IDF `idf.py build`.

The agent edits firmware sources in a sandbox-relative `project_dir` using
the standard WRITE_FILE / RUN_COMMAND tools. This check runs `idf.py build`
against that directory on the host, where the ESP-IDF toolchain lives.

On success the resulting ELF under `<project_dir>/build/` is ready for
`flash_esp32` or `measure_esp32`.

Required host environment:
  - `idf.py` on PATH (source $IDF_PATH/export.sh) OR IDF_PATH set in .env
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from exceptions import ExperimentSetupError
from schemas import RunState, CheckResult
from pydantic import BaseModel, Field
from checks.common import binary_result
from hardware.esp32_compiler import ESP32Compiler


def _binary_with_feedback(state: RunState, checks: list[tuple[str, bool, float, str]]) -> CheckResult:
    r = binary_result(checks=checks)
    return r if state.metadata.get("feedback_enabled", True) else r.model_copy(update={"feedback": None})


class Input(BaseModel):
    project_dir: str = Field(
        description="Sandbox-relative path to the ESP-IDF project root (the directory "
        "containing CMakeLists.txt). After compilation, build/flasher_args.json and "
        "the application ELF are written here by idf.py."
    )
    jobs: int = Field(
        default=8,
        ge=1,
        le=64,
        description="Parallel job count passed to idf.py build (-j N).",
    )


def check(state: RunState, input: Input) -> CheckResult:
    """Run `idf.py build` against the sandbox ESP32 project and verify the ELF."""
    idf_path = os.environ.get("IDF_PATH")
    idf_on_path = shutil.which("idf.py") is not None
    if not idf_path and not idf_on_path:
        raise ExperimentSetupError(
            "IDF_PATH is not set and idf.py is not on PATH. "
            "Run scripts/setup_esp32.sh, then source .env."
        )

    checks: list[tuple[str, bool, float, str]] = []

    try:
        project_root = Path(state.sandbox._resolve_relative_path(input.project_dir))
    except ValueError as exc:
        checks.append(("project_dir inside sandbox", False, 1.0, str(exc)))
        return _binary_with_feedback(state, checks)

    if not project_root.is_dir():
        checks.append(
            (
                "project_dir exists",
                False,
                1.0,
                f"{input.project_dir} is not a directory",
            )
        )
        return _binary_with_feedback(state, checks)
    checks.append(("project_dir exists", True, 0.1, str(project_root)))

    cmakelists = project_root / "CMakeLists.txt"
    if not cmakelists.exists():
        checks.append(
            (
                "CMakeLists.txt present",
                False,
                1.0,
                f"missing {cmakelists}",
            )
        )
        return _binary_with_feedback(state, checks)
    checks.append(("CMakeLists.txt present", True, 0.1, "found"))

    compiler = ESP32Compiler(project_root=str(project_root), jobs=input.jobs)
    output, success = compiler.compile()
    build_detail = (
        "build finished"
        if success
        else (output[-8000:] if output else "idf.py build failed (no compiler output)")
    )
    checks.append(("idf.py build", success, 0.5, build_detail))
    if not success:
        return _binary_with_feedback(state, checks)

    elf = compiler.get_elf()
    elf_present = elf is not None and Path(elf).stat().st_size > 0
    checks.append(
        (
            "ELF produced",
            elf_present,
            0.3,
            elf if elf_present else "no ELF found in build/ after idf.py build",  # type: ignore
        )
    )
    return _binary_with_feedback(state, checks)
