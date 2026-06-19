"""Compile a sandbox MAX78000 firmware project via the Maxim SDK Makefile.

The agent edits firmware sources in a sandbox-relative `project_dir` (most
commonly the seeded copy of `firmware/max78000/yolo-pico`) using the standard
WRITE_FILE / RUN_COMMAND tools. This check then runs `make` against that
directory on the host, where the Arm GNU Toolchain and Maxim SDK live.
On success the resulting `<project_dir>/build/<project_name>.elf` is ready
for `flash_max78000` or `measure_max78000`.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from exceptions import ExperimentSetupError
from schemas import RunState, CheckResult
from pydantic import BaseModel, Field
from checks.common import binary_result
from hardware.max78000_compiler import MAX78000Compiler


def _binary_with_feedback(state: RunState, checks: list[tuple[str, bool, float, str]]) -> CheckResult:
    r = binary_result(checks=checks)
    return r if state.metadata.get("feedback_enabled", True) else r.model_copy(update={"feedback": None})


class Input(BaseModel):
    project_dir: str = Field(
        description="Sandbox-relative path to the firmware project root (the directory "
        "containing the Makefile). After compilation, build/<project_name>.elf is written here."
    )
    project_name: str = Field(
        default="yolo-pico_llm",
        description="PROJECT= make variable used when invoking make. The output ELF is "
        "expected at <project_dir>/build/<project_name>.elf.",
    )
    target: str = Field(
        default="MAX78000",
        description="TARGET= make variable (Maxim SDK chip identifier).",
    )
    board: str = Field(
        default="CAM02_RevA",
        description="BOARD= make variable (Maxim SDK board identifier matching your hardware).",
    )
    jobs: int = Field(
        default=8,
        ge=1,
        le=64,
        description="Parallel job count passed to make (-j N).",
    )


def check(state: RunState, input: Input) -> CheckResult:
    """Run `make` against the sandbox firmware project and verify the ELF."""
    if shutil.which("arm-none-eabi-gcc") is None:
        raise ExperimentSetupError(
            "arm-none-eabi-gcc is not on PATH. Run scripts/setup_max78000.sh."
        )
    if not os.environ.get("MAXIM_PATH"):
        raise ExperimentSetupError(
            "MAXIM_PATH is not set. Run scripts/setup_max78000.sh, then source .env."
        )

    checks: list[tuple[str, bool, float, str]] = []

    try:
        project_root = Path(state.sandbox._resolve_relative_path(input.project_dir))
    except ValueError as exc:
        checks.append(("project_dir inside sandbox", False, 0.20, str(exc)))
        return _binary_with_feedback(state, checks)

    if not project_root.is_dir():
        checks.append(
            (
                "project_dir exists",
                False,
                0.20,
                f"{input.project_dir} is not a directory",
            )
        )
        return _binary_with_feedback(state, checks)
    checks.append(("project_dir exists", True, 0.10, "ok"))

    makefile = project_root / "Makefile"
    checks.append(
        (
            "Makefile present",
            makefile.exists(),
            0.10,
            "ok" if makefile.exists() else f"missing {makefile}",
        )
    )
    if not makefile.exists():
        return _binary_with_feedback(state, checks)

    compiler = MAX78000Compiler(
        project_root=str(project_root),
        project_name=input.project_name,
        target=input.target,
        board=input.board,
        jobs=input.jobs,
    )
    output, success = compiler.compile()
    build_detail = output[-6000:] if not success else "make finished"
    checks.append(("make completes", success, 0.50, build_detail or "make finished"))
    if not success:
        result = _binary_with_feedback(state, checks)
        _save_iter_json(state, {"compile": {"success": False, "output": output}})
        return result

    elf = project_root / "build" / f"{input.project_name}.elf"
    elf_present = elf.exists() and elf.stat().st_size > 0
    checks.append(
        (
            "ELF produced",
            elf_present,
            0.30,
            "ok" if elf_present else f"{elf} missing or empty after make",
        )
    )
    result = _binary_with_feedback(state, checks)
    _save_iter_json(state, {"compile": {"success": elf_present, "output": output}})
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
