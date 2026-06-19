from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import yaml

from schemas import CheckResult, RunState


def sandbox_path(state: RunState, relative_path: str) -> Path:
    return Path(state.sandbox._resolve_relative_path(relative_path))


def existing_file(
    state: RunState, relative_path: str
) -> tuple[Path | None, str | None]:
    if not relative_path:
        return None, "path is empty"
    try:
        path = sandbox_path(state, relative_path)
    except Exception as exc:
        return None, str(exc)
    if not path.exists():
        return None, f"{relative_path} does not exist"
    if not path.is_file():
        return None, f"{relative_path} is not a file"
    if path.stat().st_size == 0:
        return None, f"{relative_path} is empty"
    return path, None


def load_yaml_file(path: Path) -> tuple[Any | None, str | None]:
    try:
        with path.open() as f:
            return yaml.safe_load(f), None
    except Exception as exc:
        return None, f"could not parse YAML: {exc}"


def compile_python_if_needed(state: RunState, relative_path: str) -> tuple[bool, str]:
    if not relative_path.endswith(".py"):
        return True, "not a Python file"
    stdout, stderr, code = state.sandbox.run(
        ["python", "-m", "py_compile", relative_path]
    )
    if code != 0:
        return False, stderr or stdout
    return True, "Python file compiles"


def run_sandbox_python(
    state: RunState, source: str, timeout_seconds: int = 300
) -> tuple[bool, str]:
    with tempfile.NamedTemporaryFile(
        "w", suffix=".py", delete=False, dir=state.sandbox.sandbox_path
    ) as f:
        script_path = Path(f.name)
        f.write(source)
    relative = os.path.relpath(script_path, state.sandbox.sandbox_path)
    try:
        try:
            stdout, stderr, code = state.sandbox.run(
                ["python", relative], timeout_seconds=timeout_seconds
            )
        except subprocess.TimeoutExpired as exc:
            stdout = (
                exc.stdout.decode(errors="replace")
                if isinstance(exc.stdout, bytes)
                else (exc.stdout or "")
            )
            stderr = (
                exc.stderr.decode(errors="replace")
                if isinstance(exc.stderr, bytes)
                else (exc.stderr or "")
            )
            return (
                False,
                f"timed out after {timeout_seconds}s\n{stdout}\n{stderr}".strip(),
            )
        output = stdout if code == 0 else f"{stdout}\n{stderr}".strip()
        return code == 0, output
    finally:
        script_path.unlink(missing_ok=True)


def run_host_command(
    args: list[str],
    *,
    cwd: Path | None = None,
    timeout_seconds: int = 300,
) -> tuple[bool, str]:
    if not args or any(not isinstance(arg, str) for arg in args):
        return False, "host command must be a non-empty argv list of strings"
    try:
        completed = subprocess.run(
            args,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except FileNotFoundError as exc:
        return False, str(exc)
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        return False, f"timed out after {timeout_seconds}s\n{stdout}\n{stderr}".strip()
    output = (completed.stdout or "") + (
        "\n" + completed.stderr if completed.stderr else ""
    )
    return completed.returncode == 0, output.strip()


def executable_from_env(name: str, fallbacks: list[str]) -> str | None:
    configured = os.environ.get(name)
    if configured:
        return configured
    for candidate in fallbacks:
        found = shutil.which(candidate)
        if found:
            return found
    return None


def weighted_result(
    *,
    checks: list[tuple[str, bool, float, str]],
    success_threshold: float = 1.0,
    unit: str = "fraction",
) -> CheckResult:
    total = sum(weight for _, _, weight, _ in checks) or 1.0
    score = sum(weight for _, passed, weight, _ in checks if passed) / total
    feedback = "\n".join(
        f"[{'pass' if passed else 'fail'}] {name}: {detail}"
        for name, passed, _, detail in checks
    )
    return CheckResult(
        success=score >= success_threshold,
        score=score,
        score_unit=unit,
        feedback=feedback,
    )


def binary_result(
    *,
    checks: list[tuple[str, bool, float, str]],
) -> CheckResult:
    """All-or-nothing: score is 1.0 only when every check passes."""
    all_pass = all(passed for _, passed, _, _ in checks)
    feedback = "\n".join(
        f"[{'pass' if passed else 'fail'}] {name}: {detail}"
        for name, passed, _, detail in checks
    )
    return CheckResult(
        success=all_pass,
        score=1.0 if all_pass else 0.0,
        score_unit="binary",
        feedback=feedback,
    )
