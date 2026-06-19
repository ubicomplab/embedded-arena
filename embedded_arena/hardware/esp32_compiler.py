"""Firmware build tool for the ESP32 target using ESP-IDF.

Wraps `idf.py build` so checks can trigger a build without re-implementing
the IDF environment handling. IDF_PATH must be set (by sourcing
$IDF_PATH/export.sh or running scripts/setup_esp32.sh).

The compiled ELF is discovered from the build/ directory because the project
name is defined in CMakeLists.txt and may not match the directory name.
"""

from __future__ import annotations

import glob
import os
import re
import subprocess
from pathlib import Path

# ESP-IDF often prints only a pointer to real logs, e.g.:
# "output of the command is in the .../idf_py_stderr_output_NNN and .../idf_py_stdout_output_NNN"
_IDF_SPILL_LOG_RE = re.compile(
    r"(/[^\s]+idf_py_(?:stderr|stdout)_output_\d+)",
    re.MULTILINE,
)


class ESP32Compiler:
    """Invokes `idf.py build` against an ESP-IDF firmware project.

    All paths are constructor parameters so multiple firmware projects can
    be compiled from the same host. Anything not passed in falls back to
    the environment then to documented defaults.
    """

    def __init__(
        self,
        project_root: str,
        idf_path: str | None = None,
        jobs: int = 8,
    ):
        self.project_root = os.path.abspath(project_root)
        self.idf_path = idf_path or os.environ.get("IDF_PATH", "")
        self.jobs = jobs

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def compile(self) -> tuple[str, bool]:
        """Build the firmware with `idf.py build`.

        Runs an incremental build first. If that fails (e.g. stale CMake cache),
        runs `idf.py fullclean` once and rebuilds so normal iterations stay fast.

        Logs are streamed to the parent process stdout/stderr. The returned string
        is a short summary for feedback.

        Returns:
            (output, success) — brief summary; full logs on the console.
        """
        idf_py = self._find_idf_py()
        if idf_py is None:
            return (
                "idf.py not found. Source $IDF_PATH/export.sh or run "
                "scripts/setup_esp32.sh, then source .env.",
                False,
            )

        env = self._make_idf_env(idf_py)

        # Use ESP-IDF's Python venv for idf.py so imports (esp_idf_monitor, etc.)
        # resolve correctly and don't depend on the script's Python environment.
        idf_python = env.get("IDF_PYTHON_ENV_PATH")
        if idf_python and os.path.isdir(idf_python):
            idf_python = os.path.join(idf_python, "bin", "python")
        if not idf_python:
            # Look under the common ~/.espressif/python_env directory
            pyenv_dir = os.path.expanduser("~/.espressif/python_env")
            if os.path.isdir(pyenv_dir):
                # pick the first env that contains bin/python
                for name in sorted(os.listdir(pyenv_dir)):
                    candidate = os.path.join(pyenv_dir, name, "bin", "python")
                    if os.path.exists(candidate):
                        idf_python = candidate
                        break

        if not idf_python or not os.path.exists(idf_python):
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

        # idf.py does not accept `-j` after `build` on many ESP-IDF versions
        # ("Error: No such option: -j"). Honor parallel job count via CMake's env
        # (used when idf invokes cmake --build) instead.
        build_cmd = [idf_python, idf_py, "build"]
        build_env = dict(env)
        if self.jobs and self.jobs > 0:
            build_env["CMAKE_BUILD_PARALLEL_LEVEL"] = str(self.jobs)

        def _run_build() -> subprocess.CompletedProcess:
            print(f"ESP32 compile command: {' '.join(build_cmd)}")
            return self._run_idf_command(
                build_cmd,
                env=build_env,
                timeout=600,
            )

        def _timeout_msg(exc: subprocess.TimeoutExpired) -> str:
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
                f"Build timed out after 600s (partial output may appear above).\n{tail}"
            ).strip()

        try:
            process = _run_build()
        except subprocess.TimeoutExpired as exc:
            return _timeout_msg(exc), False
        except FileNotFoundError as exc:
            return f"idf.py not executable: {exc}", False

        if process.returncode == 0:
            return "build finished (full log streamed to terminal)", True

        first_build_output = self._inline_idf_spill_logs(self._process_output(process))
        print(
            "[ESP32Compiler] incremental build failed; running idf.py fullclean "
            "then building again",
            flush=True,
        )
        fullclean_cmd = [idf_python, idf_py, "fullclean"]
        print(f"ESP32 fullclean command: {' '.join(fullclean_cmd)}")
        try:
            self._run_idf_command(
                fullclean_cmd,
                env=env,
                timeout=60,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        try:
            process = _run_build()
        except subprocess.TimeoutExpired as exc:
            return _timeout_msg(exc), False
        except FileNotFoundError as exc:
            return f"idf.py not executable: {exc}", False

        success = process.returncode == 0
        second_build_output = self._inline_idf_spill_logs(self._process_output(process))
        combined_output = "\n\n".join(
            part for part in [first_build_output, second_build_output] if part.strip()
        )
        if success:
            output = (
                "build finished after fullclean+rebuild (full log streamed to terminal)"
            )
        else:
            output = (
                f"idf.py build failed with exit code {process.returncode} "
                f"(after fullclean retry).\n{combined_output}".strip()
            )
        return output, success

    def _run_idf_command(
        self,
        command: list[str],
        env: dict[str, str],
        timeout: int,
    ) -> subprocess.CompletedProcess:
        """Run idf in the project cwd; inherit stdout/stderr so logs appear in the host terminal."""
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

    def _inline_idf_spill_logs(self, captured: str, *, max_chars_per_file: int = 400_000) -> str:
        """Append contents of idf.py spill log files when IDF only prints their paths.

        Newer ESP-IDF redirects verbose ninja/cmake output to build/log/idf_py_*_output_*.
        Without inlining, captured stdout/stderr is useless for agent feedback.
        """
        if not captured or "idf_py_" not in captured:
            return captured
        proj_real = os.path.realpath(self.project_root)
        paths: list[str] = []
        for m in _IDF_SPILL_LOG_RE.finditer(captured):
            p = m.group(1)
            if p not in paths:
                paths.append(p)
        if not paths and (
            "output of the command is in" in captured.lower()
            or "idf_py_stderr" in captured
            or "idf_py_stdout" in captured
        ):
            log_dir = Path(self.project_root) / "build" / "log"
            if log_dir.is_dir():
                candidates = [p for p in log_dir.glob("idf_py_*_output_*") if p.is_file()]
                candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                paths = [str(p) for p in candidates[:2]]
        extras: list[str] = []
        for abs_path in paths:
            try:
                log_real = os.path.realpath(abs_path)
            except OSError:
                continue
            if not log_real.startswith(proj_real + os.sep) and log_real != proj_real:
                continue
            path = Path(abs_path)
            if not path.is_file():
                continue
            try:
                body = path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                extras.append(f"\n--- could not read {abs_path}: {exc} ---\n")
                continue
            if len(body) > max_chars_per_file:
                body = (
                    f"...[log truncated, showing last {max_chars_per_file} chars]\n"
                    + body[-max_chars_per_file:]
                )
            extras.append(f"\n--- contents of {abs_path} ---\n{body}")
        if not extras:
            return captured
        return (captured.strip() + "\n" + "\n".join(extras)).strip()

    def get_elf(self) -> str | None:
        """Return the path to the main application ELF, or None if not found."""
        build_dir = os.path.join(self.project_root, "build")
        elfs = glob.glob(os.path.join(build_dir, "*.elf"))
        if not elfs:
            return None
        main_elfs = [e for e in elfs if "bootloader" not in os.path.basename(e).lower()]
        return main_elfs[0] if main_elfs else elfs[0]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_idf_py(self) -> str | None:
        import shutil
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

    parser = argparse.ArgumentParser(description="ESP32 firmware build utility")
    parser.add_argument("project_root", nargs="?", default=".", help="Firmware project root")
    args = parser.parse_args()

    compiler = ESP32Compiler(project_root=args.project_root)
    output, success = compiler.compile()
    print(output)
    print("ELF:", compiler.get_elf())
    print("Success:", success)
