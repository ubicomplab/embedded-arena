"""Firmware build tool for the MAX78000 target using the Maxim SDK.

Wraps a `make` invocation with the correct environment variables so a
caller can trigger a build from any working directory.

Defaults are read from environment variables (matches the AgenticHL
.env workflow set up by scripts/setup_max78000.sh):
  - MAXIM_PATH:  Maxim SDK root (required at build time).

`project_root` defaults to the playground firmware directory shipped with
this repo, but any project that uses the same Makefile layout can be
passed in explicitly.
"""

from __future__ import annotations

import os
import subprocess


DEFAULT_PROJECT_ROOT = os.environ.get(
    "MAX78000_PROJECT_ROOT",
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "yolo-pico_playground")
    ),
)


class MAX78000Compiler:
    """Invokes `make` against a MAX78000 firmware project.

    Paths/identifiers are constructor parameters so multiple firmware
    projects can be built from the same host. Anything not passed in
    falls back to the environment then to documented defaults.
    """

    def __init__(
        self,
        project_root: str | None = None,
        project_name: str = "yolo-pico_llm",
        target: str = "MAX78000",
        board: str = "CAM02_RevA",
        maxim_path: str | None = None,
        jobs: int = 8,
    ):
        self.project_root = os.path.abspath(project_root or DEFAULT_PROJECT_ROOT)
        self.project_name = project_name
        self.target = target
        self.board = board
        self.maxim_path = maxim_path or os.environ.get("MAXIM_PATH", "")
        self.jobs = jobs

    def compile(self) -> tuple[str, bool]:
        """Build the firmware.

        Returns:
            (output, success) — output contains stdout on success or
            stdout+stderr on failure.
        """
        assert self.maxim_path, (
            "MAXIM_PATH is not set; pass maxim_path= or run scripts/setup_max78000.sh"
        )

        command = [
            "make", "-r", "-j", str(self.jobs),
            "--output-sync=target",
            "--no-print-directory",
            f"TARGET={self.target}",
            f"BOARD={self.board}",
            f"MAXIM_PATH={self.maxim_path}",
            "MAKE=make",
            f"PROJECT={self.project_name}",
        ]
        print(f"Compile command: {' '.join(command)}")

        try:
            process = subprocess.run(
                " ".join(command),
                cwd=self.project_root,
                shell=True,
                check=True,
                capture_output=True,
                text=True,
                executable="/bin/zsh" if os.path.exists("/bin/zsh") else "/bin/bash",
            )
            return process.stdout, True
        except subprocess.CalledProcessError as e:
            return (e.stdout or "") + "\n" + (e.stderr or ""), False

    def clean(self) -> tuple[str, bool]:
        """Run `make clean` for the firmware project."""
        assert self.maxim_path, "MAXIM_PATH is not set"

        command = [
            "make", "-j", str(self.jobs), "clean",
            "--output-sync=target",
            "--no-print-directory",
            f"TARGET={self.target}",
            f"BOARD={self.board}",
            f"MAXIM_PATH={self.maxim_path}",
            "MAKE=make",
            f"PROJECT={self.project_name}",
        ]
        try:
            process = subprocess.run(
                " ".join(command),
                cwd=self.project_root,
                shell=True,
                check=True,
                capture_output=True,
                text=True,
                executable="/bin/zsh" if os.path.exists("/bin/zsh") else "/bin/bash",
            )
            return process.stdout, True
        except subprocess.CalledProcessError as e:
            return (e.stdout or "") + "\n" + (e.stderr or ""), False


if __name__ == "__main__":
    compiler = MAX78000Compiler()
    output, success = compiler.compile()
    print(output)
    print("Success:", success)
