"""Firmware flash and erase tool for the MAX78000 target.

Programs (or mass-erases) the M4 core over JTAG/SWD using
arm-none-eabi-gdb + OpenOCD. The flash path reuses the project's
.vscode/flash.gdb script which defines the `flash_m4` / `flash_m4_run`
GDB commands; the erase path drives `openocd` directly with
`max32xxx mass_erase 0` (matching the .vscode/tasks.json "erase flash"
task in the legacy firmware project).
"""

from __future__ import annotations

import os
import subprocess


DEFAULT_PROJECT_DIR = os.environ.get(
    "MAX78000_PROJECT_ROOT",
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "yolo-pico_playground")
    ),
)


class MAX78000Flasher:
    """Programs/erases the compiled ELF onto a MAX78000 over JTAG/SWD.

    All paths and OpenOCD config files are constructor parameters so
    different firmware layouts and JTAG adapters can be supported. Maxim
    SDK location is discovered via MAXIM_PATH (set by
    scripts/setup_max78000.sh), with the legacy MaximSDK install path
    accepted as a fallback for backwards compatibility.
    """

    def __init__(
        self,
        project_dir: str | None = None,
        elf_file: str = "build/yolo-pico_llm.elf",
        gdb_script: str = ".vscode/flash.gdb",
        maxim_sdk_path: str | None = None,
        ocd_interface_file: str = "cmsis-dap.cfg",
        ocd_target_file: str = "MAX78000.cfg",
        gdb: str = "arm-none-eabi-gdb",
        openocd: str = "openocd",
    ):
        self.gdb = gdb
        self.openocd = openocd
        self.project_dir = os.path.abspath(project_dir or DEFAULT_PROJECT_DIR)
        self.elf_file = elf_file
        self.gdb_script = gdb_script
        self.maxim_sdk_path = (
            maxim_sdk_path
            or os.environ.get("MAXIM_PATH")
            or os.path.expanduser("~/MaximSDK")
        )
        self.ocd_interface_file = ocd_interface_file
        self.ocd_target_file = ocd_target_file

    @property
    def _ocd_scripts_dir(self) -> str:
        return os.path.join(self.maxim_sdk_path, "Tools", "OpenOCD", "scripts")

    def flash(self, run_after: bool = True) -> tuple[str, bool]:
        """Flash the compiled ELF to the device.

        Args:
            run_after: True invokes `flash_m4_run` (program + resume into
                user code); False invokes `flash_m4` (program + halt).

        Returns:
            (output, success)
        """
        elf_path = os.path.join(self.project_dir, self.elf_file)
        gdb_script_path = os.path.join(self.project_dir, self.gdb_script)

        ocd_dir = os.path.join(self.maxim_sdk_path, "Tools", "OpenOCD")
        gdb_command = "flash_m4_run" if run_after else "flash_m4"
        flash_arg = (
            f"{gdb_command} {ocd_dir} "
            f"{self.ocd_interface_file} {self.ocd_target_file}"
        )

        command = [
            self.gdb,
            f'--cd="{self.project_dir}"',
            f'--se="{elf_path}"',
            f"--symbols={elf_path}",
            f'-x="{gdb_script_path}"',
            f'--ex="{flash_arg}"',
            "--batch",
        ]
        print(f"Flash command: {' '.join(command)}")

        try:
            process = subprocess.run(
                " ".join(command),
                shell=True,
                check=True,
                capture_output=True,
                text=True,
            )
            return process.stdout, True
        except subprocess.CalledProcessError as e:
            return (e.stdout or "") + "\n" + (e.stderr or ""), False

    def erase(self) -> tuple[str, bool]:
        """Mass-erase the MAX78000 internal flash.

        Uses the `max32xxx mass_erase 0` OpenOCD command (Maxim's vendor
        extension) — same as the "erase flash" VS Code task that ships
        with the firmware project.

        Returns:
            (output, success)
        """
        scripts_dir = self._ocd_scripts_dir
        command = [
            self.openocd,
            "-s", scripts_dir,
            "-f", f"interface/{self.ocd_interface_file}",
            "-f", f"target/{self.ocd_target_file}",
            "-c", "init; reset halt; max32xxx mass_erase 0;",
            "-c", "exit",
        ]
        print(f"Erase command: {' '.join(command)}")

        try:
            process = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
            )
            return (process.stdout or "") + (process.stderr or ""), True
        except subprocess.CalledProcessError as e:
            return (e.stdout or "") + "\n" + (e.stderr or ""), False
        except FileNotFoundError as e:
            return f"openocd not found on PATH: {e}", False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MAX78000 flash/erase utility")
    parser.add_argument("action", choices=["flash", "erase"])
    parser.add_argument("--no-run", action="store_true",
                        help="(flash only) program but don't resume execution")
    args = parser.parse_args()

    flasher = MAX78000Flasher()
    if args.action == "flash":
        output, success = flasher.flash(run_after=not args.no_run)
    else:
        output, success = flasher.erase()
    print(output)
    print("Success:", success)
