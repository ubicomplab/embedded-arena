"""Environment diagnostics for EmbeddedArena."""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

CHECK = "[ok]"
WARN = "[warn]"
FAIL = "[fail]"


def _print(label: str, message: str) -> None:
    print(f"{label} {message}")


def _cmd_ok(args: list[str], timeout: int = 10) -> tuple[bool, str]:
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)
    except Exception as exc:
        return False, str(exc)
    text = (result.stdout or result.stderr or "").strip()
    return result.returncode == 0, text


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Check EmbeddedArena host setup.")
    parser.add_argument("--strict", action="store_true", help="Exit nonzero on warnings as well as failures.")
    args = parser.parse_args(argv)

    load_dotenv()
    warnings = 0
    failures = 0

    if sys.version_info >= (3, 10):
        _print(CHECK, f"Python {sys.version.split()[0]}")
    else:
        _print(FAIL, f"Python >=3.10 required, found {sys.version.split()[0]}")
        failures += 1

    if shutil.which("docker"):
        ok, detail = _cmd_ok(["docker", "ps"])
        if ok:
            _print(CHECK, "Docker is running")
        else:
            _print(FAIL, f"Docker command failed: {detail}")
            failures += 1
    else:
        _print(FAIL, "Docker CLI not found")
        failures += 1

    for key in ("OPENAI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY"):
        if os.environ.get(key):
            _print(CHECK, f"{key} is set")
        else:
            _print(WARN, f"{key} is not set")
            warnings += 1

    for asset in [
        ".data/coco.zip",
        ".data/huggingface/models/KoelLabs/xlsr-english-01",
        ".data/huggingface/datasets/KoelLabs/SpeechOcean",
    ]:
        if Path(asset).exists():
            _print(CHECK, f"asset present: {asset}")
        else:
            _print(WARN, f"asset missing: {asset}")
            warnings += 1

    for key, label in [
        ("MAXIM_PATH", "MAX78000 checks"),
        ("IDF_PATH", "ESP32 checks"),
        ("STM32AI_COMMAND", "STM32 synthesis checks"),
    ]:
        if os.environ.get(key):
            _print(CHECK, f"{key} is set")
        else:
            _print(WARN, f"{key} is not set; {label} will fail")
            warnings += 1

    try:
        import ppk2_api  # type: ignore  # noqa: F401
        _print(CHECK, "ppk2_api import works")
    except Exception:
        _print(WARN, "ppk2_api not installed; PPK2 checks may run only in simulation/fail")
        warnings += 1

    _print(CHECK if failures == 0 else FAIL, f"doctor finished with {failures} failure(s), {warnings} warning(s)")
    if failures or (args.strict and warnings):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
