"""Command-line entrypoint for EmbeddedArena."""
from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "run":
        from embedded_arena import run as run_module

        sys.argv = ["embedded-arena run", *argv[1:]]
        run_module.main()
        return
    if argv and argv[0] == "doctor":
        from embedded_arena import doctor as doctor_module

        doctor_module.main(argv[1:])
        return
    if argv and argv[0] in {"-h", "--help"}:
        parser = argparse.ArgumentParser(prog="embedded-arena")
        sub = parser.add_subparsers(dest="command")
        sub.add_parser("run", help="run an EmbeddedArena benchmark config")
        sub.add_parser("doctor", help="check host setup and required assets")
        parser.print_help()
        return
    if not argv:
        print("Usage: embedded-arena {run,doctor} ...")
        return
    raise SystemExit(f"Unknown command: {argv[0]}\nTry: embedded-arena --help")


if __name__ == "__main__":
    main()
