#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "embedded_arena"))

from run import load_config  # noqa: E402


def main() -> None:
    errors = []
    warnings = 0
    count = 0
    for path in sorted((ROOT / "configs").rglob("*.yaml")):
        if "environments" in path.relative_to(ROOT).parts:
            continue
        try:
            config = load_config(path)
            count += 1
        except Exception as exc:
            errors.append(f"{path.relative_to(ROOT)}: {exc}")
            continue
        for item in config.environment.files:
            src = Path(item.src)
            if not src.is_absolute():
                candidate = (path.parent / src).resolve()
                if not candidate.exists():
                    candidate = (ROOT / item.src).resolve()
                src = candidate
            if not src.exists():
                print(f"WARNING: {path}: missing asset {item.src}")
                warnings += 1
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    print(f"Validated {count} config(s): {len(errors)} error(s), {warnings} missing-asset warning(s).")
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
