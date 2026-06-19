#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
HTML_SRC_RE = re.compile(r"""(?:href|src)=["']([^"']+)["']""")
SKIP_DIRS = {".git", ".venv", ".data", "outputs", "__pycache__"}


def iter_markdown() -> list[Path]:
    files = []
    for path in ROOT.rglob("*.md"):
        rel = path.relative_to(ROOT)
        if rel.parts and rel.parts[0] in SKIP_DIRS:
            continue
        if rel.parts[:2] == ("data", "documentation"):
            continue
        files.append(path)
    return sorted(files)


def local_target_exists(source: Path, raw: str) -> bool:
    target = raw.split("#", 1)[0].strip()
    if not target or target.startswith("mailto:"):
        return True
    parsed = urlparse(target)
    if parsed.scheme or target.startswith("//"):
        return True
    path = (source.parent / target).resolve()
    try:
        path.relative_to(ROOT)
    except ValueError:
        return False
    return path.exists()


def main() -> None:
    errors = []
    files = iter_markdown()
    for path in files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        candidates = [m.group(1) for m in LINK_RE.finditer(text)]
        candidates += [m.group(1) for m in HTML_SRC_RE.finditer(text)]
        for target in candidates:
            if not local_target_exists(path, target):
                errors.append(f"{path.relative_to(ROOT)}: broken link {target!r}")
    for error in errors:
        print(error, file=sys.stderr)
    print(f"Checked {len(files)} Markdown file(s); {len(errors)} error(s).")
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
