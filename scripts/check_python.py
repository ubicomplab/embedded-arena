#!/usr/bin/env python3
from __future__ import annotations

import sys

MIN_VERSION = (3, 10)

if sys.version_info < MIN_VERSION:
    raise SystemExit(
        f"EmbeddedArena requires Python >= 3.10; found {sys.version.split()[0]}. "
        "Install a newer Python and recreate .venv."
    )
print(f"Python {sys.version.split()[0]} OK for EmbeddedArena")
