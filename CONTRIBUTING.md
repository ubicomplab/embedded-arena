# Contributing

Thank you for helping make EmbeddedArena useful beyond our lab.

Start with [docs/README.md](docs/README.md), then read [Adding Hardware and Experiments](docs/adding-hardware.md), [Hardware](docs/hardware.md), and [Safety](docs/safety.md).

## Main Contribution Path

The primary contribution path is a new hardware target or a new experiment on an existing target. A complete benchmark contribution should include:

- Configs under `configs/benchmarks/<task>/<hardware>[/<experiment>]/<variant>.yaml`.
- Deterministic checks under `embedded_arena/checks/`.
- Host-side drivers or toolchain wrappers under `embedded_arena/hardware/` when needed.
- Seed firmware or model artifacts under `firmware/` or a documented external asset source.
- Setup and hardware docs, including photos/diagrams when helpful.
- Baseline runs on the latest generally available OpenAI model, the latest generally available Google Gemini model, and one additional model of your choice.

Use `scripts/run_required_baselines.sh CONFIG` as a starting point, and update/document the exact model IDs if the defaults are stale.

## Before Opening A PR

```bash
python scripts/check_docs_links.py
python scripts/check_configs.py
embedded-arena doctor
```

Do not commit `.env`, large datasets, vendor toolchains, generated outputs, or logs containing secrets/local paths.
