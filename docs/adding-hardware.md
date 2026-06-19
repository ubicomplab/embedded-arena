# Adding Hardware And Experiments

**Docs:** [Overview](README.md) | [Setup](setup.md) | [Hardware](hardware.md) | [Data/assets](data-assets.md) | [Experiments](experiments.md) | [Results](results.md) | [Adding benchmarks](adding-hardware.md) | [Model providers](model-providers.md) | [Safety](safety.md)

The main contribution path is adding a new hardware target or a new experiment for an existing target. The goal is for another lab to clone the repo, follow your docs, wire the hardware, and reproduce your checks without private assumptions.

## What A Complete Benchmark Adds

```text
embedded_arena/checks/<your_check>.py       deterministic check wrapper
embedded_arena/hardware/<your_driver>.py    host-side hardware/toolchain driver
firmware/<target>/<workload>/               seed firmware or buildable sample
configs/benchmarks/<task>/<hardware>/...    score/documentation/hil YAML variants
docs/assets/...                             photos/diagrams when helpful
docs/...                                    setup and hardware instructions
```

Use this config path convention:

```text
configs/benchmarks/<task>/<hardware>[/<experiment>]/<variant>.yaml
```

Examples:

- `configs/benchmarks/power/max78000/peak-current/hil.yaml`
- `configs/benchmarks/thermal/esp32/contact/documentation.yaml`
- `configs/benchmarks/compression/stm32n6/score.yaml`

## Check Design

Each check module must expose:

```python
class Input(StrictBaseModel):
    ...

def check(state: RunState, input: Input) -> CheckResult:
    ...
```

Guidelines:

- Prefer deterministic checks over LLM judging. If behavior can be enforced with serial handshakes, checkpoints, trace signatures, source inspection, or protocol validation, implement that in code.
- Keep agent-controlled fields narrow. Put fixed measurement parameters in YAML `params` so the agent cannot change the target voltage, duration, sample rate, or safety limits.
- Fail loudly on missing hardware or toolchains with `ExperimentSetupError`; do not fall back to simulated measurements inside benchmark checks.
- Return concise feedback. Include enough information to guide iteration, but avoid dumping huge files into the agent context.
- Put machine-specific values in `.env` or documented CLI overrides, not committed YAML.

## Firmware And Seed Artifacts

Seed firmware should be buildable before the agent edits it. Include a local README with:

- Provenance and upstream citation.
- Required vendor SDK/toolchain.
- The benchmark workload contract.
- Any safety or recovery notes.
- How the harness builds/flashes/measures it.

If the seed is adapted from third-party code, update `THIRD_PARTY_NOTICES.md`.

## Documentation Requirements

For every new hardware setup, document:

- Bill of materials.
- Wiring diagram or photo.
- Required host toolchains with official links.
- Setup script for every non-committed asset or manual vendor installation steps when redistribution is prohibited.
- Smoke test or bring-up command.
- Expected ports/environment variables.
- Recovery procedure after a bad flash or sleep-state bug.
- What data is collected and where it is written.

Photos from the paper-style setup are welcome in `docs/assets/` when they make wiring clearer.

## Asset Setup Scripts

If your benchmark needs large datasets, model snapshots, vendor archives, calibration corpora, or generated fixtures that are not committed to git, add a setup script under `scripts/`. The script must create the exact paths used by your YAML configs, must not contain absolute paths from your workstation, and should be safe to rerun.

Good examples in the current repo:

- `scripts/setup_coco_subset.py` builds `.data/coco.zip` from native COCO/Ultralytics sources.
- `scripts/setup_huggingface_assets.sh` downloads `KoelLabs/xlsr-english-01` and `KoelLabs/SpeechOcean`, which natively live on Hugging Face.
- `scripts/setup_stm32ai.sh` installs from a user-downloaded ST vendor zip whose license prevents redistribution.

Do not rehost third-party assets solely to simplify setup. Prefer reproducible setup scripts that fetch from the native source or clearly instruct the user where to download manually.

## Baseline Runs

New benchmark PRs should include runs on at least:

1. The latest generally available OpenAI model.
2. The latest generally available Google Gemini model.
3. One additional model of your choice.

Use:

```bash
scripts/run_required_baselines.sh configs/benchmarks/<task>/<hardware>[/<experiment>]/<variant>.yaml
```

Update the script or document the exact model list if the current defaults are stale. Include exact model IDs, dates, config paths, iteration counts, and hardware revisions in the PR.

## Pull Request Checklist

- `python scripts/check_docs_links.py`
- `python scripts/check_configs.py`
- `embedded-arena doctor`
- No secrets in `.env`, logs, screenshots, or docs.
- No large datasets, vendor zips, or generated outputs committed.
- All new docs are linked from [docs/README.md](README.md).
