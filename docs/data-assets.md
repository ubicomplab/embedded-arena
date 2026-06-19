# Data And Toolchain Assets

**Docs:** [Overview](README.md) | [Setup](setup.md) | [Hardware](hardware.md) | [Data/assets](data-assets.md) | [Experiments](experiments.md) | [Results](results.md) | [Adding benchmarks](adding-hardware.md) | [Model providers](model-providers.md) | [Safety](safety.md)

Large datasets, model checkpoints, vendor toolchains, and run outputs are intentionally not committed. The repository uses `.data/` as the local cache root and keeps only small curated documentation under `data/documentation/`.

## Local Layout

```text
.data/
  huggingface/
    models/KoelLabs/xlsr-english-01/
    datasets/KoelLabs/SpeechOcean/
  toolchains/
    max78000/
    esp-idf/
    stm32ai/
outputs/
  <run-name>/
```

`scripts/check_configs.py` reports missing large assets as warnings until the relevant benchmark is run. That is expected for a fresh clone.

## Hugging Face Assets

The STM32N6 speech-to-IPA benchmark uses:

- [KoelLabs/xlsr-english-01](https://huggingface.co/KoelLabs/xlsr-english-01)
- `KoelLabs/SpeechOcean` dataset snapshot

If a repository is gated, accept its terms in the browser first, then run:

```bash
python -m pip install -e '.[assets]'
huggingface-cli login
./scripts/setup_huggingface_assets.sh
set -a; source .env; set +a
```

The script writes these `.env` keys:

```bash
HF_REFERENCE_MODEL_ID=
HF_REFERENCE_MODEL_DIR=
HF_REFERENCE_DATASET_ID=
HF_REFERENCE_DATASET_DIR=
```

The benchmark configs copy the snapshots into each sandbox as `reference_model` and `reference_dataset`. Do not add downloaded snapshots to git.

## COCO And YOLO Assets

The MAX78000 compression benchmark expects a YOLO-style seed and COCO-derived data assets when running the full paper task. Keep large archives and checkpoints under `.data/` or another local cache and reference them through config environment files or `.env`. For public PRs, document where the asset comes from, its license, and a checksum or Hugging Face revision when possible.

## Vendor Toolchains

Toolchain setup scripts place vendor code under `.data/toolchains/`:

- `scripts/setup_max78000.sh` -> `.data/toolchains/max78000/`
- `scripts/setup_esp32.sh` -> `.data/toolchains/esp-idf/`
- `scripts/setup_stm32ai.sh` -> `.data/toolchains/stm32ai/`

These directories are local installation caches, not benchmark source. They should remain untracked.

## Adding New Large Assets

Prefer this order:

1. Use a small synthetic smoke fixture committed to the repo.
2. Use a public Hugging Face dataset/model with a pinned revision.
3. Use a gated Hugging Face repository when licensing or human-subject constraints require acceptance terms.
4. Use manual vendor downloads only when redistribution is prohibited.

Every new benchmark should include a short asset table in its docs: asset name, source URL, license/terms, expected local path, and setup command.
