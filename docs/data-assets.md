# Data And Toolchain Assets

**Docs:** [Overview](README.md) | [Setup](setup.md) | [Hardware](hardware.md) | [Data/assets](data-assets.md) | [Experiments](experiments.md) | [Results](results.md) | [Adding benchmarks](adding-hardware.md) | [Model providers](model-providers.md) | [Safety](safety.md)

Large datasets, model checkpoints, vendor toolchains, and run outputs are intentionally not committed. The repository uses `.data/` as the local cache root and keeps only small curated documentation under `data/documentation/`.

## Local Layout

```text
.data/
  coco.zip
  coco_build/
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

`scripts/check_configs.py` reports missing large assets as warnings until the relevant benchmark setup scripts have been run. That is expected for a fresh clone.

## Hugging Face Assets

The STM32N6 speech-to-IPA benchmark uses assets that natively live on Hugging Face:

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

The MAX78000 compression benchmark expects `data.zip` in the agent sandbox, sourced locally from `.data/coco.zip`. Build that archive from public COCO assets with:

```bash
./scripts/setup_coco_subset.py
```

The script downloads Ultralytics' COCO 2017 segmentation labels, samples labeled examples deterministically, downloads only the selected images from `images.cocodataset.org`, writes `.data/coco_build/coco/`, and packages `.data/coco.zip`. By default it recreates the subset used by the benchmark scale: 8,000 training images, 2,000 validation images, and 2,000 test images. For a smaller local smoke asset, override the counts:

```bash
./scripts/setup_coco_subset.py --train-count 256 --val-count 64 --test-count 64 --output .data/coco-small.zip
```

The YOLO reference checkpoint used by the MAX78000 compression configs is committed at `data/models/yolo11n-seg.pt` because it is small enough for the repository. Do not commit generated COCO archives.

## Vendor Toolchains

Toolchain setup scripts place vendor code under `.data/toolchains/`:

- `scripts/setup_max78000.sh` -> `.data/toolchains/max78000/`
- `scripts/setup_esp32.sh` -> `.data/toolchains/esp-idf/`
- `scripts/setup_stm32ai.sh` -> `.data/toolchains/stm32ai/`

These directories are local installation caches, not benchmark source. They should remain untracked.

## Adding New Large Assets

Every new benchmark that needs non-committed assets must include a setup script under `scripts/`. The script should create the exact local paths referenced by the benchmark config, avoid absolute machine-specific paths, be idempotent, and print clear instructions when an upstream asset requires manual license or terms acceptance.

Use this order of preference:

1. Commit a small synthetic smoke fixture when it is tiny and license-safe.
2. Add a setup script that downloads or derives the asset from its native upstream source.
3. Use a native Hugging Face repository only when the dataset/model already lives there, as with `KoelLabs/xlsr-english-01` and `KoelLabs/SpeechOcean`; `scripts/setup_huggingface_assets.sh` is the current example.
4. Use a script that installs from a manually downloaded vendor archive when redistribution is prohibited, as with `scripts/setup_stm32ai.sh`.

Do not rehost assets solely for EmbeddedArena convenience. Every new benchmark should include a short asset table in its docs: asset name, native source URL, license/terms, expected local path, setup command, and whether network or manual acceptance is required.
