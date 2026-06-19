#!/usr/bin/env python3
"""Create the COCO subset archive used by EmbeddedArena MAX78000 compression.

The original research notebook sampled images from a local YOLO-format COCO
copy. This script makes the process reproducible from public upstream assets:

1. Download Ultralytics' COCO 2017 segmentation label archive.
2. Sample labeled train/val examples with a deterministic seed.
3. Download only the selected images from images.cocodataset.org.
4. Write .data/coco_build/coco/{images,labels,...} and .data/coco.zip.

The resulting zip contains a top-level coco/ directory, which is what
embedded_arena/checks/train.py expects.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import random
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

LABELS_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels-segments.zip"
IMAGE_URL = "http://images.cocodataset.org/{split}/{image_name}"
COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]


def log(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=".data/coco.zip", help="Output zip path.")
    parser.add_argument("--work-dir", default=".data/coco_build", help="Temporary build/cache directory.")
    parser.add_argument("--train-count", type=int, default=8000, help="Training images to include.")
    parser.add_argument("--val-count", type=int, default=2000, help="Validation images to include, sampled from train2017 labels.")
    parser.add_argument("--test-count", type=int, default=2000, help="Test images to include, sampled from val2017 labels.")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic sampling seed.")
    parser.add_argument("--workers", type=int, default=16, help="Parallel image downloads.")
    parser.add_argument("--labels-url", default=LABELS_URL, help="YOLO segmentation labels archive URL.")
    parser.add_argument("--force", action="store_true", help="Recreate output even if it exists.")
    return parser.parse_args()


def download(url: str, dst: Path, retries: int = 4) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        return
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as response, tmp.open("wb") as handle:
                shutil.copyfileobj(response, handle)
            tmp.replace(dst)
            return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            tmp.unlink(missing_ok=True)
            time.sleep(min(2 ** attempt, 10))
    raise RuntimeError(f"failed to download {url}: {last_error}")


def ensure_labels(labels_url: str, work_dir: Path) -> Path:
    archive = work_dir / "downloads" / "coco2017labels-segments.zip"
    extract_dir = work_dir / "labels_source"
    download(labels_url, archive)
    expected = extract_dir / "coco" / "labels" / "train2017"
    if not expected.exists():
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        extract_dir.mkdir(parents=True)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(extract_dir)
    labels_root = extract_dir / "coco" / "labels"
    if not labels_root.exists():
        raise RuntimeError(f"labels archive did not contain coco/labels: {archive}")
    return labels_root


def sample_labels(labels_root: Path, split: str, count: int, rng: random.Random) -> list[Path]:
    labels = sorted((labels_root / split).glob("*.txt"))
    labels = [p for p in labels if p.read_text(errors="ignore").strip()]
    if count > len(labels):
        raise RuntimeError(f"requested {count} {split} labels but only found {len(labels)}")
    return rng.sample(labels, count)


def copy_label(label: Path, dst_root: Path, split: str) -> str:
    dst = dst_root / "labels" / split / label.name
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(label, dst)
    return f"images/{split}/{label.with_suffix('.jpg').name}"


def fetch_image(item: tuple[str, str, Path]) -> tuple[str, bool, str]:
    split, image_name, dst = item
    url = IMAGE_URL.format(split=split, image_name=image_name)
    try:
        download(url, dst)
        return image_name, True, ""
    except Exception as exc:  # noqa: BLE001 - collect all download failures for a clear summary
        return image_name, False, str(exc)


def write_yaml(coco_dir: Path) -> None:
    names = "\n".join(f"  {i}: {name}" for i, name in enumerate(COCO_NAMES))
    (coco_dir / "data.yaml").write_text(
        "path: .\n"
        "train: train.txt\n"
        "val: val.txt\n"
        "test: test.txt\n"
        "names:\n"
        f"{names}\n",
        encoding="utf-8",
    )


def zip_dir(source: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.unlink(missing_ok=True)
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(source.parent))
    tmp.replace(output)


def main() -> int:
    args = parse_args()
    root = Path.cwd()
    output = (root / args.output).resolve()
    work_dir = (root / args.work_dir).resolve()
    coco_dir = work_dir / "coco"

    if output.exists() and not args.force:
        log(f"{output} already exists; use --force to recreate it")
        return 0

    rng = random.Random(args.seed)
    log(f"Preparing COCO subset in {work_dir}")
    labels_root = ensure_labels(args.labels_url, work_dir)
    log("Sampling labeled examples")
    train_labels = sample_labels(labels_root, "train2017", args.train_count + args.val_count, rng)
    test_labels = sample_labels(labels_root, "val2017", args.test_count, rng)

    if coco_dir.exists():
        shutil.rmtree(coco_dir)
    for split in ["train2017", "val2017"]:
        (coco_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (coco_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    train_lines: list[str] = []
    val_lines: list[str] = []
    test_lines: list[str] = []
    jobs: list[tuple[str, str, Path]] = []

    for label in train_labels[: args.val_count]:
        rel = copy_label(label, coco_dir, "train2017")
        val_lines.append(rel)
        jobs.append(("train2017", label.with_suffix(".jpg").name, coco_dir / rel))
    for label in train_labels[args.val_count :]:
        rel = copy_label(label, coco_dir, "train2017")
        train_lines.append(rel)
        jobs.append(("train2017", label.with_suffix(".jpg").name, coco_dir / rel))
    for label in test_labels:
        rel = copy_label(label, coco_dir, "val2017")
        test_lines.append(rel)
        jobs.append(("val2017", label.with_suffix(".jpg").name, coco_dir / rel))

    log(f"Downloading {len(jobs)} selected COCO images...")
    failures: list[tuple[str, str]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for image_name, ok, detail in pool.map(fetch_image, jobs):
            if not ok:
                failures.append((image_name, detail))
    if failures:
        for image_name, detail in failures[:20]:
            print(f"ERROR {image_name}: {detail}", file=sys.stderr)
        raise RuntimeError(f"failed to download {len(failures)} image(s)")

    (coco_dir / "train.txt").write_text("\n".join(train_lines) + "\n", encoding="utf-8")
    (coco_dir / "val.txt").write_text("\n".join(val_lines) + "\n", encoding="utf-8")
    (coco_dir / "test.txt").write_text("\n".join(test_lines) + "\n", encoding="utf-8")
    write_yaml(coco_dir)
    log(f"Packaging {output}")
    zip_dir(coco_dir, output)

    log(f"Wrote {output}")
    log(f"  train={len(train_lines)} val={len(val_lines)} test={len(test_lines)} seed={args.seed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
