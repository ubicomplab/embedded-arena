#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ROOT_DIR}/.env"

if [ -f "${ENV_FILE}" ]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

REMOTE_TRAIN_HOST="${REMOTE_TRAIN_HOST:-${1:-}}"
REMOTE_TRAIN_ROOT="${REMOTE_TRAIN_ROOT:-edgedl-runs}"
REMOTE_TRAIN_CACHE="${REMOTE_TRAIN_CACHE:-edgedl-cache}"
REMOTE_TRAIN_PYTHON="${REMOTE_TRAIN_PYTHON:-${REMOTE_TRAIN_CACHE}/venv/bin/python}"

if [ -z "${REMOTE_TRAIN_HOST}" ]; then
  echo "REMOTE_TRAIN_HOST is not set. Add it to .env or pass it as the first argument." >&2
  exit 1
fi

dataset_key() {
  python3 - "$1" <<'PY'
import hashlib
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
if path.is_file():
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    print(f"{path.stem}-{path.stat().st_size}-{digest.hexdigest()[:12]}")
else:
    print(path.name)
PY
}

remote_file_matches() {
  local source="$1"
  local destination="$2"
  local expected
  expected="$(python3 - "$source" <<'PY'
import hashlib
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
digest = hashlib.sha256()
with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
print(path.stat().st_size, digest.hexdigest())
PY
)"
  local actual
  actual="$(ssh "${REMOTE_TRAIN_HOST}" "if [ ! -f $(printf '%q' "${destination}") ]; then echo missing; else python3 - $(printf '%q' "${destination}") <<'PY'
import hashlib
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
digest = hashlib.sha256()
with path.open('rb') as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b''):
        digest.update(chunk)
print(path.stat().st_size, digest.hexdigest())
PY
fi" | tail -n 1)"
  [ "${actual}" = "${expected}" ]
}

upload_file_atomic() {
  local source="$1"
  local destination="$2"
  local temp_destination="${destination}.tmp-$$"
  ssh "${REMOTE_TRAIN_HOST}" "mkdir -p $(printf '%q' "$(dirname "${destination}")")"
  scp "${source}" "${REMOTE_TRAIN_HOST}:${temp_destination}"
  ssh "${REMOTE_TRAIN_HOST}" "mv $(printf '%q' "${temp_destination}") $(printf '%q' "${destination}")"
}

upload_dir_atomic() {
  local source="$1"
  local destination="$2"
  local temp_destination="${destination}.tmp-$$"
  ssh "${REMOTE_TRAIN_HOST}" "mkdir -p $(printf '%q' "$(dirname "${destination}")") && rm -rf $(printf '%q' "${temp_destination}")"
  scp -r "${source}" "${REMOTE_TRAIN_HOST}:${temp_destination}"
  ssh "${REMOTE_TRAIN_HOST}" "rm -rf $(printf '%q' "${destination}") && mv $(printf '%q' "${temp_destination}") $(printf '%q' "${destination}")"
}

directory_matches() {
  local source="$1"
  local destination="$2"
  local expected
  expected="$(python3 - "$source" <<'PY'
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
files = [path for path in root.rglob("*") if path.is_file()]
print(len(files), sum(path.stat().st_size for path in files))
PY
)"
  local actual
  actual="$(ssh "${REMOTE_TRAIN_HOST}" "if [ ! -d $(printf '%q' "${destination}") ]; then echo missing; else python3 - $(printf '%q' "${destination}") <<'PY'
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
files = [path for path in root.rglob('*') if path.is_file()]
print(len(files), sum(path.stat().st_size for path in files))
PY
fi" | tail -n 1)"
  [ "${actual}" = "${expected}" ]
}

ssh "${REMOTE_TRAIN_HOST}" "mkdir -p $(printf '%q' "${REMOTE_TRAIN_ROOT}") $(printf '%q' "${REMOTE_TRAIN_CACHE}")"
ssh "${REMOTE_TRAIN_HOST}" "\
if [ ! -x $(printf '%q' "${REMOTE_TRAIN_PYTHON}") ]; then \
  python3 -m venv $(printf '%q' "$(dirname "$(dirname "${REMOTE_TRAIN_PYTHON}")")") && \
  $(printf '%q' "${REMOTE_TRAIN_PYTHON}") -m pip install -U pip wheel; \
fi && \
$(printf '%q' "${REMOTE_TRAIN_PYTHON}") - <<'PY'
import importlib.util
import subprocess
import sys

def import_ok(name):
    try:
        __import__(name)
        return True
    except Exception:
        return False

if not import_ok("torch") or not import_ok("torchvision"):
    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "--index-url",
        "https://download.pytorch.org/whl/cu128",
        "--force-reinstall",
        "torch",
        "torchvision",
    ])

missing = [
    package
    for package, module in [
        ("numpy", "numpy"),
        ("pillow", "PIL"),
        ("pyarrow", "pyarrow"),
        ("pyyaml", "yaml"),
        ("ultralytics", "ultralytics"),
    ]
    if importlib.util.find_spec(module) is None
]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
PY"

if [ -f "${ROOT_DIR}/.data/coco.zip" ]; then
  coco_key="$(dataset_key "${ROOT_DIR}/.data/coco.zip")"
  coco_remote="${REMOTE_TRAIN_CACHE%/}/datasets/${coco_key}.zip"
  if ! remote_file_matches "${ROOT_DIR}/.data/coco.zip" "${coco_remote}"; then
    upload_file_atomic "${ROOT_DIR}/.data/coco.zip" "${coco_remote}"
  fi
fi

if [ -d "${ROOT_DIR}/.data/huggingface/datasets/KoelLabs/SpeechOcean" ]; then
  speechocean_remote="${REMOTE_TRAIN_CACHE%/}/datasets/reference_dataset"
  if ! directory_matches "${ROOT_DIR}/.data/huggingface/datasets/KoelLabs/SpeechOcean" "${speechocean_remote}"; then
    upload_dir_atomic "${ROOT_DIR}/.data/huggingface/datasets/KoelLabs/SpeechOcean" "${speechocean_remote}"
  fi
fi

cat <<EOF
Remote training setup complete.
EOF
