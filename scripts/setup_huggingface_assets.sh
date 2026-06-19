#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASSET_DIR="${ROOT_DIR}/.data/huggingface"
MODEL_ID="${HF_REFERENCE_MODEL_ID:-KoelLabs/xlsr-english-01}"
DATASET_ID="${HF_REFERENCE_DATASET_ID:-KoelLabs/SpeechOcean}"
MODEL_DIR="${ASSET_DIR}/models/${MODEL_ID}"
DATASET_DIR="${ASSET_DIR}/datasets/${DATASET_ID}"
ENV_FILE="${ROOT_DIR}/.env"

if [ -z "${PYTHON_BIN:-}" ]; then
  if [ -x "${ROOT_DIR}/.venv/bin/python" ]; then
    PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
  elif [ -x "${ROOT_DIR}/venv/bin/python" ]; then
    PYTHON_BIN="${ROOT_DIR}/venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

mkdir -p "${MODEL_DIR}" "${DATASET_DIR}"

if ! "${PYTHON_BIN}" -c "import huggingface_hub" >/dev/null 2>&1; then
  "${PYTHON_BIN}" -m pip install -U "huggingface_hub>=0.23"
fi

"${PYTHON_BIN}" - "${MODEL_ID}" "${MODEL_DIR}" "${DATASET_ID}" "${DATASET_DIR}" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

from huggingface_hub import snapshot_download

model_id, model_dir, dataset_id, dataset_dir = sys.argv[1:5]

downloads = [
    (model_id, "model", Path(model_dir)),
    (dataset_id, "dataset", Path(dataset_dir)),
]

for repo_id, repo_type, local_dir in downloads:
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {repo_type} {repo_id} -> {local_dir}")
    snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        local_dir=local_dir,
        token=True,
    )
PY

touch "${ENV_FILE}"
set_env() {
  local key="$1"
  local value="$2"
  local quoted
  quoted="$("${PYTHON_BIN}" -c 'import shlex, sys; print(shlex.quote(sys.argv[1]))' "${value}")"
  if grep -q "^${key}=" "${ENV_FILE}"; then
    "${PYTHON_BIN}" - "${ENV_FILE}" "${key}" "${quoted}" <<'PY'
from pathlib import Path
import sys
path = Path(sys.argv[1])
key = sys.argv[2]
value = sys.argv[3]
lines = path.read_text().splitlines()
path.write_text("\n".join(f"{key}={value}" if line.startswith(f"{key}=") else line for line in lines) + "\n")
PY
  else
    printf '%s=%s\n' "${key}" "${quoted}" >> "${ENV_FILE}"
  fi
}

set_env "HF_REFERENCE_MODEL_ID" "${MODEL_ID}"
set_env "HF_REFERENCE_MODEL_DIR" "${MODEL_DIR}"
set_env "HF_REFERENCE_DATASET_ID" "${DATASET_ID}"
set_env "HF_REFERENCE_DATASET_DIR" "${DATASET_DIR}"

cat <<EOF
Hugging Face assets setup complete.

Updated ${ENV_FILE} with:
  HF_REFERENCE_MODEL_ID=${MODEL_ID}
  HF_REFERENCE_MODEL_DIR=${MODEL_DIR}
  HF_REFERENCE_DATASET_ID=${DATASET_ID}
  HF_REFERENCE_DATASET_DIR=${DATASET_DIR}

The downloads use your current Hugging Face token/session. If access fails, run:
  huggingface-cli login
EOF
