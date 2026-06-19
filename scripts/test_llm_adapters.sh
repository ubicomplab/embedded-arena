#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${ROOT_DIR}/configs/experiments/smoke-gradient-flow.yaml"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/venv/bin/python}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-llm-adapter-smoke}"
REASONING="${REASONING:-high}"

if [ ! -x "${PYTHON_BIN}" ]; then
  PYTHON_BIN="python"
fi

if [ -f "${ROOT_DIR}/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/.env"
  set +a
fi

MODELS=(
  "openai/gpt-5.4"
  "claude/claude-opus-4-7"
  "gemini/gemini-3.1-pro-preview"
  "ollama/gemma4:e2b"
)

validate_summary() {
  local summary_path="$1"
  "${PYTHON_BIN}" - "${summary_path}" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text())
if summary.get("status"):
    raise SystemExit(f"run ended with status={summary.get('status')}: {summary.get('llm_error') or summary.get('setup_error')}")
try:
    check = summary["trials"][0]["iterations"][0]["checks"]["gradient_flow.py"]
except (KeyError, IndexError) as exc:
    raise SystemExit(f"summary missing gradient_flow.py result: {exc}")
if not check.get("success"):
    raise SystemExit(f"gradient_flow.py failed: {check.get('feedback')}")
print(f"PASS tokens={summary.get('tokens', {}).get('total', 0)} tool_calls={summary.get('tool_calls', 0)}")
PY
}

echo "Running LLM adapter smoke tests with ${CONFIG}"
echo

for model in "${MODELS[@]}"; do
  safe_name="${model//\//-}"
  safe_name="${safe_name//:/-}"
  output_name="${OUTPUT_PREFIX}-${safe_name}"
  output_dir="${ROOT_DIR}/outputs/${output_name}"

  echo "==> ${model}"
  rm -rf "${output_dir}"
  (
    cd "${ROOT_DIR}"
    "${PYTHON_BIN}" -m src.run "${CONFIG}" \
      --llm "${model}" \
      --reasoning "${REASONING}" \
      --output-name "${output_name}"
  )
  validate_summary "${output_dir}/summary.json"
  echo
done

echo "All LLM adapter smoke tests passed."
