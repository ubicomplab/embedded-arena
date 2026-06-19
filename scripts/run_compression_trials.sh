#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/venv/bin/python}"
REASONING="${REASONING:-high}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-}"
DRY_RUN="${DRY_RUN:-0}"
PARALLEL_JOBS="${PARALLEL_JOBS:-${JOBS:-2}}"
SANDBOX_ROOT="${SANDBOX_ROOT:-${ROOT_DIR}/.data/sandboxes}"
LAUNCH_LOG_DIR="${LAUNCH_LOG_DIR:-${ROOT_DIR}/outputs/_compression_trial_launcher_logs}"
CONFIG_FILTER="${CONFIG_FILTER:-}"
MODEL_FILTER="${MODEL_FILTER:-}"
IGNORE_EXISTING="${IGNORE_EXISTING:-0}"

if [ ! -x "${PYTHON_BIN}" ]; then
  PYTHON_BIN="python"
fi

usage() {
  cat <<'EOF'
Usage:
  scripts/run_compression_trials.sh N

Runs the compression experiment matrix until each experiment/model/reasoning
combination has exactly N completed trials, resuming existing output directories
when possible.

Environment overrides:
  PYTHON_BIN=/path/to/python
  REASONING=high
  OUTPUT_PREFIX=optional-prefix
  DRY_RUN=1
  PARALLEL_JOBS=2
  SANDBOX_ROOT=.data/sandboxes
  LAUNCH_LOG_DIR=outputs/_compression_trial_launcher_logs
  CONFIG_FILTER=comma-separated-config-stems-or-paths
  MODEL_FILTER=comma-separated-provider/model-names
  IGNORE_EXISTING=1
EOF
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

if [ "$#" -ne 1 ]; then
  usage >&2
  exit 2
fi

TARGET_TRIALS="$1"
if ! [[ "${TARGET_TRIALS}" =~ ^[0-9]+$ ]] || [ "${TARGET_TRIALS}" -lt 1 ]; then
  echo "N must be a positive integer, got: ${TARGET_TRIALS}" >&2
  exit 2
fi
if ! [[ "${PARALLEL_JOBS}" =~ ^[0-9]+$ ]] || [ "${PARALLEL_JOBS}" -lt 1 ]; then
  echo "PARALLEL_JOBS must be a positive integer, got: ${PARALLEL_JOBS}" >&2
  exit 2
fi

if [ -f "${ROOT_DIR}/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/.env"
  set +a
fi

CONFIGS=(
  "configs/experiments/compression-no-feedback-max78000.yaml"
  "configs/experiments/compression-documentation-max78000.yaml"
  "configs/experiments/compression-feedback-max78000.yaml"
  "configs/experiments/compression-no-feedback-stm32nucleo.yaml"
  "configs/experiments/compression-documentation-stm32nucleo.yaml"
  "configs/experiments/compression-feedback-stm32nucleo.yaml"
)

MODELS=(
  "openai/gpt-5.4"
  "openai/gpt-5.4-mini"
  "claude/claude-opus-4-7"
  "claude/claude-sonnet-4-6"
  "gemini/gemini-3.1-pro-preview"
  "gemini/gemini-3-flash-preview"
)

safe_name() {
  local value="$1"
  value="${value//\//-}"
  value="${value//:/-}"
  value="${value//[/}"
  value="${value//]/}"
  value="${value// /-}"
  echo "${value}"
}

matches_csv_filter() {
  local value="$1"
  local filter="$2"
  local item
  if [ -z "${filter}" ]; then
    return 0
  fi
  IFS=',' read -r -a items <<< "${filter}"
  for item in "${items[@]}"; do
    item="${item#"${item%%[![:space:]]*}"}"
    item="${item%"${item##*[![:space:]]}"}"
    if [ "${value}" = "${item}" ]; then
      return 0
    fi
  done
  return 1
}

find_existing_run() {
  local config_path="$1"
  local model="$2"
  local reasoning="$3"
  "${PYTHON_BIN}" - "${ROOT_DIR}" "${config_path}" "${model}" "${reasoning}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
config_path = (root / sys.argv[2]).resolve()
model = sys.argv[3]
reasoning = sys.argv[4] or None

matches = []
for summary_path in (root / "outputs").glob("*/summary.json"):
    try:
        summary = json.loads(summary_path.read_text())
    except Exception:
        continue
    try:
        summary_config = Path(summary.get("config_path", "")).resolve()
    except Exception:
        continue
    if summary_config != config_path:
        continue
    if summary.get("llm") != model:
        continue
    if (summary.get("reasoning") or None) != reasoning:
        continue
    trials = len(summary.get("trials") or [])
    mtime = summary_path.stat().st_mtime
    matches.append((trials, mtime, summary_path.parent.name))

if not matches:
    print("0\t")
else:
    # Resume the matching run with the most existing trials, then newest mtime.
    trials, _, output_name = sorted(matches, reverse=True)[0]
    print(f"{trials}\t{output_name}")
PY
}

echo "Running compression matrix to ${TARGET_TRIALS} trial(s) per combo."
echo "Reasoning: ${REASONING}"
echo "Parallel jobs: ${PARALLEL_JOBS}"
[ -z "${CONFIG_FILTER}" ] || echo "Config filter: ${CONFIG_FILTER}"
[ -z "${MODEL_FILTER}" ] || echo "Model filter: ${MODEL_FILTER}"
[ "${IGNORE_EXISTING}" = "0" ] || echo "Ignoring existing runs when scheduling."
echo

if [ "${DRY_RUN}" != "1" ]; then
  mkdir -p "${SANDBOX_ROOT}" "${LAUNCH_LOG_DIR}"
fi

failed_jobs_file="$(mktemp "${TMPDIR:-/tmp}/edgedl-compression-failures.XXXXXX")"
cleanup() {
  local pids
  pids="$(jobs -rp)"
  if [ -n "${pids}" ]; then
    # shellcheck disable=SC2086
    kill ${pids} 2>/dev/null || true
  fi
  rm -f "${failed_jobs_file}"
}
trap cleanup EXIT
trap 'cleanup; exit 130' INT TERM

running_job_count() {
  jobs -rp | wc -l | tr -d ' '
}

wait_for_one_slot() {
  while [ "$(running_job_count)" -ge "${PARALLEL_JOBS}" ]; do
    sleep 5
  done
}

wait_for_all_jobs() {
  while [ "$(running_job_count)" -gt 0 ]; do
    sleep 5
  done
  wait || true
}

for config in "${CONFIGS[@]}"; do
  config_stem="$(basename "${config}" .yaml)"
  if ! matches_csv_filter "${config}" "${CONFIG_FILTER}" && ! matches_csv_filter "${config_stem}" "${CONFIG_FILTER}"; then
    continue
  fi
  for model in "${MODELS[@]}"; do
    if ! matches_csv_filter "${model}" "${MODEL_FILTER}"; then
      continue
    fi
    model_safe="$(safe_name "${model}")"
    reasoning_safe="$(safe_name "${REASONING}")"
    default_output="${config_stem}__${model_safe}__${reasoning_safe}"
    if [ -n "${OUTPUT_PREFIX}" ]; then
      default_output="${OUTPUT_PREFIX}-${default_output}"
    fi

    if [ "${IGNORE_EXISTING}" = "1" ]; then
      existing_info=$'0\t'
    else
      existing_info="$(find_existing_run "${config}" "${model}" "${REASONING}")"
    fi
    existing_trials="${existing_info%%$'\t'*}"
    existing_output="${existing_info#*$'\t'}"
    output_name="${existing_output:-${default_output}}"

    if [ "${existing_trials}" -ge "${TARGET_TRIALS}" ]; then
      echo "==> ${config_stem} | ${model} [${REASONING}]: already ${existing_trials}/${TARGET_TRIALS}, skipping (${output_name})"
      continue
    fi

    missing=$((TARGET_TRIALS - existing_trials))
    sandbox_path="${SANDBOX_ROOT}/${output_name}"
    launcher_log="${LAUNCH_LOG_DIR}/${output_name}.log"
    echo "==> ${config_stem} | ${model} [${REASONING}]: running ${missing} trial(s), output=${output_name}, sandbox=${sandbox_path}"

    use_resume=0
    if [ -n "${existing_output}" ]; then
      use_resume=1
    fi

    cmd=(
      "${PYTHON_BIN}" -m src.run "${config}"
      task.trials "${missing}"
      --llm "${model}"
      --reasoning "${REASONING}"
      --output-name "${output_name}"
      --sandbox-path "${sandbox_path}"
    )
    if [ "${use_resume}" = "1" ]; then
      cmd+=(--resume)
    elif [ "${IGNORE_EXISTING}" = "1" ]; then
      cmd+=(--overwrite)
    fi

    if [ "${DRY_RUN}" = "1" ]; then
      printf 'DRY_RUN:'
      printf ' %q' "${cmd[@]}"
      printf '\n'
    else
      wait_for_one_slot
      (
        set +e
        echo "START $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
        printf 'COMMAND:'
        printf ' %q' "${cmd[@]}"
        printf '\n\n'
        cd "${ROOT_DIR}" && "${cmd[@]}"
        status=$?
        echo
        echo "END $(date -u '+%Y-%m-%dT%H:%M:%SZ') status=${status}"
        if [ "${status}" -ne 0 ]; then
          echo "${output_name}" >> "${failed_jobs_file}"
        fi
        exit "${status}"
      ) > "${launcher_log}" 2>&1 &
      echo "    launched pid=$! log=${launcher_log}"
    fi
    echo
  done
done

if [ "${DRY_RUN}" != "1" ]; then
  wait_for_all_jobs
  if [ -s "${failed_jobs_file}" ]; then
    echo "Compression matrix finished with failed jobs:" >&2
    sed 's/^/  - /' "${failed_jobs_file}" >&2
    echo "See launcher logs in ${LAUNCH_LOG_DIR}" >&2
    exit 1
  fi
fi

echo "Compression matrix complete."
