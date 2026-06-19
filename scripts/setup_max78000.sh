#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLCHAIN_DIR="${ROOT_DIR}/.data/toolchains/max78000"
SYNTH_DIR="${TOOLCHAIN_DIR}/ai8x-synthesis"
TRAIN_DIR="${TOOLCHAIN_DIR}/ai8x-training"
SYNTH_VENV="${TOOLCHAIN_DIR}/venvs/ai8x-synthesis"
TRAIN_VENV="${TOOLCHAIN_DIR}/venvs/ai8x-training"
MSDK_DIR="${SYNTH_DIR}/sdk"
DOWNLOAD_DIR="${TOOLCHAIN_DIR}/downloads"
ARM_GNU_VERSION="${ARM_GNU_VERSION:-15.2.rel1}"
ARM_GNU_ROOT="${TOOLCHAIN_DIR}/arm-gnu-toolchain"

AI8X_SYNTHESIS_COMMIT="${AI8X_SYNTHESIS_COMMIT:-1411cb1358adae90bd159c42a6be3e605a8db432}"
AI8X_TRAINING_COMMIT="${AI8X_TRAINING_COMMIT:-1030e842c285cab182a5994e1340103d3bb247be}"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"

mkdir -p "${TOOLCHAIN_DIR}/venvs" "${DOWNLOAD_DIR}" "${ARM_GNU_ROOT}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

clone_or_update() {
  local url="$1"
  local dir="$2"
  local commit="$3"
  if [ ! -d "${dir}/.git" ]; then
    git clone --recursive "${url}" "${dir}"
  fi
  git -C "${dir}" fetch --tags origin
  git -C "${dir}" checkout "${commit}"
  git -C "${dir}" submodule update --init --recursive
}

create_venv() {
  local dir="$1"
  local venv="$2"
  local install_requirements="${3:-1}"
  if [ ! -x "${venv}/bin/python" ]; then
    "${PYTHON_BIN}" -m venv "${venv}"
  fi
  "${venv}/bin/python" -m pip install -U pip wheel setuptools
  if [ "${install_requirements}" = "1" ]; then
    (
      cd "${dir}"
      "${venv}/bin/python" -m pip install -r requirements.txt
    )
  fi
}

arm_gnu_host() {
  local os_name
  local arch_name
  os_name="$(uname -s)"
  arch_name="$(uname -m)"
  case "${os_name}:${arch_name}" in
    Darwin:arm64) echo "darwin-arm64" ;;
    Darwin:x86_64) echo "darwin-x86_64" ;;
    Linux:x86_64) echo "x86_64" ;;
    Linux:aarch64|Linux:arm64) echo "aarch64" ;;
    *)
      echo "Unsupported host for automatic Arm GNU Toolchain install: ${os_name}/${arch_name}" >&2
      return 1
      ;;
  esac
}

install_arm_gnu_toolchain() {
  local host
  local package
  local archive
  local url
  local install_dir
  host="$(arm_gnu_host)"
  package="arm-gnu-toolchain-${ARM_GNU_VERSION}-${host}-arm-none-eabi"
  archive="${DOWNLOAD_DIR}/${package}.tar.xz"
  install_dir="${ARM_GNU_ROOT}/${package}"

  if command -v arm-none-eabi-gcc >/dev/null 2>&1; then
    ARM_GNU_BIN="$(dirname "$(command -v arm-none-eabi-gcc)")"
    return
  fi
  if [ ! -x "${install_dir}/bin/arm-none-eabi-gcc" ]; then
    url="https://developer.arm.com/-/media/Files/downloads/gnu/${ARM_GNU_VERSION}/binrel/${package}.tar.xz"
    if [ ! -f "${archive}" ]; then
      curl -L --fail --output "${archive}" "${url}"
    fi
    tar -xJf "${archive}" -C "${ARM_GNU_ROOT}"
  fi
  ARM_GNU_BIN="${install_dir}/bin"
}

clone_or_update "https://github.com/analogdevicesinc/ai8x-synthesis.git" "${SYNTH_DIR}" "${AI8X_SYNTHESIS_COMMIT}"
clone_or_update "https://github.com/analogdevicesinc/ai8x-training.git" "${TRAIN_DIR}" "${AI8X_TRAINING_COMMIT}"
create_venv "${SYNTH_DIR}" "${SYNTH_VENV}" 1
if [ "${MAX78000_INSTALL_TRAINING_REQUIREMENTS:-0}" = "1" ]; then
  create_venv "${TRAIN_DIR}" "${TRAIN_VENV}" 1
else
  create_venv "${TRAIN_DIR}" "${TRAIN_VENV}" 0
fi

if [ ! -d "${MSDK_DIR}/.git" ]; then
  rm -rf "${MSDK_DIR}"
  git clone --recursive https://github.com/analogdevicesinc/msdk.git "${MSDK_DIR}"
else
  git -C "${MSDK_DIR}" pull --ff-only
  git -C "${MSDK_DIR}" submodule update --init --recursive
fi
install_arm_gnu_toolchain

ENV_FILE="${ROOT_DIR}/.env"
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

set_env "AI8X_SYNTHESIS_DIR" "${SYNTH_DIR}"
set_env "AI8X_TRAINING_DIR" "${TRAIN_DIR}"
set_env "AI8X_PYTHON" "${SYNTH_VENV}/bin/python"
set_env "AI8X_TEST_DIR" "sdk/Examples/MAX78000/CNN"
set_env "MAXIM_PATH" "${MSDK_DIR}"
set_env "ARM_GNU_TOOLCHAIN_BIN" "${ARM_GNU_BIN}"
set_env "PATH" "${ARM_GNU_BIN}:${PATH}"
set_env "PYTHONPATH" "${TRAIN_DIR}:${TRAIN_DIR}/distiller"

cat <<EOF
MAX78000 setup complete.

Updated ${ENV_FILE} with:
  AI8X_SYNTHESIS_DIR=${SYNTH_DIR}
  AI8X_TRAINING_DIR=${TRAIN_DIR}
  AI8X_PYTHON=${SYNTH_VENV}/bin/python
  AI8X_TEST_DIR=sdk/Examples/MAX78000/CNN
  MAXIM_PATH=${MSDK_DIR}
  ARM_GNU_TOOLCHAIN_BIN=${ARM_GNU_BIN}
  PATH=${ARM_GNU_BIN}:...
  PYTHONPATH=${TRAIN_DIR}:${TRAIN_DIR}/distiller

Before running experiments, load the variables:
  set -a; source .env; set +a

Note: Full ai8x-training dependencies are optional for synthesis smoke tests and are skipped by default.
To install them too, rerun with:
  MAX78000_INSTALL_TRAINING_REQUIREMENTS=1 ./scripts/setup_max78000.sh
EOF
