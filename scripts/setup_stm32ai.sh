#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLCHAIN_DIR="${ROOT_DIR}/.data/toolchains/stm32ai"
INSTALL_DIR="${TOOLCHAIN_DIR}/x-cube-ai"
ENV_FILE="${ROOT_DIR}/.env"

usage() {
  cat <<EOF
Usage: $0 /path/to/x-cube-ai-macarm-v10.2.0.zip

Download X-CUBE-AI for macOS/Apple Silicon from:
  https://www.st.com/en/embedded-software/x-cube-ai.html#get-software

The downloaded zip is gated by ST sign-in and is not fetched by this script.
EOF
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

ARCHIVE="${1:-${X_CUBE_AI_ZIP:-}}"
if [ -z "${ARCHIVE}" ]; then
  usage >&2
  exit 2
fi
if [ ! -f "${ARCHIVE}" ]; then
  echo "X-CUBE-AI zip does not exist: ${ARCHIVE}" >&2
  exit 2
fi

mkdir -p "${TOOLCHAIN_DIR}" "${INSTALL_DIR}"

echo "Installing STM32 X-CUBE-AI from ${ARCHIVE}"
rm -rf "${INSTALL_DIR}"
mkdir -p "${INSTALL_DIR}"
unzip -q "${ARCHIVE}" -d "${INSTALL_DIR}"

INNER_ZIP="$(find "${INSTALL_DIR}" -maxdepth 1 -type f -name 'stedgeai-*.zip' | head -n 1)"
if [ -z "${INNER_ZIP}" ]; then
  echo "Could not find stedgeai-*.zip inside ${ARCHIVE}" >&2
  exit 1
fi
unzip -q "${INNER_ZIP}" -d "${INSTALL_DIR}/stedgeai"

STM32AI_FOUND="$(find "${INSTALL_DIR}/stedgeai" -type f \( -name stedgeai -o -name stm32ai \) | head -n 1)"
if [ -z "${STM32AI_FOUND}" ]; then
  echo "Could not find stedgeai/stm32ai executable after extraction." >&2
  exit 1
fi
chmod +x "${STM32AI_FOUND}"
if command -v xattr >/dev/null 2>&1; then
  xattr -dr com.apple.quarantine "${INSTALL_DIR}" 2>/dev/null || true
fi

touch "${ENV_FILE}"
set_env() {
  local key="$1"
  local value="$2"
  local quoted
  quoted="$(python3 -c 'import shlex, sys; print(shlex.quote(sys.argv[1]))' "${value}")"
  if grep -q "^${key}=" "${ENV_FILE}"; then
    python3 - "${ENV_FILE}" "${key}" "${quoted}" <<'PY'
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

ARM_GNU_FOUND="${ARM_GNU_TOOLCHAIN_BIN:-}"
if [ -z "${ARM_GNU_FOUND}" ] && command -v arm-none-eabi-gcc >/dev/null 2>&1; then
  ARM_GNU_FOUND="$(dirname "$(command -v arm-none-eabi-gcc)")"
fi
if [ -z "${ARM_GNU_FOUND}" ]; then
  ARM_GNU_FOUND="$(find "${ROOT_DIR}/.data/toolchains" -path '*/bin/arm-none-eabi-gcc' -type f 2>/dev/null | head -n 1)"
  if [ -n "${ARM_GNU_FOUND}" ]; then
    ARM_GNU_FOUND="$(dirname "${ARM_GNU_FOUND}")"
  fi
fi

set_env "STM32AI_COMMAND" "${STM32AI_FOUND}"
set_env "STM32AI_DIR" "${INSTALL_DIR}/stedgeai"
set_env "STM32_TARGET" "${STM32_TARGET:-NUCLEO-N657X0-Q}"
if [ -n "${ARM_GNU_FOUND}" ]; then
  set_env "ARM_GNU_TOOLCHAIN_BIN" "${ARM_GNU_FOUND}"
fi

cat <<EOF
STM32 AI setup complete.

Updated ${ENV_FILE} with:
  STM32AI_COMMAND=${STM32AI_FOUND}
  STM32AI_DIR=${INSTALL_DIR}/stedgeai
  STM32_TARGET=${STM32_TARGET:-NUCLEO-N657X0-Q}
  ARM_GNU_TOOLCHAIN_BIN=${ARM_GNU_FOUND:-not found; install Arm GNU Toolchain or run scripts/setup_max78000.sh}

Documentation is available at:
  ${INSTALL_DIR}/stedgeai/Documentation

Before running experiments, load the variables:
  set -a; source .env; set +a
EOF
