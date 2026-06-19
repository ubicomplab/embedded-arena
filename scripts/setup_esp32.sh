#!/usr/bin/env bash
# Set up the ESP-IDF toolchain and detect the ESP32 serial port.
# Writes IDF_PATH and ESP32_PORT (plus optionally ESP32_BAUD) to .env.
# Usage: ./scripts/setup_esp32.sh [--idf-version v5.3.2] [--port /dev/tty...]
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ROOT_DIR}/.env"

# ---- configurable defaults -----------------------------------------------
IDF_VERSION="${IDF_VERSION:-v5.3.2}"
IDF_INSTALL_DIR="${IDF_INSTALL_DIR:-${ROOT_DIR}/.data/toolchains/esp-idf}"
ESP32_BAUD="${ESP32_BAUD:-921600}"

# ---- argument parsing --------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --idf-version) IDF_VERSION="$2"; shift 2 ;;
    --idf-dir)     IDF_INSTALL_DIR="$2"; shift 2 ;;
    --port)        MANUAL_PORT="$2"; shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

# ---- helper: write/update a key=value line in .env --------------------------
touch "${ENV_FILE}"
set_env() {
  local key="$1"
  local value="$2"
  if grep -q "^${key}=" "${ENV_FILE}" 2>/dev/null; then
    # Replace in-place (portable sed)
    python3 - "${ENV_FILE}" "${key}" "${value}" <<'PY'
from pathlib import Path
import sys
path, key, value = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
lines = path.read_text().splitlines()
path.write_text("\n".join(
    f"{key}={value}" if line.startswith(f"{key}=") else line
    for line in lines
) + "\n")
PY
  else
    printf '%s=%s\n' "${key}" "${value}" >> "${ENV_FILE}"
  fi
}

# ---- 1. ESP-IDF install / update --------------------------------------------
if [ -d "${IDF_INSTALL_DIR}/.git" ]; then
  echo "ESP-IDF already cloned at ${IDF_INSTALL_DIR}; checking out ${IDF_VERSION}…"
  git -C "${IDF_INSTALL_DIR}" fetch --tags origin
  git -C "${IDF_INSTALL_DIR}" checkout "${IDF_VERSION}"
  git -C "${IDF_INSTALL_DIR}" submodule update --init --recursive
else
  echo "Cloning ESP-IDF ${IDF_VERSION} into ${IDF_INSTALL_DIR}…"
  mkdir -p "$(dirname "${IDF_INSTALL_DIR}")"
  git clone --depth 1 --branch "${IDF_VERSION}" \
      --recurse-submodules \
      https://github.com/espressif/esp-idf.git \
      "${IDF_INSTALL_DIR}"
fi

# ---- 2. Run ESP-IDF install script ------------------------------------------
echo "Running ${IDF_INSTALL_DIR}/install.sh …"
"${IDF_INSTALL_DIR}/install.sh" all

# ---- 3. Detect ESP32 serial port --------------------------------------------
if [ -n "${MANUAL_PORT:-}" ]; then
  ESP32_PORT="${MANUAL_PORT}"
  echo "Using manually specified port: ${ESP32_PORT}"
else
  # Glob common USB-CDC device names on macOS and Linux
  DETECTED=""
  for pat in \
      /dev/tty.usbmodem* \
      /dev/tty.SLAB_USBtoUART* \
      /dev/ttyUSB* \
      /dev/ttyACM*; do
    for candidate in ${pat}; do
      if [ -c "${candidate}" ]; then
        DETECTED="${candidate}"
        break 2
      fi
    done
  done

  if [ -n "${DETECTED}" ]; then
    ESP32_PORT="${DETECTED}"
    echo "Auto-detected ESP32 port: ${ESP32_PORT}"
  else
    echo ""
    echo "WARNING: Could not auto-detect an ESP32 serial port."
    echo "Connect the ESP32 and set ESP32_PORT manually in ${ENV_FILE},"
    echo "or re-run with: ./scripts/setup_esp32.sh --port /dev/tty..."
    ESP32_PORT=""
  fi
fi

# ---- 4. Write .env ----------------------------------------------------------
set_env "IDF_PATH" "${IDF_INSTALL_DIR}"
[ -n "${ESP32_PORT}" ] && set_env "ESP32_PORT" "${ESP32_PORT}"
set_env "ESP32_BAUD"  "${ESP32_BAUD}"

cat <<EOF

ESP-IDF setup complete.

Updated ${ENV_FILE} with:
  IDF_PATH=${IDF_INSTALL_DIR}
  ESP32_PORT=${ESP32_PORT:-"(not set — update manually)"}
  ESP32_BAUD=${ESP32_BAUD}

Before running experiments, load .env so IDF_PATH and ESP32_PORT are set:
  set -a; source .env; set +a

Do not source ${IDF_INSTALL_DIR}/export.sh before running src.run.
That script is for interactive ESP-IDF shell use and can replace your
current Python interpreter with the ESP-IDF one. The framework only needs
IDF_PATH so it can call ${IDF_INSTALL_DIR}/tools/idf.py directly.

If you want to run idf.py manually in a terminal, source export.sh in a
separate shell after activating whatever Python environment you need.

idf.py should now be on your PATH. Verify with:
  idf.py --version
EOF
