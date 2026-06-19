# Setup

**Docs:** [Overview](README.md) | [Setup](setup.md) | [Hardware](hardware.md) | [Data/assets](data-assets.md) | [Experiments](experiments.md) | [Results](results.md) | [Adding benchmarks](adding-hardware.md) | [Model providers](model-providers.md) | [Safety](safety.md)

This page gets a new machine from a fresh clone to a runnable benchmark. Install only the target-specific toolchains you need; the smoke test and most code checks do not require all hardware.

## Prerequisites

- macOS or Linux.
- Python 3.10 or newer.
- Docker. On macOS, install [Docker Desktop](https://www.docker.com/products/docker-desktop/). On Linux, install [Docker Engine](https://docs.docker.com/engine/install/) and make sure `docker ps` works without `sudo`.
- Git and a shell with standard Unix tools.

Hardware checks run on the host, not inside Docker. The Docker sandbox is for agent-authored code execution; flashing and measurement need host USB access to devices such as CMSIS-DAP/J-Link, Nordic PPK2, ESP32 serial ports, and the MLX90640 camera bridge.

## Install The Python Package

```bash
git clone https://github.com/ubicomplab/embedded-arena.git
cd embedded-arena
python3 scripts/check_python.py
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install -e '.[providers,dev]'
cp .env.example .env
embedded-arena doctor
```

Optional extras:

```bash
# Hardware measurements that use Nordic PPK2
python -m pip install -e '.[hardware]'

# Hugging Face download helper
python -m pip install -e '.[assets]'
```

The default sandbox root is `~/.cache/embedded-arena/sandboxes/default`. Leave `EMBEDDED_ARENA_CACHE_DIR` and `EMBEDDED_ARENA_SANDBOX_PATH` blank in `.env` unless you intentionally want a custom path outside the repository.

## Provider Keys

Edit `.env` and add keys only for the providers you plan to run:

```bash
OPENAI_API_KEY=...
GOOGLE_API_KEY=...
ANTHROPIC_API_KEY=...
```

Then load `.env` in each shell before running benchmarks:

```bash
set -a; source .env; set +a
```

Local and scripted adapters do not need API keys; see [Model providers](model-providers.md).

## Smoke Test

This command should pass before any board-specific setup:

```bash
CLI_LLM_SCRIPT=examples/cli_smoke_gradient_flow.jsonl \
  embedded-arena run configs/smoke/gradient-flow.yaml \
  --llm cli/scripted \
  --iterations 1 \
  --output-dir outputs/smoke \
  --overwrite
```

## MAX78000 Toolchain

MAX78000 compression and power experiments use Analog Devices/Maxim tooling:

- [Analog Devices MSDK](https://github.com/analogdevicesinc/msdk) and [MSDK user guide](https://analogdevicesinc.github.io/msdk/USERGUIDE/)
- [ai8x-synthesis](https://github.com/analogdevicesinc/ai8x-synthesis)
- [ai8x-training](https://github.com/analogdevicesinc/ai8x-training)
- [Arm GNU Toolchain](https://developer.arm.com/downloads/-/arm-gnu-toolchain-downloads)
- [SEGGER J-Link software](https://www.segger.com/downloads/jlink/) if your debugger uses J-Link

The setup script clones pinned ai8x sources, installs a synthesis virtual environment, clones MSDK, installs or detects `arm-none-eabi-gcc`, and writes paths to `.env`:

```bash
./scripts/setup_max78000.sh
set -a; source .env; set +a
```

By default, full `ai8x-training` Python dependencies are skipped because the synthesis smoke path does not need them and some optional pins are platform-specific. Install them when you need ai8x training/evaluation:

```bash
MAX78000_INSTALL_TRAINING_REQUIREMENTS=1 ./scripts/setup_max78000.sh
```

Verify synthesis without physical hardware:

```bash
CLI_LLM_SCRIPT=examples/cli_smoke_synthesis_max78000.jsonl \
  embedded-arena run configs/smoke/synthesis-max78000.yaml \
  --llm cli/scripted \
  --iterations 1 \
  --output-dir outputs/smoke-max78000 \
  --overwrite
```

For power/energy HIL experiments, also install the PPK2 extra:

```bash
python -m pip install -e '.[hardware]'
```

Then continue with [MAX78000 hardware setup](hardware.md#max78000-power-and-energy).

## ESP32-S3 Toolchain

ESP32 thermal experiments use ESP-IDF for the target TinyLLaMA firmware and PlatformIO for the MLX90640 IR camera bridge:

- [ESP-IDF installation guide](https://docs.espressif.com/projects/esp-idf/en/stable/esp32/get-started/index.html#installation)
- [ESP-IDF GitHub](https://github.com/espressif/esp-idf)
- [PlatformIO Core](https://docs.platformio.org/en/latest/core/installation/index.html)

The setup script clones ESP-IDF under `.data/toolchains/esp-idf`, runs Espressif's installer, detects a serial port when possible, and writes `IDF_PATH`, `IDF_PYTHON_ENV_PATH`, `ESP32_PORT`, and `ESP32_BAUD` to `.env`:

```bash
./scripts/setup_esp32.sh
# or choose the target port explicitly
./scripts/setup_esp32.sh --port /dev/tty.usbmodemXXXX
set -a; source .env; set +a
```

If an ESP-IDF download is interrupted and later submodule recovery fails, rerun:

```bash
./scripts/setup_esp32.sh --force-reinstall
```

This moves the partial install aside and clones a clean ESP-IDF copy.

Do not source `export.sh` in the same shell before running `embedded-arena`; the framework calls ESP-IDF's `idf.py` through `IDF_PATH` and `IDF_PYTHON_ENV_PATH` while keeping your project virtual environment active.

Verify the ESP-IDF install without a connected board:

```bash
IDF_PYTHON_ENV_PATH="$IDF_PYTHON_ENV_PATH" \
  "$IDF_PYTHON_ENV_PATH/bin/python" "$IDF_PATH/tools/idf.py" --version
```

Flash the IR camera bridge once after wiring the MLX90640 sensor:

```bash
cd firmware/ir-camera
pio run -t upload
cd ../..
python scripts/test_ir_camera.py
```

Then continue with [ESP32-S3 thermal hardware setup](hardware.md#esp32-s3-thermal-management).

## STM32N6 Toolchain

STM32N6 compression checks use ST Edge AI / STM32Cube.AI tooling:

- [X-CUBE-AI / STM32Cube.AI](https://www.st.com/en/embedded-software/x-cube-ai.html#get-software)
- [STM32CubeProgrammer](https://www.st.com/en/development-tools/stm32cubeprog.html) for flashing/on-board validation flows
- [STM32CubeIDE](https://www.st.com/en/development-tools/stm32cubeide.html) for the ST-LINK GDB server used by some board validation paths

ST distributes X-CUBE-AI behind a signed-in download gate, so the script does not fetch it for you. Download the zip from ST, then run:

```bash
./scripts/setup_stm32ai.sh /path/to/x-cube-ai-macarm-v10.2.0.zip
set -a; source .env; set +a
```

The script extracts `stedgeai`, writes `STM32AI_COMMAND`, `STM32AI_DIR`, and `STM32_TARGET` to `.env`, and points documentation-enabled experiments at the extracted ST docs. The default target is `NUCLEO-N657X0-Q` / `STM32N657X0H3Q`, with 512-Mbit Octo-SPI Flash on the board and 4.2 MB contiguous SRAM on the MCU. Override `STM32_TARGET`, `STM32_FLASH_BYTES`, `STM32_RAM_BYTES`, or `STM32_MEMORY_HEADROOM_FRACTION` only when intentionally targeting a different board or memory budget.

Verify synthesis:

```bash
CLI_LLM_SCRIPT=examples/cli_smoke_synthesis_stm32n6.jsonl \
  embedded-arena run configs/smoke/synthesis-stm32n6.yaml \
  --llm cli/scripted \
  --iterations 1 \
  --output-dir outputs/smoke-stm32n6 \
  --overwrite
```

## Hugging Face Assets

The STM32 speech-to-IPA benchmark uses gated Hugging Face assets. Log in with an account that has accepted the repository terms, then download snapshots into `.data/huggingface`:

```bash
huggingface-cli login
./scripts/setup_huggingface_assets.sh
set -a; source .env; set +a
```

See [Data/assets](data-assets.md) for the expected local layout.

## COCO Subset Asset

The MAX78000 compression benchmark needs `.data/coco.zip`. Build it from native COCO/Ultralytics sources with:

```bash
./scripts/setup_coco_subset.py
```

Use `--train-count`, `--val-count`, and `--test-count` for a smaller local subset while testing the pipeline.

## Optional Remote Training

Training checks can offload candidate training to a remote GPU host while the agent and sandbox run locally. Configure these `.env` keys:

```bash
REMOTE_TRAIN_HOST=
REMOTE_TRAIN_PASSWORD=
REMOTE_TRAIN_ROOT=
REMOTE_TRAIN_CACHE=
REMOTE_TRAIN_PYTHON=
```

Then prime the remote environment:

```bash
./scripts/setup_remote_training.sh
```

When `REMOTE_TRAIN_HOST` is set, the check uploads only the candidate model, reuses cached datasets on the remote host, selects a visible GPU with `nvidia-smi`, and returns metrics to the local run.

## Validation Helpers

Before sending a PR or handing the repo to another lab, run:

```bash
python scripts/check_docs_links.py
python scripts/check_configs.py
embedded-arena doctor
```
