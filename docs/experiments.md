# Experiments

**Docs:** [Overview](README.md) | [Setup](setup.md) | [Hardware](hardware.md) | [Data/assets](data-assets.md) | [Experiments](experiments.md) | [Results](results.md) | [Adding benchmarks](adding-hardware.md) | [Model providers](model-providers.md) | [Safety](safety.md)

Experiment configs are ordinary YAML files. The paper-aligned configs live under:

```text
configs/benchmarks/<task>/<hardware>[/<experiment>]/<variant>.yaml
```

`<variant>` is one of:

- `score`: minimal feedback, usually pass/fail plus scalar score.
- `documentation`: score feedback plus curated documentation in the sandbox.
- `hil`: detailed hardware-in-the-loop feedback after each iteration.

## Smoke Test

Run this first on any new machine:

```bash
CLI_LLM_SCRIPT=examples/cli_smoke_gradient_flow.jsonl \
  embedded-arena run configs/smoke/gradient-flow.yaml \
  --llm cli/scripted \
  --iterations 1 \
  --output-dir outputs/smoke \
  --overwrite
```

## CLI Pattern

```bash
embedded-arena run CONFIG.yaml \
  --llm provider/model \
  --reasoning high \
  --iterations 10 \
  --trials 1 \
  --output-dir outputs/my-run \
  --overwrite
```

Useful flags:

- `--resume`: append additional trials to an existing output directory.
- `--snapshot-sandbox`: copy sandbox source files after each iteration for forensic debugging. This can use a lot of disk.
- `task.trials 2`: override config values using dotted keys after the config path.

Example override:

```bash
embedded-arena run configs/benchmarks/power/max78000/peak-current/hil.yaml \
  task.trials 2 \
  --llm openai/gpt-5.4 \
  --output-dir outputs/max78000-peak-hil-gpt \
  --overwrite
```

## Model Compression

MAX78000 YOLO/COCO compression:

```bash
embedded-arena run configs/benchmarks/compression/max78000/hil.yaml \
  --llm openai/gpt-5.4 \
  --reasoning high \
  --output-dir outputs/compression-max78000-hil-gpt \
  --overwrite
```

STM32N6 speech-to-IPA compression:

```bash
embedded-arena run configs/benchmarks/compression/stm32n6/hil.yaml \
  --llm gemini/gemini-3.1-pro \
  --reasoning high \
  --output-dir outputs/compression-stm32n6-hil-gemini \
  --overwrite
```

Before running STM32N6, install ST Edge AI and download Hugging Face assets; see [Setup](setup.md#stm32n6-toolchain) and [Data/assets](data-assets.md#hugging-face-assets).

## MAX78000 Power Minimization

Peak current:

```bash
embedded-arena run configs/benchmarks/power/max78000/peak-current/hil.yaml \
  --llm openai/gpt-5.4 \
  --reasoning high \
  --output-dir outputs/max78000-peak-current-hil-gpt \
  --overwrite
```

Total energy:

```bash
embedded-arena run configs/benchmarks/power/max78000/total-energy/hil.yaml \
  --llm openai/gpt-5.4 \
  --reasoning high \
  --output-dir outputs/max78000-total-energy-hil-gpt \
  --overwrite
```

These require the MAX78000 hardware setup, PPK2, debugger, and UART described in [Hardware](hardware.md#max78000-power-and-energy).

## ESP32-S3 Thermal Management

Room-temperature peak temperature:

```bash
embedded-arena run configs/benchmarks/thermal/esp32/room/hil.yaml \
  --llm gemini/gemini-3.1-pro \
  --reasoning high \
  --output-dir outputs/esp32-room-hil-gemini \
  --overwrite
```

Contact-heated peak temperature:

```bash
embedded-arena run configs/benchmarks/thermal/esp32/contact/hil.yaml \
  --llm gemini/gemini-3.1-pro \
  --reasoning high \
  --output-dir outputs/esp32-contact-hil-gemini \
  --overwrite
```

These require ESP-IDF, the target ESP32-S3, and the MLX90640 bridge described in [Hardware](hardware.md#esp32-s3-thermal-management).

## Required Baselines For New Contributions

For a new benchmark contribution, run at least:

```bash
scripts/run_required_baselines.sh configs/benchmarks/<task>/<hardware>[/<experiment>]/<variant>.yaml
```

The script is a convenience wrapper. Keep the model list current with the latest generally available OpenAI model, the latest generally available Gemini model, and one additional model of your choice. Record exact model IDs and dates in the PR or benchmark README.

## Information Settings

The benchmark separates static knowledge from physical feedback:

- Score (`score.yaml`): the agent gets the task, seed files, and scalar/pass-fail feedback.
- Documentation (`documentation.yaml`): the agent also receives curated hardware/toolchain documentation.
- HIL (`hil.yaml`): the agent receives detailed feedback after each attempt, such as compiler diagnostics, resource usage, UART logs, current traces, thermal statistics, and feedback images.

This separation is important for papers and leaderboard-style comparisons. Do not silently add hardware feedback to a score-only or documentation-only config.
