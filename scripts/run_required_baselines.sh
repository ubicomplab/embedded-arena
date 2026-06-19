#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:?usage: scripts/run_required_baselines.sh CONFIG [OUTPUT_PREFIX]}
PREFIX=${2:-outputs/baselines}

models=(
  "openai/gpt-5.4"
  "gemini/gemini-3.1-pro"
  "claude/claude-sonnet-4.6"
)

for model in "${models[@]}"; do
  safe_name=${model//\//-}
  embedded-arena run "$CONFIG" --llm "$model" --iterations 10 --output-dir "$PREFIX/$safe_name" --overwrite
 done
