#!/bin/bash

set -e

# Configuration
CONFIGS=(
    "configs/experiments/hil-firmware-esp32.yaml"
    "configs/experiments/hil-firmware-esp32-doc.yaml"
    "configs/experiments/hil-firmware-esp32-no-feedback.yaml"
)

MODELS=(
    "openai/gpt-5.4"
    "openai/gpt-5.4-mini"
    "claude/claude-opus-4-7"
    "claude/claude-sonnet-4-6"
    "gemini/gemini-3.1-pro-preview"
    "gemini/gemini-3-flash-preview"
)

MODEL_NAMES=(
    "gpt"
    "gptl"
    "claude"
    "claudel"
    "gemini"
    "geminil"
)

RUNS_PER_MODEL=1  # Run 1 time per model per config
CONSECUTIVE_FAILURES_LIMIT=3

# Tracking
CONSECUTIVE_FAILURES=0
TOTAL_RUNS=0
TOTAL_SUCCESSFUL=0
TOTAL_FAILED=0
FAILED_RUNS=()

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Extract experiment name from config path
get_experiment_name() {
    basename "$1" .yaml
}

# Run a single experiment
run_experiment() {
    local config="$1"
    local llm="$2"
    local model_name="$3"
    local run_num="$4"
    local experiment_name=$(get_experiment_name "$config")

    # Output name uses experiment name (which already includes doc/no-feedback suffix) + model + run
    local output_name="${experiment_name}_${model_name}_${run_num}"
    local output_dir="outputs/${output_name}"

    TOTAL_RUNS=$((TOTAL_RUNS + 1))

    # Check if output directory already exists
    if [ -d "$output_dir" ]; then
        echo ""
        echo "=========================================="
        echo "Run $TOTAL_RUNS of $((${#CONFIGS[@]} * ${#MODELS[@]} * RUNS_PER_MODEL))"
        echo "Config: $config"
        echo "Model: $llm ($model_name)"
        echo "Output: $output_dir"
        echo "=========================================="
        echo -e "${YELLOW}⊘ SKIPPED${NC}: $output_name (directory already exists)"
        CONSECUTIVE_FAILURES=0  # Don't count skips as failures
        return 0
    fi

    echo ""
    echo "=========================================="
    echo "Run $TOTAL_RUNS of $((${#CONFIGS[@]} * ${#MODELS[@]} * RUNS_PER_MODEL))"
    echo "Config: $config"
    echo "Model: $llm ($model_name)"
    echo "Output: $output_dir"
    echo "=========================================="

    if python -m src.run "$config" --llm "$llm" --reasoning high --snapshot-sandbox --output-name "$output_name"; then
        echo -e "${GREEN}✓ SUCCESS${NC}: $output_name"
        TOTAL_SUCCESSFUL=$((TOTAL_SUCCESSFUL + 1))
        CONSECUTIVE_FAILURES=0
    else
        EXIT_CODE=$?
        echo -e "${RED}✗ FAILED${NC}: $output_name (exit code: $EXIT_CODE)"
        TOTAL_FAILED=$((TOTAL_FAILED + 1))
        CONSECUTIVE_FAILURES=$((CONSECUTIVE_FAILURES + 1))
        FAILED_RUNS+=("$output_name")

        if [ $CONSECUTIVE_FAILURES -ge $CONSECUTIVE_FAILURES_LIMIT ]; then
            echo ""
            echo -e "${RED}ERROR: $CONSECUTIVE_FAILURES_LIMIT consecutive failures. Cancelling batch run.${NC}"
            return 1
        fi
    fi
    return 0
}

# Main batch run
main() {
    local start_time=$(date +%s)

    echo ""
    echo "=========================================="
    echo "Starting Batch HIL Experiments"
    echo "Configs: ${#CONFIGS[@]}"
    echo "Models: ${#MODELS[@]}"
    echo "Runs per model: $RUNS_PER_MODEL"
    echo "Total runs: $((${#CONFIGS[@]} * ${#MODELS[@]} * RUNS_PER_MODEL))"
    echo "=========================================="

    # Iterate through all combinations
    for config in "${CONFIGS[@]}"; do
        for model_idx in "${!MODELS[@]}"; do
            model="${MODELS[$model_idx]}"
            model_name="${MODEL_NAMES[$model_idx]}"

            for run_num in $(seq 1 $RUNS_PER_MODEL); do
                run_experiment "$config" "$model" "$model_name" "$run_num"
                if [ $? -ne 0 ]; then
                    # Batch cancelled due to consecutive failures
                    end_time=$(date +%s)
                    duration=$((end_time - start_time))
                    print_summary "$duration"
                    exit 1
                fi
            done
        done
    done

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    print_summary "$duration"
}

# Print summary
print_summary() {
    local duration=$1
    local hours=$((duration / 3600))
    local minutes=$(( (duration % 3600) / 60 ))
    local seconds=$((duration % 60))

    echo ""
    echo "=========================================="
    echo "Batch Run Complete"
    echo "=========================================="
    echo "Total runs:    $TOTAL_RUNS"
    echo "Successful:    ${GREEN}$TOTAL_SUCCESSFUL${NC}"
    echo "Failed:        ${RED}$TOTAL_FAILED${NC}"
    echo "Duration:      ${hours}h ${minutes}m ${seconds}s"

    if [ ${#FAILED_RUNS[@]} -gt 0 ]; then
        echo ""
        echo "Failed runs:"
        for run in "${FAILED_RUNS[@]}"; do
            echo "  - $run"
        done
    fi
    echo "=========================================="

    if [ $TOTAL_FAILED -eq 0 ]; then
        echo -e "${GREEN}All runs completed successfully!${NC}"
        return 0
    else
        echo -e "${YELLOW}Some runs failed. Check logs above for details.${NC}"
        return 1
    fi
}

# Run main
main
