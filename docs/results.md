# Results

**Docs:** [Overview](README.md) | [Setup](setup.md) | [Hardware](hardware.md) | [Data/assets](data-assets.md) | [Experiments](experiments.md) | [Results](results.md) | [Adding benchmarks](adding-hardware.md) | [Model providers](model-providers.md) | [Safety](safety.md)

Every run writes a structured output directory. Use `--output-dir` for explicit paths, or `--output-name` to place a run under `outputs/`.

## Output Tree

```text
outputs/<run>/
  summary.json
  run.log
  run_short.log
  trial_0/
    iter_0/
      check_outputs/
      feedback_images/
      sandbox_snapshot/        # only when --snapshot-sandbox is enabled
```

`summary.json` is the compact run-level artifact for scripts and papers. `run.log` is JSONL with full iteration events, tool calls, check results, and feedback. `run_short.log` is truncated for quick inspection.

## Inspecting A Run

```bash
python scripts/analyze_hil.py outputs/<run>/summary.json
python scripts/plot_results.py outputs/<run>/summary.json --out outputs/<run>/plots
```

The exact plotting arguments may differ by benchmark; check each script's `--help` output.

## Feedback Images

Checks may return image paths through `feedback_image_paths`. The runner includes these in the next agent observation when feedback is enabled. Thermal experiments often produce IR frames or time panels; MAX78000 power experiments may include current traces.

## Resume And Snapshots

Use `--resume` to append trials to an existing output directory:

```bash
embedded-arena run configs/benchmarks/compression/stm32n6/hil.yaml \
  --llm openai/gpt-5.4 \
  --trials 2 \
  --output-dir outputs/compression-stm32n6-hil-gpt \
  --resume
```

Use `--snapshot-sandbox` only when you need source-level forensics. It copies candidate files after every iteration and can consume substantial storage on multi-trial HIL runs.

## Reporting Results

When sharing a benchmark result, include:

- Git commit hash.
- Config path and any dotted-key overrides.
- Exact model ID, reasoning setting, and provider.
- Number of iterations and trials.
- Whether the run used `score`, `documentation`, or `hil`.
- Hardware serial/board revision and relevant toolchain versions.
- Any failed setup checks or manual recovery steps.
