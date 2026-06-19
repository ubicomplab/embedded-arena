# Model Providers

**Docs:** [Overview](README.md) | [Setup](setup.md) | [Hardware](hardware.md) | [Data/assets](data-assets.md) | [Experiments](experiments.md) | [Results](results.md) | [Adding benchmarks](adding-hardware.md) | [Model providers](model-providers.md) | [Safety](safety.md)

Use `--llm provider/model` to choose an adapter. Provider keys live in `.env`.

## Built-In Providers

| Provider | Example | Env var |
| --- | --- | --- |
| OpenAI | `openai/gpt-5.4` | `OPENAI_API_KEY` |
| Google Gemini | `gemini/gemini-3.1-pro` | `GOOGLE_API_KEY` |
| Anthropic Claude | `claude/claude-sonnet-4.6` | `ANTHROPIC_API_KEY` |
| Ollama | `ollama/<local-model>` | none |
| Scripted CLI | `cli/scripted` | `CLI_LLM_SCRIPT` |

Install hosted provider dependencies with:

```bash
python -m pip install -e '.[providers]'
```

## Reasoning Setting

Pass a provider-specific reasoning hint when supported:

```bash
embedded-arena run configs/smoke/gradient-flow.yaml \
  --llm openai/gpt-5.4 \
  --reasoning high \
  --output-dir outputs/smoke-gpt \
  --overwrite
```

Adapters that do not support reasoning should ignore the hint.

## Scripted Adapter

The scripted adapter is useful for smoke tests and CI:

```bash
CLI_LLM_SCRIPT=examples/cli_smoke_gradient_flow.jsonl \
  embedded-arena run configs/smoke/gradient-flow.yaml \
  --llm cli/scripted \
  --iterations 1 \
  --output-dir outputs/smoke \
  --overwrite
```

The JSONL file contains pre-recorded assistant turns. This lets the framework test sandboxing, check execution, and logging without using paid API calls.

## Adding A Provider

Add `embedded_arena/llms/<provider>.py` with a `build(model, reasoning=None)` function returning the common LLM interface. Keep provider-specific dependencies optional when possible, document required env vars here, and add at least one smoke test path that does not require hardware.

## Baseline Policy

For new benchmark contributions, run the latest generally available OpenAI model, latest generally available Gemini model, and one additional model. Record exact model names and dates because provider aliases change over time.
