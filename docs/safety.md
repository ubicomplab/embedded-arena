# Safety

**Docs:** [Overview](README.md) | [Setup](setup.md) | [Hardware](hardware.md) | [Data/assets](data-assets.md) | [Experiments](experiments.md) | [Results](results.md) | [Adding benchmarks](adding-hardware.md) | [Model providers](model-providers.md) | [Safety](safety.md)

EmbeddedArena can power, flash, heat, and reset physical devices. Safety and recoverability should be enforced by framework checks and instrument settings, not by prompt instructions alone.

## Sandbox Safety

Agent-authored code runs in a Docker sandbox. The sandbox cleaner:

- Defaults outside the repository at `~/.cache/embedded-arena/sandboxes/default`.
- Treats blank `EMBEDDED_ARENA_CACHE_DIR` and `EMBEDDED_ARENA_SANDBOX_PATH` as unset.
- Refuses repository paths.
- Only cleans directories marked with `.embedded-arena-sandbox`.

Never set `--sandbox-path` to the repository root, a home directory, or any directory containing important files.

## Host Hardware Boundary

Hardware checks run on the host after the agent submits a final JSON object. The agent does not get a direct "flash this board" tool. The trust boundary is:

1. Pydantic `Input` schemas validate agent-supplied fields.
2. YAML `params` hold fixed safety-critical measurement settings.
3. Host-side check code owns flashing, power, serial capture, and instruments.

This design is necessary because macOS Docker Desktop does not expose host USB devices to containers, and because vendor flashing tools live on the host.

## Power Safety

- Verify PPK2 source voltage before connecting the target.
- Tie grounds between PPK2, debugger, target board, and serial adapters.
- Start with conservative current limits and short measurement windows.
- Abort on unexpected heating, repeated brownouts, or unstable USB devices.
- Do not allow benchmark configs to expose target voltage or safety limits as agent-controlled fields.

## Thermal Safety

- Keep contact-heated setups attended.
- Use conservative temperature ceilings for skin-contact or wearable scenarios.
- Avoid enclosing ESP32 boards during exploratory thermal runs unless the enclosure is part of the benchmark.
- Let boards cool between repeated runs when comparing peak temperature.

## Recoverability

Firmware workloads should remain recoverable for the next flash. Enforce this with handshakes, watchdogs, post-task idle behavior, or hardware reset procedures where possible. Do not rely only on prompt text asking the agent to avoid unrecoverable sleep states.

If a board becomes hard to flash:

1. Remove external power.
2. Hold the vendor boot/reset sequence.
3. Reconnect the debugger/USB.
4. Run a known-good flash command manually.
5. Only then resume the benchmark.

## Secrets And Logs

- `.env` is local and should never be committed.
- Review `run.log`, `summary.json`, screenshots, and docs for API keys, serial numbers, local usernames, or private paths before sharing.
- Prefer relative paths in docs and configs.
