# ESP32-S3 TinyLLaMA Thermal Workload

Related docs: [Hardware](../../../docs/hardware.md#esp32-s3-thermal-management) | [Setup](../../../docs/setup.md#esp32-s3-toolchain) | [Experiments](../../../docs/experiments.md#esp32-s3-thermal-management)

This seed firmware is adapted from [DaveBben/esp32-llm](https://github.com/DaveBben/esp32-llm). EmbeddedArena uses it as the ESP32-S3 thermal-management workload: agents edit this project, the harness builds and flashes it with ESP-IDF, and the measurement check verifies that the required TinyLLaMA/UDP workload still runs while thermal measurements are collected with the MLX90640 bridge.

The benchmark-specific behavior contract is enforced by `embedded_arena/checks/measure_esp32.py` and serial/thermal instrumentation, not solely by prompt text. In particular, the firmware must preserve the host start-byte handshake, SoftAP/UDP token fan-out, all required prompts, UART progress output, and the final `firmware task complete checkpoint` line.

The generated text itself is not evaluated; the workload exists to create a repeatable compute, WiFi, memory, and thermal profile for optimization.
