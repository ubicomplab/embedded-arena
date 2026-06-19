# MAX78000 YOLO Pico Firmware

Related docs: [Hardware](../../../docs/hardware.md#max78000-power-and-energy) | [Setup](../../../docs/setup.md#max78000-toolchain) | [Experiments](../../../docs/experiments.md#max78000-power-minimization)

This seed firmware is adapted from [SanderGi/YADES](https://github.com/SanderGi/YADES). EmbeddedArena uses it as the MAX78000 camera-inference workload for power-minimization experiments: agents edit this project, the harness builds it with the Analog Devices/Maxim SDK, flashes it to the board, and measures current with the configured hardware instruments.

The benchmark-specific behavior contract is enforced by the compile, flash, and measurement checks in `embedded_arena/checks/`. Hardware runs must preserve the live camera path, CNN execution, post-processing, UART progress logging, and the final `firmware task complete checkpoint` line required by the measurement harness.

The project is copied into each sandbox at `firmware/`; edits made by an agent during a run do not modify this seed directory.
