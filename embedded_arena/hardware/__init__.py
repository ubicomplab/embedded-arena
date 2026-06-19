"""Hardware-in-the-loop drivers and analysis utilities.

These modules wrap external hardware so checks/tools can flash firmware,
profile power, and capture thermal frames without re-implementing the
serial/JTAG plumbing each time.

Available modules:
  - max78000_compiler: build the MAX78000 firmware via Maxim SDK `make`.
  - max78000_flasher:  program and erase the MAX78000 over JTAG/SWD via
                       arm-none-eabi-gdb + OpenOCD.
  - esp32_compiler:    build an ESP-IDF firmware project via `idf.py build`.
                       Requires IDF_PATH or idf.py on PATH.
  - esp32_flasher:     flash an ESP32 via `idf.py -p PORT flash`.
                       Requires ESP32_PORT (env) or explicit port argument.
  - ppk2:              Nordic PPK2 power profiler.
  - serial_monitor:    UART serial capture helper.
  - ir_camera:         ESP32-S3 + MLX90640 thermal camera driver.
  - metrics:           per-cycle energy/latency calculations from PPK data.
  - plotter:           matplotlib helpers for current-profile / iter trends.
"""
