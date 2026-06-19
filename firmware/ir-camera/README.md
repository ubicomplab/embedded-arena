# MLX90640 IR Camera Bridge

Related docs: [Hardware](../../docs/hardware.md#esp32-s3-thermal-management) | [Setup](../../docs/setup.md#esp32-s3-toolchain)

This PlatformIO firmware streams MLX90640 thermal frames from an ESP32-S3 bridge to the EmbeddedArena host driver in `embedded_arena/hardware/ir_camera.py`. The thermal-management checks use it to measure the target ESP32-S3 workload from outside the board under test.

Typical bring-up:

```bash
cd firmware/ir-camera
pio run -t upload
cd ../..
python scripts/test_ir_camera.py
```

Keep the target board centered in the MLX90640 field of view and document the physical placement for any new benchmark variant.
