"""UART serial capture helper used alongside PPK2 measurements."""

from __future__ import annotations

import time
import traceback
from multiprocessing import Process, Queue

import serial


_MAX_RECONNECT_ATTEMPTS = 5
_RECONNECT_DELAY_S = 1.0


def serial_reader_task(
    port: str,
    baud_rate: int,
    queue: Queue,
    start_byte: bytes | None = None,
    start_byte_delay_s: float = 0.5,
) -> None:
    """Read UART lines in a subprocess and push them into a queue.

    If start_byte is provided, write it once after opening the port (after
    start_byte_delay_s seconds) so the firmware's host-handshake read
    unblocks and t=0 is taken at this point.
    """
    try:
        ser = serial.Serial(port, baud_rate, timeout=1)
        if start_byte:
            time.sleep(start_byte_delay_s)
            try:
                ser.write(start_byte)
                ser.flush()
                queue.put(f"--- harness sent start byte 0x{start_byte.hex()} ---")
            except Exception as exc:
                queue.put(f"SERIAL_START_BYTE_ERROR: {exc}")
        reconnect_attempts = 0
        while True:
            try:
                line = ser.readline().decode("utf-8", errors="ignore").strip()
                reconnect_attempts = 0
                if line:
                    queue.put(line)
            except serial.SerialException:
                if reconnect_attempts >= _MAX_RECONNECT_ATTEMPTS:
                    raise
                reconnect_attempts += 1
                time.sleep(_RECONNECT_DELAY_S)
                try:
                    ser.close()
                except Exception:
                    pass
                try:
                    ser.open()
                except Exception:
                    pass
    except Exception as exc:
        queue.put(f"SERIAL_READER_ERROR: {exc}\n{traceback.format_exc()}")


def detect_dut_port() -> str | None:
    """Best-effort detection of the DUT serial port."""
    ports = serial.tools.list_ports.comports()
    for port_info in ports:
        device = port_info.device
        if "usbmodem" in device:
            parts = device.split("usbmodem")
            if len(parts) > 1 and parts[1].isdigit():
                return device
    return None


class SerialMonitor:
    """Non-blocking UART capture with a subprocess reader."""

    def __init__(
        self,
        port: str | None,
        baud_rate: int,
        start_byte: bytes | None = None,
        start_byte_delay_s: float = 0.5,
    ):
        self.port = port
        self.baud_rate = baud_rate
        self.start_byte = start_byte
        self.start_byte_delay_s = start_byte_delay_s
        self.queue: Queue = Queue()
        self.process: Process | None = None

    def start(self) -> bool:
        if not self.port:
            return False
        self.process = Process(
            target=serial_reader_task,
            args=(
                self.port,
                self.baud_rate,
                self.queue,
                self.start_byte,
                self.start_byte_delay_s,
            ),
        )
        self.process.start()
        return True

    def stop(self) -> None:
        if self.process:
            self.process.terminate()
            self.process.join()
            self.process = None

    def collect(self) -> str:
        lines: list[str] = []
        while not self.queue.empty():
            lines.append(self.queue.get())

        if not lines:
            return "--- Serial Monitor ---\nNo data collected.\n"
        if len(lines) == 1 and "SERIAL_READER_ERROR" in lines[0]:
            return f"--- Serial Monitor ---\n{lines[0]}\nNo data due to error.\n"
        return "--- Serial Monitor ---\n" + "\n".join(lines)
