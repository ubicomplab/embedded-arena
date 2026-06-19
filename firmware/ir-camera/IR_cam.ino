/*
 * ESP32-S3 DevKitC-1 + MLX90640 thermal camera firmware — dual-core edition.
 *
 * Core assignment:
 *   Core 1 — senseTask: owns the I2C bus; calls mlx.getFrame() continuously
 *             and writes into the back buffer of a double-buffer pair.
 *   Core 0 — commsTask: owns Serial; drains commands, waits for frameReadySem,
 *             copies the front buffer, and transmits the frame.
 *
 * The two cores communicate through three primitives:
 *   frameMutex   — FreeRTOS mutex; guards the front/back index swap so comms
 *                  never reads a buffer that sense is mid-write.
 *   frameReadySem — binary semaphore; sense gives after each completed frame,
 *                  comms takes before sending. If comms is slow, the extra give
 *                  fails silently (binary semantics) and sense moves on — the
 *                  stale frame is overwritten; comms will send the latest one.
 *   volatile flags — streaming, pendingRateChange, pendingRate are volatile
 *                  booleans/enums. Writes are single-word stores on Xtensa LX7
 *                  so they are inherently atomic; volatile ensures the compiler
 *                  re-reads them every time rather than caching in a register.
 *
 * Commands (host -> device, ASCII '\n'-terminated):
 *   START        Enable continuous streaming.
 *   STOP         Disable streaming.
 *   RATE <hz>    Set refresh rate. Valid: 0.5, 1, 2, 4, 8, 16, 32, 64.
 *   PING         Liveness check; replies "PONG".
 *   INFO         Replies with an "INFO <json>" sensor-descriptor line.
 *
 * Binary frame format (device -> host, streaming):
 *   Fixed 3088-byte record with CRC-16/CCITT-FALSE protection.
 *     offset  size   field
 *      0       4     magic          0xAA 0x55 0xF0 0x0D (fixed byte order)
 *      4       2     seq            u16 LE (wraps)
 *      6       4     ts_ms          u32 LE (millis())
 *     10       2     payload_len    u16 LE (= 3072)
 *     12       2     header_crc16   u16 LE, CRC over bytes [0..11]
 *     14    3072     payload        float32 LE pixels, row-major 24x32
 *   3086       2     payload_crc16  u16 LE, CRC over bytes [14..3085]
 *
 *   ASCII command/response lines and binary frames are disjoint: text is only
 *   emitted when not mid-frame; any host-side byte that is not part of one of
 *   the known ASCII responses is framed binary data and is resynced by scanning
 *   for the 4-byte magic.
 *
 * Wiring (MLX90640 breakout -> ESP32-S3 DevKitC-1):
 *   VIN -> 3V3   GND -> GND
 *   SDA -> GPIO 8   SCL -> GPIO 9   (Arduino-ESP32 S3 defaults; add 2.2k pull-ups)
 */

#include <Wire.h>
#include <Adafruit_MLX90640.h>

// ============================================================================
// Compile-time configuration
// ============================================================================

static constexpr uint32_t UART_BAUD   = 921600;
static constexpr uint8_t  I2C_SDA    = 8;
static constexpr uint8_t  I2C_SCL    = 9;
// Run at 1 MHz (fast-mode-plus). 400 kHz cannot sustain rates above 2 Hz.
static constexpr uint32_t I2C_FREQ   = 1000000;

static constexpr uint16_t PIXEL_W    = 32;
static constexpr uint16_t PIXEL_H    = 24;
static constexpr uint16_t PIXEL_N    = PIXEL_W * PIXEL_H;  // 768
static constexpr size_t   FRAME_BYTES = PIXEL_N * sizeof(float);  // 3072

// Binary frame layout constants — keep in lockstep with ir_camera.py.
static constexpr uint8_t  MAGIC0        = 0xAA;
static constexpr uint8_t  MAGIC1        = 0x55;
static constexpr uint8_t  MAGIC2        = 0xF0;
static constexpr uint8_t  MAGIC3        = 0x0D;
static constexpr size_t   HEADER_SIZE   = 14;                               // magic..header_crc16 inclusive
static constexpr size_t   FRAME_TOTAL   = HEADER_SIZE + FRAME_BYTES + 2;    // 3088

// Max ASCII command line length. Anything longer is truncated and rejected.
static constexpr size_t   LINE_BUF_SIZE = 64;

// USB CDC TX timeout. Default is effectively 0 (non-blocking) on some cores,
// which silently drops tail bytes when the host momentarily back-pressures.
// 50 ms is enough for any sane host hiccup while still bounded.
static constexpr uint32_t CDC_TX_TIMEOUT_MS = 50;

// Task stack sizes in bytes. commsTask owns a ~3 KB local frame-transmit buffer
// plus line-accumulator state; senseTask needs stack for mlx library internals.
static constexpr uint32_t COMMS_STACK = 12 * 1024;
static constexpr uint32_t SENSE_STACK =  8 * 1024;

// ============================================================================
// Shared state between tasks
// ============================================================================

static Adafruit_MLX90640 mlx;

// Double-buffer: senseTask writes into frameBuf[backIdx]; commsTask reads
// frameBuf[frontIdx]. frontIdx is swapped under frameMutex.
static float             frameBuf[2][PIXEL_N];
static volatile uint8_t  frontIdx = 0;
static SemaphoreHandle_t frameMutex;    // protects frontIdx swap
static SemaphoreHandle_t frameReadySem; // binary; signals a new frame is ready

// Command-state flags.  All are written by commsTask and read by senseTask
// (or vice-versa for error reporting). volatile is sufficient on LX7 because
// these are single-word values.
static volatile bool                  streaming        = false;
static volatile bool                  pendingRateChange = false;
static volatile mlx90640_refreshrate_t pendingRate     = MLX90640_16_HZ;
static volatile mlx90640_refreshrate_t currentRate     = MLX90640_16_HZ;

// Set to true by senseTask once the MLX90640 has been successfully initialised.
// commsTask reads this to gate START and report sensor status via INFO.
static volatile bool                  sensorReady      = false;

// ============================================================================
// Logging / defensive-check helpers
// ============================================================================

// All Serial output goes through commsTask, but for fatal init errors we
// print directly from setup() before tasks are created. After tasks launch,
// only commsTask calls Serial.
#define LOG_INFO(msg) do { Serial.print(F("INFO ")); Serial.println(msg); } while (0)
#define LOG_ERR(msg)  do { Serial.print(F("ERR "));  Serial.println(msg); } while (0)

// Halt on unrecoverable errors: print once, then blink ERR every second
// so the host can detect the failure without a silent hang.
#define HALT_ON_FAIL(cond, msg)                 \
    do {                                         \
        if (!(cond)) {                           \
            LOG_ERR(msg);                        \
            for (;;) {                           \
                LOG_ERR(F("halted"));            \
                vTaskDelay(pdMS_TO_TICKS(1000)); \
            }                                    \
        }                                        \
    } while (0)

// ============================================================================
// Refresh-rate table
// ============================================================================

struct RateEntry { float hz; mlx90640_refreshrate_t code; };
static const RateEntry kRateTable[] = {
    {0.5f, MLX90640_0_5_HZ}, {1.0f, MLX90640_1_HZ},
    {2.0f, MLX90640_2_HZ},   {4.0f, MLX90640_4_HZ},
    {8.0f, MLX90640_8_HZ},   {16.0f, MLX90640_16_HZ},
    {32.0f, MLX90640_32_HZ}, {64.0f, MLX90640_64_HZ},
};

static bool lookupRate(float hz, mlx90640_refreshrate_t &out) {
    for (const auto &e : kRateTable)
        if (fabsf(hz - e.hz) < 0.01f) { out = e.code; return true; }
    return false;
}

// ============================================================================
// CRC-16/CCITT-FALSE  (poly 0x1021, init 0xFFFF, no refin/refout, xorout 0)
// ============================================================================

static uint16_t crc16_ccitt(const uint8_t *data, size_t len) {
    uint16_t crc = 0xFFFF;
    for (size_t i = 0; i < len; ++i) {
        crc ^= static_cast<uint16_t>(data[i]) << 8;
        for (uint8_t b = 0; b < 8; ++b) {
            crc = (crc & 0x8000) ? static_cast<uint16_t>((crc << 1) ^ 0x1021)
                                 : static_cast<uint16_t>(crc << 1);
        }
    }
    return crc;
}

// ============================================================================
// Command handler  (called exclusively from commsTask on Core 0)
// ============================================================================

// Case-insensitive match for a zero-terminated trimmed command against a
// literal ASCII keyword. Kept local because we dropped the String class.
static bool ieq(const char *cmd, const char *kw) {
    while (*cmd && *kw) {
        char a = *cmd++, b = *kw++;
        if (a >= 'A' && a <= 'Z') a = static_cast<char>(a + 32);
        if (b >= 'A' && b <= 'Z') b = static_cast<char>(b + 32);
        if (a != b) return false;
    }
    return *cmd == '\0' && *kw == '\0';
}

static bool startsWithCI(const char *cmd, const char *prefix) {
    while (*prefix) {
        char a = *cmd++, b = *prefix++;
        if (!a) return false;
        if (a >= 'A' && a <= 'Z') a = static_cast<char>(a + 32);
        if (b >= 'A' && b <= 'Z') b = static_cast<char>(b + 32);
        if (a != b) return false;
    }
    return true;
}

// Native USB CDC: false when the host closes the serial port (DTR/session
// gone). UART builds: no disconnect signal — always true. Stopping streaming
// here prevents a zombie stream that breaks the next host session.
static inline bool irHostSerialConnected() {
#if defined(ARDUINO_USB_CDC_ON_BOOT) && (ARDUINO_USB_CDC_ON_BOOT == 1)
    return static_cast<bool>(Serial);
#else
    return true;
#endif
}

static void handleCommand(char *cmd) {
    // Trim leading whitespace in place.
    while (*cmd == ' ' || *cmd == '\t' || *cmd == '\r') ++cmd;
    // Trim trailing whitespace.
    size_t n = strlen(cmd);
    while (n > 0 && (cmd[n - 1] == ' ' || cmd[n - 1] == '\t' || cmd[n - 1] == '\r')) {
        cmd[--n] = '\0';
    }
    if (n == 0) return;

    if (ieq(cmd, "START")) {
        if (!sensorReady) {
            Serial.println(F("ERR sensor not ready"));
            return;
        }
        streaming = true;
        Serial.println(F("OK"));

    } else if (ieq(cmd, "STOP")) {
        streaming = false;
        Serial.println(F("OK"));

    } else if (ieq(cmd, "PING")) {
        Serial.println(F("PONG"));

    } else if (ieq(cmd, "INFO")) {
        // Build the full line in a single buffer and emit with ONE write.
        // Chaining Serial.print() on USB CDC drops bytes mid-string on
        // ESP32-Arduino (the background TX task can race with successive
        // tiny writes); we consolidate to avoid that entirely.
        char line[192];
        int n = snprintf(line, sizeof(line),
            "INFO {\"sensor\":\"MLX90640\",\"w\":%u,\"h\":%u,\"pixels\":%u,"
            "\"bytes_per_pixel\":4,\"format\":\"float32_le\",\"ready\":%s}\r\n",
            static_cast<unsigned>(PIXEL_W),
            static_cast<unsigned>(PIXEL_H),
            static_cast<unsigned>(PIXEL_N),
            sensorReady ? "true" : "false");
        if (n > 0) {
            Serial.write(reinterpret_cast<const uint8_t *>(line),
                         static_cast<size_t>(n));
        }

    } else if (startsWithCI(cmd, "RATE ")) {
        float hz = strtof(cmd + 5, nullptr);
        mlx90640_refreshrate_t r;
        if (lookupRate(hz, r)) {
            // Flag the change; senseTask applies it between frames so the
            // I2C bus is only touched from Core 1.
            pendingRate = r;
            pendingRateChange = true;
            char line[40];
            int n = snprintf(line, sizeof(line), "OK RATE=%.2f\r\n",
                             static_cast<double>(hz));
            if (n > 0) {
                Serial.write(reinterpret_cast<const uint8_t *>(line),
                             static_cast<size_t>(n));
            }
        } else {
            Serial.println(F("ERR invalid rate (valid: 0.5,1,2,4,8,16,32,64)"));
        }

    } else {
        // One-shot write to avoid the split-print drop bug observed on USB CDC.
        char line[96];
        int n = snprintf(line, sizeof(line), "ERR unknown command: %s\r\n", cmd);
        if (n > 0) {
            Serial.write(reinterpret_cast<const uint8_t *>(line),
                         static_cast<size_t>(n));
        }
    }
}

// ============================================================================
// Frame transmission  (called exclusively from commsTask on Core 0)
// ============================================================================

// Assemble one fixed-size binary frame and push it out in a single write.
// `txBuf` must be at least FRAME_TOTAL bytes and is supplied by the caller
// (commsTask keeps it on its own stack — see COMMS_STACK sizing).
static void sendFrame(const float *data, size_t nPixels,
                      uint8_t *txBuf, uint16_t seq) {
    HALT_ON_FAIL(data  != nullptr,                    F("sendFrame: null data"));
    HALT_ON_FAIL(txBuf != nullptr,                    F("sendFrame: null buf"));
    HALT_ON_FAIL(nPixels == PIXEL_N,                  F("sendFrame: wrong pixel count"));

    const uint32_t ts = millis();

    // Header (14 bytes). Little-endian on Xtensa LX7; written byte-wise so the
    // layout is explicit and independent of any struct-packing assumptions.
    txBuf[0]  = MAGIC0;
    txBuf[1]  = MAGIC1;
    txBuf[2]  = MAGIC2;
    txBuf[3]  = MAGIC3;
    txBuf[4]  = static_cast<uint8_t>(seq & 0xFF);
    txBuf[5]  = static_cast<uint8_t>((seq >> 8) & 0xFF);
    txBuf[6]  = static_cast<uint8_t>(ts & 0xFF);
    txBuf[7]  = static_cast<uint8_t>((ts >> 8) & 0xFF);
    txBuf[8]  = static_cast<uint8_t>((ts >> 16) & 0xFF);
    txBuf[9]  = static_cast<uint8_t>((ts >> 24) & 0xFF);
    txBuf[10] = static_cast<uint8_t>(FRAME_BYTES & 0xFF);
    txBuf[11] = static_cast<uint8_t>((FRAME_BYTES >> 8) & 0xFF);

    const uint16_t hcrc = crc16_ccitt(txBuf, 12);
    txBuf[12] = static_cast<uint8_t>(hcrc & 0xFF);
    txBuf[13] = static_cast<uint8_t>((hcrc >> 8) & 0xFF);

    // Payload. ESP32 is little-endian so float32 matches Python '<f4' directly.
    memcpy(txBuf + HEADER_SIZE, data, FRAME_BYTES);

    // Payload CRC over the 3072 raw bytes.
    const uint16_t pcrc = crc16_ccitt(txBuf + HEADER_SIZE, FRAME_BYTES);
    txBuf[HEADER_SIZE + FRAME_BYTES]     = static_cast<uint8_t>(pcrc & 0xFF);
    txBuf[HEADER_SIZE + FRAME_BYTES + 1] = static_cast<uint8_t>((pcrc >> 8) & 0xFF);

    // Single write keeps the USB CDC endpoint fed with one contiguous buffer,
    // which is materially faster and less drop-prone than ~12 small prints.
    const size_t wrote = Serial.write(txBuf, FRAME_TOTAL);
    Serial.flush();
    if (wrote != FRAME_TOTAL) {
        streaming = false;
    }
}

// ============================================================================
// Core 0 — communications task
// ============================================================================

static void commsTask(void *) {
    // Local frame copy; kept on this task's stack so senseTask can write the
    // next frame while we're still serialising this one.
    float   localFrame[PIXEL_N];
    // Pre-allocated transmit buffer for one binary frame (3088 bytes). Keeping
    // it on the stack avoids per-frame malloc and keeps the write contiguous.
    uint8_t txBuf[FRAME_TOTAL];

    // Non-blocking command-line accumulator. The previous version called
    // Serial.readStringUntil('\n') which blocks up to 1 s on a partial line,
    // starving the frame loop and causing the binary-semaphore backlog to
    // drop frames. We now consume bytes as they arrive and only dispatch on
    // a complete line.
    static char lineBuf[LINE_BUF_SIZE];
    static size_t lineLen = 0;

    uint16_t seq = 0;

    for (;;) {
        // Drain whatever bytes are currently available without blocking.
        while (Serial.available() > 0) {
            int c = Serial.read();
            if (c < 0) break;
            if (c == '\r') continue;   // swallow CR; we key off LF only.
            if (c == '\n') {
                lineBuf[lineLen] = '\0';
                handleCommand(lineBuf);
                lineLen = 0;
                continue;
            }
            if (lineLen + 1 < LINE_BUF_SIZE) {
                lineBuf[lineLen++] = static_cast<char>(c);
            } else {
                // Overflow: reset and report so the host knows it was dropped.
                lineLen = 0;
                Serial.println(F("ERR command too long"));
            }
        }

        if (!streaming) {
            vTaskDelay(pdMS_TO_TICKS(5));
            continue;
        }

        if (!irHostSerialConnected()) {
            streaming = false;
            vTaskDelay(pdMS_TO_TICKS(5));
            continue;
        }

        // Block up to 5 ms for a new frame; if none arrives within the window
        // loop back to drain commands again (keeps PING responsive during slow rates).
        if (xSemaphoreTake(frameReadySem, pdMS_TO_TICKS(5)) == pdTRUE) {
            // Copy under mutex — prevents a race with the sense-task buffer swap.
            xSemaphoreTake(frameMutex, portMAX_DELAY);
            memcpy(localFrame, frameBuf[frontIdx], FRAME_BYTES);
            xSemaphoreGive(frameMutex);

            sendFrame(localFrame, PIXEL_N, txBuf, seq);
            ++seq;
        }
    }
}

// ============================================================================
// Core 1 — sensing task
// ============================================================================

static void senseTask(void *) {
    // Initialise I2C and MLX90640 here, not in setup(), so commsTask is already
    // running and can answer PING while the sensor comes up.
    Wire.begin(I2C_SDA, I2C_SCL);
    Wire.setClock(I2C_FREQ);

    // Retry loop: allow the sensor up to ~10 s to become available after power-on.
    constexpr int MAX_INIT_ATTEMPTS = 20;
    bool initOk = false;
    for (int attempt = 0; attempt < MAX_INIT_ATTEMPTS; ++attempt) {
        if (mlx.begin(MLX90640_I2CADDR_DEFAULT, &Wire)) {
            initOk = true;
            break;
        }
        vTaskDelay(pdMS_TO_TICKS(500));
    }

    if (!initOk) {
        // Keep blinking ERR so a host can detect the failure, but stay alive
        // so PING continues to work and the host knows the device is up.
        for (;;) {
            LOG_ERR(F("MLX90640 not found - check wiring"));
            vTaskDelay(pdMS_TO_TICKS(2000));
        }
    }

    mlx.setMode(MLX90640_CHESS);
    mlx.setResolution(MLX90640_ADC_18BIT);
    mlx.setRefreshRate(currentRate);
    sensorReady = true;
    LOG_INFO(F("MLX90640 ready at 16 Hz (~8 fps CHESS)"));

    uint8_t backIdx = 1;  // start writing into buf[1]; comms starts reading buf[0]

    for (;;) {
        // Apply a pending rate change before the next acquisition. This must
        // happen on Core 1 because Wire is initialised here and is not
        // thread-safe across cores.
        if (pendingRateChange) {
            mlx.setRefreshRate(pendingRate);
            currentRate     = pendingRate;
            pendingRateChange = false;
        }

        if (!streaming) {
            vTaskDelay(pdMS_TO_TICKS(5));
            continue;
        }

        // getFrame() blocks for approximately 1/rate seconds (I2C transfers).
        // During that time commsTask runs freely on Core 0.
        int rc = mlx.getFrame(frameBuf[backIdx]);
        if (rc != 0) {
            // Never print from Core 1 while streaming: interleaving Serial writes
            // with commsTask can corrupt framed binary output.
            vTaskDelay(pdMS_TO_TICKS(20));
            continue;
        }

        // Swap front/back atomically so commsTask always reads a complete frame.
        xSemaphoreTake(frameMutex, portMAX_DELAY);
        frontIdx = backIdx;
        backIdx ^= 1;
        xSemaphoreGive(frameMutex);

        // Signal commsTask. Binary semaphore: if the previous frame hasn't been
        // consumed yet, this give() returns pdFAIL and is silently dropped — the
        // stale frame will be overwritten and commsTask will send the latest one.
        xSemaphoreGive(frameReadySem);
    }
}

// ============================================================================
// Arduino lifecycle
// ============================================================================

void setup() {
    Serial.begin(UART_BAUD);
    // USB CDC TX default is effectively non-blocking on some Arduino-ESP32
    // versions, which silently drops tail bytes under host back-pressure and
    // is the root cause of "short read N/3072" on the receiving side.
    Serial.setTxTimeoutMs(CDC_TX_TIMEOUT_MS);

    // Brief settle delay for USB CDC enumeration. We deliberately don't wait
    // on `!Serial` — that blocks until the host opens the port, which often
    // means the monitor misses the first INFO/ERR lines.
    delay(300);

    LOG_INFO(F("booting"));

    // Create inter-task synchronisation objects before spawning tasks.
    frameMutex    = xSemaphoreCreateMutex();
    frameReadySem = xSemaphoreCreateBinary();
    HALT_ON_FAIL(frameMutex    != nullptr, F("failed to create frameMutex"));
    HALT_ON_FAIL(frameReadySem != nullptr, F("failed to create frameReadySem"));

    // Spawn tasks first so commsTask is responsive to PING immediately,
    // before senseTask finishes initialising the MLX90640 sensor.
    // Wire and mlx.begin() are called inside senseTask (Core 1) to keep all
    // I2C traffic on the same core.
    xTaskCreatePinnedToCore(commsTask, "comms", COMMS_STACK, nullptr, 2, nullptr, 0);
    xTaskCreatePinnedToCore(senseTask, "sense", SENSE_STACK, nullptr, 2, nullptr, 1);

    // READY means the device is up and commsTask can answer PING.
    // The sensor may still be initialising; query INFO.ready for status.
    Serial.println(F("READY"));
}

void loop() {
    // All work is delegated to pinned tasks. Suspend this task indefinitely
    // to avoid wasting Core 1 scheduler time.
    vTaskSuspend(nullptr);
}
