
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include "mxc.h"
#include "cnn.h"
#include "mxc_errors.h"
#include "camera.h"
#include "post_process.h"
#include "led.h"
#include "board.h"
#define DELAY_IN_SEC 1

#define CAMERA_FREQ 10000000

volatile uint32_t cnn_time; // Stopwatch
extern volatile uint8_t face_detected;
volatile int alarmed;
volatile uint32_t frame_count = 0;
void low_power_init();

// RTC alarm handler
void alarmHandler(void)
{
    int flags = MXC_RTC->ctrl;
    alarmed = 1;

    if ((flags & MXC_F_RTC_CTRL_SSEC_ALARM) >> MXC_F_RTC_CTRL_SSEC_ALARM_POS) {
        MXC_RTC->ctrl &= ~(MXC_F_RTC_CTRL_SSEC_ALARM);
    }

    if ((flags & MXC_F_RTC_CTRL_TOD_ALARM) >> MXC_F_RTC_CTRL_TOD_ALARM_POS) {
        MXC_RTC->ctrl &= ~(MXC_F_RTC_CTRL_TOD_ALARM);
    }
}
 // Sets the RTC alarm that will be used as the wakeup source
 void setTrigger(int delay)
 {
     alarmed = 0;
 
     while (MXC_RTC_Init(0, 0) == E_BUSY) {}
 
     while (MXC_RTC_DisableInt(MXC_F_RTC_CTRL_TOD_ALARM_IE) == E_BUSY) {}
 
     while (MXC_RTC_SetTimeofdayAlarm(delay) == E_BUSY) {}
 
     while (MXC_RTC_EnableInt(MXC_F_RTC_CTRL_TOD_ALARM_IE) == E_BUSY) {}
 
     while (MXC_RTC_Start() == E_BUSY) {}
 }

void fail(void)
{
    printf("\n*** FAIL ***\n\n");
    while (1) {}
}

int MXC_UART_WriteBytes(mxc_uart_regs_t *uart, const uint8_t *bytes, int len)
{
    int err = E_NO_ERROR;
    for (int i = 0; i < len; i++) {
        // Wait until FIFO has space for the character.
        while (MXC_UART_GetTXFIFOAvailable(uart) < 1) {}

        if ((err = MXC_UART_WriteCharacterRaw(uart, bytes[i])) != E_NO_ERROR) {
            return err;
        }
    }

    return E_NO_ERROR;
}

// Data input: HWC 3x224x168 (112896 bytes total / 37632 bytes per channel):
void load_input(void)
{
    uint8_t *raw;
    uint32_t imglen, w, h;

    camera_sleep(0); // Disable sleep mode.

    camera_start_capture_image();
    while (!camera_is_image_rcv()) {}

    camera_sleep(1); // Enable sleep mode.

    camera_get_image(&raw, &imglen, &w, &h);
    uint8_t ur = 0, ug = 0, ub = 0;
    int8_t r = 0, g = 0, b = 0;
    uint32_t rgb888 = 0;

    for (unsigned int i = 0; i < imglen; i += 2) {
        // Decode RGB565
        ur = (raw[i] & 0xF8) >> 3;
        ug = (raw[i] & 0b111) << 3;
        ug |= (raw[i + 1] & 0xE0) >> 5;
        ub = (raw[i + 1] & 0x1F);

        // Convert to RGB888
        ur = ur << 3;
        ug = ug << 2;
        ub = ub << 3;

        // Normalize from [0, 255] -> [-128, 127]
        r = ur - 128;
        g = ug - 128;
        b = ub - 128;

        // Pack to RGB888 (0x00BBGGRR)
        rgb888 = r | (g << 8) | (b << 16);

        // Loading data into the CNN fifo
        while (((*((volatile uint32_t *)0x50000004) & 1)) != 0) {}
        // Wait for FIFO 0
        *((volatile uint32_t *)0x50000008) = rgb888; // Write FIFO 0
    }
}

int main(void)
{
    int ret = 0, id = 0;
    MXC_ICC_Enable(MXC_ICC0); // Enable cache

    // Switch to 100 MHz clock
    MXC_SYS_Clock_Select(MXC_SYS_CLOCK_ISO);
    SystemCoreClockUpdate();
    MXC_NVIC_SetVector(RTC_IRQn, alarmHandler);
    MXC_LP_EnableRTCAlarmWakeup();
    printf("Waiting...\n");

    // DO NOT DELETE THIS LINE:
    MXC_Delay(SEC(2)); // Let debugger interrupt if needed

    // Initialize DMA and acquire a channel for the camera interface to use
    printf("Initializing DMA\n");
    MXC_DMA_Init();
    int dma_channel = MXC_DMA_AcquireChannel();

    // Initialize the camera driver.
    printf("Initializing camera\n");
    camera_init(CAMERA_FREQ);

    int slaveAddress = camera_get_slave_address();
    printf("Camera I2C slave address: %02x\n", slaveAddress);

    ret = camera_get_manufacture_id(&id);
    if (ret != STATUS_OK) {
        printf("Error returned from reading camera id. Error %d\n", ret);
        return -1;
    }
    printf("Camera ID detected: %04x\n", id);

    camera_set_hmirror(0);
    camera_set_vflip(0);

    ret = camera_setup(IMAGE_SIZE_X, // width
                       IMAGE_SIZE_Y, // height
                       PIXFORMAT_RGB565, // pixel format
                       FIFO_FOUR_BYTE, // FIFO mode (four bytes is suitable for most cases)
                       USE_DMA, // DMA (enabling DMA will drastically decrease capture time)
                       dma_channel); // Allocate the DMA channel retrieved in initialization

    if (ret != E_NO_ERROR) {
        printf("Failed to setup camera!\n");
        return ret;
    }

    printf("\n*** CNN Inference Test facedet_tinierssd ***\n");
    setTrigger(DELAY_IN_SEC);
    MXC_LP_EnterMicroPowerMode();

    while (1) {
        face_detected = 0;
        LED_On(1);

        // Switch to high power fast IPO clock
        MXC_SYS_ClockEnable(MXC_SYS_CLOCK_ISO);
        MXC_SYS_Clock_Select(MXC_SYS_CLOCK_ISO);

        // Enable peripheral, enable CNN interrupt, turn on CNN clock
        // CNN clock: APB (50 MHz) div 1
        cnn_enable(MXC_F_GCR_PCLKDIV_CNNCLKSEL, MXC_S_GCR_PCLKDIV_CNNCLKDIV_DIV1);
        cnn_init(); // Bring state machine into consistent state
        cnn_load_weights(); // Load kernels
        cnn_load_bias();
        cnn_configure(); // Configure state machine
        cnn_start(); // Start CNN processing
        load_input(); // Load data input via FIFO

        while (cnn_time == 0) MXC_LP_EnterSleepMode(); // Wait for CNN

        // Run Non-Maximal Suppression (NMS) on bounding boxes
        get_priors();
        localize_objects();
        LED_Off(1);

        if (!face_detected) {
            LED_Off(0);
        } else {
            LED_On(0);
        }

        cnn_disable(); // Shut down CNN clock, disable peripheral

        // Switch to low power IBRO clock
        MXC_SYS_ClockEnable(MXC_SYS_CLOCK_IBRO);
        MXC_SYS_Clock_Select(MXC_SYS_CLOCK_IBRO);
        MXC_SYS_ClockDisable(MXC_SYS_CLOCK_ISO);

        printf("firmware task complete checkpoint\n");

        MXC_Delay(MXC_DELAY_MSEC(200)); // Slight delay to allow LED to be seen

#ifdef CNN_INFERENCE_TIMER
        printf("Approximate data loading and inference time: %u us\n\n", cnn_time);
#endif

        LED_Off(0);
        setTrigger(DELAY_IN_SEC);
        MXC_LP_EnterMicroPowerMode();


    }

    return 0;
}
