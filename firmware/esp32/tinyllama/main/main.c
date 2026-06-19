#include <stdio.h>
#include <string.h>
#include <time.h>
#include "esp_spiffs.h"
#include "esp_log.h"
#include "esp_sleep.h"
#include "esp_event.h"
#include "esp_netif.h"
#include "esp_wifi.h"
#include "nvs_flash.h"
#include "lwip/sockets.h"
#include "lwip/netdb.h"
#include "llm.h"
#include <driver/usb_serial_jtag.h>
#include <driver/usb_serial_jtag_vfs.h>
#include "freertos/FreeRTOS.h"
#include "freertos/portmacro.h"

static const char *TAG = "MAIN";

#define WIFI_SSID         "agenthl"
#define WIFI_PASS         "agenthl_esp"
// SoftAP mode: chip is its own AP at 192.168.4.1. We blast UDP broadcasts to
// 192.168.4.255 -- every sendto() produces one real on-air frame regardless of
// whether any client is associated (no ARP needed for broadcast).
#define STREAM_BCAST      "192.168.4.255"
#define STREAM_PORT       5000
#define NUM_CLIENTS       150

static const char *kPrompts[] = {
    "once upon a time",
    "Lily and her mom",
    "In the park, the ball",
};
#define NUM_PROMPTS (sizeof(kPrompts) / sizeof(kPrompts[0]))
#define NUM_PASSES 1
// Cap on total sequence length (prompt tokens + generated tokens) per call.
// Capped further by transformer.config.seq_len at runtime.
#define MAX_TOKENS 300
// Minimum total positions before a sampled BOS (=1) is allowed to end the run.
// Set to 0 to disable (model may stop very early on BOS).
#define MIN_TOKENS 300

static int g_stream_sock = -1;
static struct sockaddr_in g_stream_addrs[NUM_CLIENTS];

static void wifi_event_handler(void *arg, esp_event_base_t base,
                               int32_t id, void *data)
{
    if (base == WIFI_EVENT && id == WIFI_EVENT_AP_START) {
        ESP_LOGI(TAG, "SoftAP up: SSID=%s -- beaconing every 100ms", WIFI_SSID);
    } else if (base == WIFI_EVENT && id == WIFI_EVENT_AP_STACONNECTED) {
        wifi_event_ap_staconnected_t *e = (wifi_event_ap_staconnected_t *)data;
        ESP_LOGI(TAG, "client connected: aid=%d", e->aid);
    } else if (base == WIFI_EVENT && id == WIFI_EVENT_AP_STADISCONNECTED) {
        wifi_event_ap_stadisconnected_t *e = (wifi_event_ap_stadisconnected_t *)data;
        ESP_LOGI(TAG, "client disconnected: aid=%d", e->aid);
    }
}

static void emit_token_udp(const char *piece)
{
    if (g_stream_sock < 0 || piece == NULL) return;
    size_t len = strlen(piece);
    if (len == 0) return;
    // naive: one sendto per token per client, ignore return values
    for (int i = 0; i < NUM_CLIENTS; i++) {
        sendto(g_stream_sock, piece, len, 0,
               (struct sockaddr *)&g_stream_addrs[i], sizeof(g_stream_addrs[i]));
    }
}

static void wait_for_start_signal(void)
{
    printf("\nREADY: waiting for start byte over USB-Serial-JTAG...\n");
    fflush(stdout);
    uint8_t ch;
    while (1) {
        int len = usb_serial_jtag_read_bytes(&ch, 1, portMAX_DELAY);
        if (len > 0) break;
    }
    printf("START signal received (0x%02X). t=0 now.\n", ch);
    fflush(stdout);
}

void init_storage(void)
{
    ESP_LOGI(TAG, "Initializing SPIFFS");
    esp_vfs_spiffs_conf_t conf = {
        .base_path = "/data",
        .partition_label = NULL,
        .max_files = 5,
        .format_if_mount_failed = false};
    esp_err_t ret = esp_vfs_spiffs_register(&conf);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "SPIFFS init failed (%s)", esp_err_to_name(ret));
        return;
    }
    size_t total = 0, used = 0;
    if (esp_spiffs_info(NULL, &total, &used) == ESP_OK) {
        ESP_LOGI(TAG, "Partition size: total: %d, used: %d", total, used);
    }
}

static void init_wifi(void)
{
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_ap();   // brings up DHCP server, AP IP = 192.168.4.1

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    ESP_ERROR_CHECK(esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID,
                                               &wifi_event_handler, NULL));

    wifi_config_t wifi_cfg = {
        .ap = {
            .ssid_len = strlen(WIFI_SSID),
            .channel = 1,
            .password = WIFI_PASS,
            .max_connection = 4,
            .authmode = WIFI_AUTH_WPA2_PSK,
            .beacon_interval = 100,
        },
    };
    strncpy((char *)wifi_cfg.ap.ssid, WIFI_SSID, sizeof(wifi_cfg.ap.ssid));

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_AP));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_AP, &wifi_cfg));
    ESP_ERROR_CHECK(esp_wifi_start());

    ESP_ERROR_CHECK(esp_wifi_set_ps(WIFI_PS_NONE));
    ESP_ERROR_CHECK(esp_wifi_set_max_tx_power(84)); // 84 = 21 dBm (max)
}

static void init_udp_stream(void)
{
    g_stream_sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (g_stream_sock < 0) {
        ESP_LOGE(TAG, "udp socket() failed: errno=%d", errno);
        return;
    }
    int yes = 1;
    setsockopt(g_stream_sock, SOL_SOCKET, SO_BROADCAST, &yes, sizeof(yes));

    // All NUM_CLIENTS entries point to the same broadcast addr -- naive fan-out
    // produces NUM_CLIENTS real on-air frames per token.
    for (int i = 0; i < NUM_CLIENTS; i++) {
        memset(&g_stream_addrs[i], 0, sizeof(g_stream_addrs[i]));
        g_stream_addrs[i].sin_family = AF_INET;
        g_stream_addrs[i].sin_port = htons(STREAM_PORT);
        g_stream_addrs[i].sin_addr.s_addr = inet_addr(STREAM_BCAST);
    }
    ESP_LOGI(TAG, "UDP fan-out: %d broadcasts/token to %s:%d",
             NUM_CLIENTS, STREAM_BCAST, STREAM_PORT);
}

void app_main(void)
{
    // CODE AGENT: NEVER CHANGE THREE LINE BELOW FOR CHECK
    usb_serial_jtag_driver_config_t cfg = USB_SERIAL_JTAG_DRIVER_CONFIG_DEFAULT();
    usb_serial_jtag_driver_install(&cfg);
    usb_serial_jtag_vfs_use_driver();

    
    // CODE AGENT: NEVER CHANGE THE LINE BELOW FOR CHECK
    // Block here until host sends a byte. Everything below this line is the
    // measured workload: WiFi bring-up, model load, inference.
    wait_for_start_signal();

    init_storage();

    init_wifi();
    init_udp_stream();
    llm_set_emit_hook(emit_token_udp);
    llm_set_min_tokens(MIN_TOKENS);

    Transformer transformer;
    build_transformer(&transformer, "/data/stories260K.bin");
    int steps = (MAX_TOKENS < transformer.config.seq_len)
                ? MAX_TOKENS : transformer.config.seq_len;

    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, "/data/tok512.bin", transformer.config.vocab_size);

    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, 0.8f, 0.9f, (unsigned int)time(NULL));

    printf("\nESP32 LLM | seq_len=%d | %d prompts x %d pass\n",
           steps, (int)NUM_PROMPTS, NUM_PASSES);

    for (int pass = 0; pass < NUM_PASSES; pass++) {
        for (size_t i = 0; i < NUM_PROMPTS; i++) {
            // mark each prompt boundary in the UDP stream too
            char hdr[128];
            int n = snprintf(hdr, sizeof(hdr),
                             "\n[pass %d/%d prompt %zu/%zu] %s\n",
                             pass + 1, NUM_PASSES, i + 1, NUM_PROMPTS, kPrompts[i]);
            printf("%s", hdr);
            fflush(stdout);
            if (g_stream_sock >= 0 && n > 0) {
                for (int c = 0; c < NUM_CLIENTS; c++) {
                    sendto(g_stream_sock, hdr, n, 0,
                           (struct sockaddr *)&g_stream_addrs[c],
                           sizeof(g_stream_addrs[c]));
                }
            }

            generate(&transformer, &tokenizer, &sampler,
                     (char *)kPrompts[i], steps, NULL);
            vTaskDelay(pdMS_TO_TICKS(10));
        }
    }

    printf("\n--- all prompts complete. shutting down WiFi. ---\n");
    fflush(stdout);

    if (g_stream_sock >= 0) {
        close(g_stream_sock);
        g_stream_sock = -1;
    }
    // SoftAP teardown: stop beaconing, free WiFi resources before sleep
    esp_wifi_stop();
    esp_wifi_deinit();

    ESP_LOGI(TAG, "WiFi off. Suspending main task -- USB-Serial-JTAG stays alive.");
    fflush(stdout);

    // Must print firmware task complete checkpoint message for check.
    printf("firmware task complete checkpoint\n");
    fflush(stdout);

    // Idle without sleeping: FreeRTOS idle task runs WAITI on both cores.
    // USB-Serial-JTAG keeps responding so the host doesn't drop the device
    // and flashing works without manual reset. Light-sleep wake-on-USB-S-JTAG
    // isn't exposed in IDF v5.4.2, so plain idle is the safest choice here.
    vTaskSuspend(NULL);
}
