#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <map>
#include <string>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "nvs_flash.h"
#include "esp_netif.h"
#include "esp_http_server.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_system.h"
#include "esp_heap_caps.h"
#include "esp_cpu.h"
#include "hal/usb_serial_jtag_ll.h"

// --- ESP-DL Specifics ---
#include "dl_model_base.hpp"
#include "dl_tensor_base.hpp"

static const char *TAG = "AgroStake_S3";

// --- Constants ---
#define IMG_H        96
#define IMG_W        96
#define IMG_C        3
#define IMG_SIZE     (IMG_H * IMG_W * IMG_C)
#define NUM_CLASSES  4
#define AP_SSID      "AgroStake_Field"
#define AP_PASS      "agrostake2025"

static const char *CLASS_NAMES[NUM_CLASSES] = {
    "Black_rot", "Esca", "Healthy", "Leaf_blight"
};

// --- Global State ---
static SemaphoreHandle_t s_infer_ready = NULL;
static SemaphoreHandle_t s_infer_done  = NULL;
static uint8_t          *s_img_buf     = NULL;
static char              s_result_json[256] = {};
static dl::Model        *s_model       = NULL;
static std::string       s_input_name;

// --- Embedded Binaries ---
extern const uint8_t model_data_start[] asm("_binary_TinyAgCNN_v2_s8_espdl_start");
extern const uint8_t model_data_end[]   asm("_binary_TinyAgCNN_v2_s8_espdl_end");
extern const char    index_html_start[] asm("_binary_index_html_start");
extern const char    index_html_end[]   asm("_binary_index_html_end");

// --- Forward Declarations ---
static httpd_handle_t start_webserver(void);

// ─── Safe Printf Wrapper ──────────────────────────────────────────────────────
/**
 * jtag_safe_printf() — prevents host-side driver hangs.
 * If JTAG is attached, it flushes the TX FIFO to clear data stuck from 
 * the previous CPU halt.
 */
static void jtag_safe_printf(const char *fmt, ...)
{
    if (esp_cpu_dbgr_is_attached()) {
        // Flush stale data from previous halt
        usb_serial_jtag_ll_txfifo_flush();

        // If FIFO is full, the host isn't reading; skip to avoid blocking
        if (!usb_serial_jtag_ll_txfifo_writable()) {
            return;
        }
    }

    char buf[256];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    printf("%s", buf);
}

// Softmax over the INT8 logits. The result is a relative confidence, not a
// calibrated probability; the ESP-DL output scale is intentionally ignored.
static int confidence_percent(const int8_t *scores, int n)
{
    int max_idx = 0;
    for (int i = 1; i < n; i++) {
        if (scores[i] > scores[max_idx]) max_idx = i;
    }
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += expf((float)(scores[i] - scores[max_idx]));
    }
    return (int)lroundf(100.0f / sum);
}

// --- Inference Task (Pinned to Core 1) ---
static void inference_task(void *arg) {
    int8_t *input_i8 = (int8_t *)heap_caps_aligned_alloc(32, IMG_SIZE, MALLOC_CAP_SPIRAM);
    if (!input_i8) {
        ESP_LOGE(TAG, "PSRAM allocation failed for input buffer");
        vTaskDelete(NULL);
        return;
    }

    while (true) {
        xSemaphoreTake(s_infer_ready, portMAX_DELAY);

        for (int i = 0; i < IMG_SIZE; i++) {
            input_i8[i] = (int8_t)((int)s_img_buf[i] - 128);
        }

        dl::TensorBase *input_tensor = new dl::TensorBase({1, IMG_H, IMG_W, IMG_C}, input_i8, -7, dl::DATA_TYPE_INT8);
        std::map<std::string, dl::TensorBase *> inputs = {{s_input_name, input_tensor}};

        int64_t t0 = esp_timer_get_time();
        s_model->run(inputs);
        int64_t elapsed_ms = (esp_timer_get_time() - t0) / 1000;

        dl::TensorBase *output = s_model->get_outputs().begin()->second;
        int8_t *scores = (int8_t *)output->get_element_ptr();
        
        int max_idx = 0;
        int8_t max_val = scores[0];
        for (int i = 1; i < NUM_CLASSES; i++) {
            if (scores[i] > max_val) { max_val = scores[i]; max_idx = i; }
        }

        int conf = confidence_percent(scores, NUM_CLASSES);

        snprintf(s_result_json, sizeof(s_result_json),
                "{\"class\":\"%s\",\"confidence\":%d,\"time_ms\":%lld}",
                CLASS_NAMES[max_idx], conf, elapsed_ms);

        jtag_safe_printf("RESULT:class=%s,confidence=%d,time=%lld\n",
                        CLASS_NAMES[max_idx], conf, elapsed_ms);

        delete input_tensor;
        xSemaphoreGive(s_infer_done);
    }
}

// --- Command Handler (JTAG-Safe Logic) ---
// ─── Serial Handler Task - Supports Text Commands + Binary Image ─────────────
static void serial_handler_task(void *arg) {
    char line[128];
    int line_idx = 0;
    bool receiving_image = false;
    int bytes_received = 0;

    // Signal host that we are ready after boot
    jtag_safe_printf("READY\n"); 

    while (1) {
        if (receiving_image) {
            // High-speed binary ingestion
            int bytes_to_read = IMG_SIZE - bytes_received;
            int read_bytes = fread((char*)s_img_buf + bytes_received, 1, bytes_to_read, stdin);
            if (read_bytes > 0) {
                bytes_received += read_bytes;
                if (bytes_received >= IMG_SIZE) {
                    jtag_safe_printf("Image received\n");
                    xSemaphoreGive(s_infer_ready);
                    receiving_image = false;
                    bytes_received = 0;
                }
            }
        } 
        else {
            // Non-blocking Command Listen
            int c = fgetc(stdin);
            if (c != EOF) {
                if (c == '\n' || c == '\r') {
                    if (line_idx > 0) {
                        line[line_idx] = '\0';
                        if (strcmp(line, "IMAGE") == 0) {
                            jtag_safe_printf("READY_FOR_IMAGE\n");
                            receiving_image = true;
                        }
                        line_idx = 0;
                    }
                } else if (line_idx < sizeof(line) - 1) {
                    line[line_idx++] = (char)c;
                }
            }
        }
        vTaskDelay(pdMS_TO_TICKS(5)); // Maintain low latency
    }
}
static esp_err_t detect_post_handler(httpd_req_t *req) {
    if (req->content_len != IMG_SIZE) {
        httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "Invalid Image Size");
        return ESP_FAIL;
    }

    int rcvd = 0;
    while (rcvd < IMG_SIZE) {
        int ret = httpd_req_recv(req, (char *)s_img_buf + rcvd, IMG_SIZE - rcvd);
        if (ret <= 0) return ESP_FAIL;
        rcvd += ret;
    }

    xSemaphoreGive(s_infer_ready);
    xSemaphoreTake(s_infer_done, pdMS_TO_TICKS(5000));

    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    httpd_resp_send(req, s_result_json, strlen(s_result_json));
    return ESP_OK;
}

static esp_err_t index_get_handler(httpd_req_t *req) {
    httpd_resp_send(req, index_html_start, index_html_end - index_html_start);
    return ESP_OK;
}

static httpd_handle_t start_webserver(void) {
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    config.core_id = 0;
    httpd_handle_t server = NULL;

    if (httpd_start(&server, &config) == ESP_OK) {
        static const httpd_uri_t index_uri = { .uri = "/", .method = HTTP_GET, .handler = index_get_handler, .user_ctx = NULL };
        static const httpd_uri_t det_uri = { .uri = "/detect", .method = HTTP_POST, .handler = detect_post_handler, .user_ctx = NULL };
        httpd_register_uri_handler(server, &index_uri);
        httpd_register_uri_handler(server, &det_uri);
    }
    return server;
}

// --- App Main ---
extern "C" void app_main(void) {
    ESP_ERROR_CHECK(nvs_flash_init());
    
    // Ensure hardware FIFO is clean on boot
    usb_serial_jtag_ll_txfifo_flush();

    s_infer_ready = xSemaphoreCreateBinary();
    s_infer_done  = xSemaphoreCreateBinary();
    s_img_buf = (uint8_t *)heap_caps_aligned_alloc(32, IMG_SIZE, MALLOC_CAP_SPIRAM);
    if (!s_img_buf) {
        ESP_LOGE(TAG, "PSRAM allocation failed for image buffer");
        return;
    }

    size_t m_size = model_data_end - model_data_start;
    uint8_t *m_buf = (uint8_t *)heap_caps_aligned_alloc(32, m_size, MALLOC_CAP_SPIRAM);
    if (!m_buf) {
        ESP_LOGE(TAG, "PSRAM allocation failed for model");
        return;
    }
    memcpy(m_buf, model_data_start, m_size);

    s_model = new dl::Model((const char *)m_buf, fbs::MODEL_LOCATION_IN_FLASH_RODATA, 0, dl::MEMORY_MANAGER_GREEDY);
    s_input_name = s_model->get_inputs().begin()->first;

    xTaskCreatePinnedToCore(inference_task, "infer", 8192, NULL, 5, NULL, 1);
    xTaskCreate(serial_handler_task, "serial", 4096, NULL, 1, NULL);

    // WiFi AP Config
    esp_netif_init();
    esp_event_loop_create_default();
    esp_netif_create_default_wifi_ap();
    wifi_init_config_t w_cfg = WIFI_INIT_CONFIG_DEFAULT();
    esp_wifi_init(&w_cfg);
    
    wifi_config_t ap_cfg = {};
    strcpy((char*)ap_cfg.ap.ssid, AP_SSID);
    strcpy((char*)ap_cfg.ap.password, AP_PASS);
    ap_cfg.ap.authmode = WIFI_AUTH_WPA2_PSK;
    ap_cfg.ap.max_connection = 4;
    
    esp_wifi_set_mode(WIFI_MODE_AP);
    esp_wifi_set_config(WIFI_IF_AP, &ap_cfg);
    esp_wifi_start();
    
    start_webserver();

    // Final READY signal wrapped in safe printf
    jtag_safe_printf("READY\n"); 
}