#include <math.h>
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
#include "esp_task_wdt.h"
#include "esp_heap_caps.h"
#include "dl_model_base.hpp"
#include "dl_tensor_base.hpp"

static const char *TAG = "AgroStake";

// ─── Constants ───────────────────────────────────────────────────────────────
#define IMG_H        96
#define IMG_W        96
#define IMG_C        3
#define IMG_SIZE     (IMG_H * IMG_W * IMG_C)   // 27,648 bytes
#define NUM_CLASSES  4
#define AP_SSID      "AgroStake"
#define AP_PASS      "agrostake2025"

static const char *CLASS_NAMES[NUM_CLASSES] = {
    "Black_rot", "Esca", "Healthy", "Leaf_blight"
};

// ─── Embedded Binaries ───────────────────────────────────────────────────────
extern const uint8_t model_data_start[] asm("_binary_TinyAgCNN_v2_s8_espdl_start");
extern const uint8_t model_data_end[]   asm("_binary_TinyAgCNN_v2_s8_espdl_end");
extern const char    index_html_start[] asm("_binary_index_html_start");
extern const char    index_html_end[]   asm("_binary_index_html_end");

// ─── Dual-Core Inference State ───────────────────────────────────────────────
// Core 0: HTTP server (receives image, sends response)
// Core 1: Inference task (owns the model during the inference window)
static SemaphoreHandle_t s_infer_ready = NULL;  // HTTP->Infer: image is in buffer
static SemaphoreHandle_t s_infer_done  = NULL;  // Infer->HTTP: result is in s_json
static uint8_t          *s_img_buf     = NULL;  // Shared PSRAM image buffer
static char              s_json[256]   = {};
static dl::Model        *s_model       = NULL;
static std::string       s_input_name;

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

// ─── Inference Task (pinned to Core 1) ───────────────────────────────────────
static void inference_task(void *arg)
{
    // Persistent quantized buffer in PSRAM (allocated once, reused every inference)
    int8_t *input_i8 = (int8_t *)heap_caps_aligned_alloc(32, IMG_SIZE, MALLOC_CAP_SPIRAM);
    if (!input_i8) {
        ESP_LOGE(TAG, "PSRAM alloc failed for input_i8");
        vTaskDelete(NULL);
        return;
    }

    while (true) {
        // Block here until HTTP handler deposits a new image
        xSemaphoreTake(s_infer_ready, portMAX_DELAY);

        // Quantize: raw uint8 RGB -> int8 (subtract 128 to match exponent -7)
        for (int i = 0; i < IMG_SIZE; i++) {
            input_i8[i] = (int8_t)((int)s_img_buf[i] - 128);
        }

        // Build input tensor — NHWC layout, exponent -7 (scale = 2^-7 = 1/128)
        dl::TensorBase *input_tensor = new dl::TensorBase(
            {1, IMG_H, IMG_W, IMG_C}, input_i8, -7, dl::DATA_TYPE_INT8
        );
        std::map<std::string, dl::TensorBase *> inputs = {{s_input_name, input_tensor}};

        int64_t t0 = esp_timer_get_time();
        esp_task_wdt_reset();
        s_model->run(inputs);
        int64_t elapsed_ms = (esp_timer_get_time() - t0) / 1000;

        // Find argmax over raw INT8 scores
        dl::TensorBase *output  = s_model->get_outputs().begin()->second;
        int8_t         *scores  = (int8_t *)output->get_element_ptr();
        int             max_idx = 0;
        int8_t          max_val = scores[0];
        for (int i = 1; i < NUM_CLASSES; i++) {
            if (scores[i] > max_val) { max_val = scores[i]; max_idx = i; }
        }

        int confidence = confidence_percent(scores, NUM_CLASSES);

        snprintf(s_json, sizeof(s_json),
            "{\"class\":\"%s\",\"confidence\":%d,\"time_ms\":%lld}",
            CLASS_NAMES[max_idx], confidence, elapsed_ms);

        ESP_LOGI(TAG, "Result: %s | conf=%d%% | %lldms",
            CLASS_NAMES[max_idx], confidence, elapsed_ms);

        delete input_tensor;

        // Unblock the HTTP handler — it can now read s_json and send the response
        xSemaphoreGive(s_infer_done);
    }
}

// ─── HTTP Handler: GET / ──────────────────────────────────────────────────────
static esp_err_t index_get_handler(httpd_req_t *req)
{
    httpd_resp_set_type(req, "text/html");
    httpd_resp_send(req, index_html_start, index_html_end - index_html_start);
    return ESP_OK;
}

// ─── HTTP Handler: POST /detect ───────────────────────────────────────────────
static esp_err_t detect_post_handler(httpd_req_t *req)
{
    // Strict size check — phone must have already resized to 96x96
    if (req->content_len != IMG_SIZE) {
        ESP_LOGW(TAG, "Bad payload: got %d bytes, need %d", req->content_len, IMG_SIZE);
        httpd_resp_set_status(req, "400 Bad Request");
        httpd_resp_set_type(req, "application/json");
        httpd_resp_send(req, "{\"error\":\"Expected 27648 bytes (96x96 RGB888)\"}", HTTPD_RESP_USE_STRLEN);
        return ESP_FAIL;
    }

    // Stream directly into shared PSRAM buffer
    int received = 0;
    while (received < IMG_SIZE) {
        int ret = httpd_req_recv(req,
                                 (char *)s_img_buf + received,
                                 IMG_SIZE - received);
        if (ret == HTTPD_SOCK_ERR_TIMEOUT) continue;
        if (ret <= 0) {
            httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "Receive failed");
            return ESP_FAIL;
        }
        received += ret;
    }

    ESP_LOGI(TAG, "Image received (%d B) — dispatching to Core 1", received);

    // Signal Core 1 and wait up to 5s for inference result
    xSemaphoreGive(s_infer_ready);
    if (xSemaphoreTake(s_infer_done, pdMS_TO_TICKS(5000)) != pdTRUE) {
        httpd_resp_set_status(req, "500 Internal Server Error");
        httpd_resp_set_type(req, "application/json");
        httpd_resp_send(req, "{\"error\":\"Inference timeout\"}", HTTPD_RESP_USE_STRLEN);
        return ESP_FAIL;
    }

    // s_json was written by Core 1 — safe to read now
    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    httpd_resp_send(req, s_json, HTTPD_RESP_USE_STRLEN);
    return ESP_OK;
}

// ─── HTTP Server Init ─────────────────────────────────────────────────────────
// ─── HTTP Server Init ─────────────────────────────────────────────────────────
static httpd_handle_t start_webserver(void)
{
    httpd_config_t config    = HTTPD_DEFAULT_CONFIG();
    config.server_port       = 80;
    config.core_id           = 0;       // HTTP server pinned to Core 0
    config.stack_size        = 8192;

    httpd_handle_t server = NULL;
    if (httpd_start(&server, &config) != ESP_OK) {
        ESP_LOGE(TAG, "Failed to start HTTP server");
        return NULL;
    }

    static const httpd_uri_t index_uri = {
        .uri       = "/",
        .method    = HTTP_GET,
        .handler   = index_get_handler,
        .user_ctx  = NULL
    };

    static const httpd_uri_t detect_uri = {
        .uri       = "/detect",
        .method    = HTTP_POST,
        .handler   = detect_post_handler,
        .user_ctx  = NULL
    };

    httpd_register_uri_handler(server, &index_uri);
    httpd_register_uri_handler(server, &detect_uri);

    ESP_LOGI(TAG, "HTTP server running on Core 0 — port 80");
    return server;
}
// ─── Wi-Fi SoftAP Init ───────────────────────────────────────────────────────
static void wifi_init_softap(void)
{
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_ap();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    wifi_config_t ap_config     = {};
    ap_config.ap.ssid_len       = strlen(AP_SSID);
    ap_config.ap.channel        = 1;
    ap_config.ap.max_connection = 2;
    ap_config.ap.authmode       = WIFI_AUTH_WPA2_PSK;
    strncpy((char *)ap_config.ap.ssid,     AP_SSID, sizeof(ap_config.ap.ssid));
    strncpy((char *)ap_config.ap.password, AP_PASS,  sizeof(ap_config.ap.password));

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_AP));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_AP, &ap_config));
    ESP_ERROR_CHECK(esp_wifi_start());

    ESP_LOGI(TAG, "SoftAP: SSID=\"%s\" | IP=192.168.4.1", AP_SSID);
}

// ─── Entry Point ─────────────────────────────────────────────────────────────
extern "C" void app_main(void)
{
    // NVS is required by esp_wifi — erase if partition is corrupt
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    // Semaphores for Core 0 <-> Core 1 handshake
    s_infer_ready = xSemaphoreCreateBinary();
    s_infer_done  = xSemaphoreCreateBinary();

    // Shared PSRAM image buffer (32-byte aligned for S3 cache line)
    s_img_buf = (uint8_t *)heap_caps_aligned_alloc(32, IMG_SIZE, MALLOC_CAP_SPIRAM);
    if (!s_img_buf) {
        ESP_LOGE(TAG, "CRITICAL: PSRAM alloc failed for image buffer"); return;
    }

    // Copy model from Flash rodata into PSRAM
    size_t   model_size = model_data_end - model_data_start;
    uint8_t *model_buf  = (uint8_t *)heap_caps_aligned_alloc(32, model_size, MALLOC_CAP_SPIRAM);
    if (!model_buf) {
        ESP_LOGE(TAG, "CRITICAL: PSRAM alloc failed for model (%u B)", (unsigned)model_size); return;
    }
    memcpy(model_buf, model_data_start, model_size);

    s_model = new dl::Model((const char *)model_buf,
                            fbs::MODEL_LOCATION_IN_FLASH_RODATA,
                            0,
                            dl::MEMORY_MANAGER_GREEDY);

    if (s_model->get_inputs().empty()) {
        ESP_LOGE(TAG, "CRITICAL: Model load failed"); return;
    }
    s_input_name = s_model->get_inputs().begin()->first;
    ESP_LOGI(TAG, "Model loaded | %u bytes | input=%s",
        (unsigned)model_size, s_input_name.c_str());

    // Inference task pinned to Core 1, priority 5
    xTaskCreatePinnedToCore(inference_task, "infer_task", 8192, NULL, 5, NULL, 1);

    // SoftAP must be up before HTTP server starts
    wifi_init_softap();
    start_webserver();

    ESP_LOGI(TAG, "================================================");
    ESP_LOGI(TAG, "AgroStake Web Service READY");
    ESP_LOGI(TAG, "Wi-Fi: \"%s\" / \"%s\"", AP_SSID, AP_PASS);
    ESP_LOGI(TAG, "URL:   http://192.168.4.1");
    ESP_LOGI(TAG, "================================================");
}