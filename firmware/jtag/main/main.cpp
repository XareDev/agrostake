#include <math.h>
#include <stdio.h>
#include <string.h>
#include <map>
#include <string>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/usb_serial_jtag.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_task_wdt.h"
#include "dl_model_base.hpp"
#include "dl_tensor_base.hpp"

static const char *TAG = "AgroStake";

#define IMG_H       96
#define IMG_W       96
#define IMG_C       3
#define NUM_CLASSES 4
#define IMG_SIZE    (IMG_H * IMG_W * IMG_C)

static const char *CLASS_NAMES[NUM_CLASSES] = {
    "Black_rot", "Esca", "Healthy", "Leaf_blight"
};

extern const uint8_t model_data_start[] asm("_binary_TinyAgCNN_v2_s8_espdl_start") __attribute__((aligned(16)));
extern const uint8_t model_data_end[]   asm("_binary_TinyAgCNN_v2_s8_espdl_end");

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

extern "C" void app_main(void)
{
    // 1. USB Serial JTAG Init
    usb_serial_jtag_driver_config_t usb_cfg = {
        .tx_buffer_size = 256,
        .rx_buffer_size = IMG_SIZE * 2
    };
    
    // Official IDF v6 driver installation
    ESP_ERROR_CHECK(usb_serial_jtag_driver_install(&usb_cfg));
    ESP_LOGI(TAG, "USB Serial JTAG Driver Installed");

    // 2. Copy the embedded model into PSRAM (required by ESP-DL on the S3).
    size_t model_size = model_data_end - model_data_start;
    uint8_t *model_buf = (uint8_t *)heap_caps_aligned_alloc(32, model_size, MALLOC_CAP_SPIRAM);
    if (!model_buf) {
        ESP_LOGE(TAG, "PSRAM allocation failed for model");
        return;
    }
    memcpy(model_buf, model_data_start, model_size);

    // 3. Model Loading
    dl::Model *model = new dl::Model((const char *)model_buf, 
                                     fbs::MODEL_LOCATION_IN_FLASH_RODATA, 
                                     0, 
                                     dl::MEMORY_MANAGER_GREEDY);

    if (model->get_inputs().empty()) {
        ESP_LOGE(TAG, "MODEL LOADING FAILED!");
        while(1) vTaskDelay(pdMS_TO_TICKS(1000)); 
    }
    
    // 4. Image buffers (32-byte aligned for the S3 cache).
    uint8_t *raw = (uint8_t *)heap_caps_aligned_alloc(32, IMG_SIZE, MALLOC_CAP_SPIRAM);
    int8_t *input_i8 = (int8_t *)heap_caps_aligned_alloc(32, IMG_SIZE, MALLOC_CAP_SPIRAM);
    if (!raw || !input_i8) {
        ESP_LOGE(TAG, "PSRAM allocation failed for image buffers");
        return;
    }

    std::string input_name = model->get_inputs().begin()->first;
    ESP_LOGI(TAG, "AgroStake READY via USB-JTAG");

    // The console is on UART0, so ESP_LOGI() is not visible on the native USB
    // port. Send the handshake explicitly on the USB data channel.
    static const char ready[] = "READY\n";
    usb_serial_jtag_write_bytes(ready, sizeof(ready) - 1, pdMS_TO_TICKS(1000));

    while (true) {
        uint8_t magic = 0;
        // Wait for magic byte 0xAA
        while (magic != 0xAA) {
            usb_serial_jtag_read_bytes(&magic, 1, portMAX_DELAY);
        }

        int received = 0;
        while (received < IMG_SIZE) {
            // Read bulk image data
            int n = usb_serial_jtag_read_bytes(raw + received, IMG_SIZE - received, pdMS_TO_TICKS(5000));
            if (n <= 0) break;
            received += n;
        }

        if (received < IMG_SIZE) {
            ESP_LOGE(TAG, "USB Timeout: Received %d/%d", received, IMG_SIZE);
            continue;
        }

        // Copy raw buffer to input buffer
        memcpy(input_i8, raw, IMG_SIZE);

        // Build the input tensor (NHWC, exponent -7 = 1/128 scale).
        dl::TensorBase *input_tensor = new dl::TensorBase({1, IMG_H, IMG_W, IMG_C}, input_i8, -7, dl::DATA_TYPE_INT8);
        std::map<std::string, dl::TensorBase *> inputs = {{input_name, input_tensor}};

        int64_t t0 = esp_timer_get_time();
        esp_task_wdt_reset();
        model->run(inputs);
        int64_t elapsed_ms = (esp_timer_get_time() - t0) / 1000;

        // Process Results
        auto outputs = model->get_outputs();
        dl::TensorBase *output = outputs.begin()->second;
        int8_t *scores = (int8_t *)output->get_element_ptr();

        int max_idx = 0;
        int8_t max_score = scores[0];
        for (int i = 1; i < NUM_CLASSES; i++) {
            if (scores[i] > max_score) {
                max_score = scores[i];
                max_idx = i;
            }
        }

        char result[160];
        int len = snprintf(result, sizeof(result),
                           "CLASS:%s SCORE:%d CONF:%d TIME:%lldms\n",
                           CLASS_NAMES[max_idx], (int)max_score,
                           confidence_percent(scores, NUM_CLASSES), elapsed_ms);
        
        // Write results back through USB
        usb_serial_jtag_write_bytes(result, len, pdMS_TO_TICKS(100));

        delete input_tensor;
    }
}