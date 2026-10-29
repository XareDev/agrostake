#include <math.h>
#include <stdio.h>
#include <string.h>
#include <map>
#include <string>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/uart.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "dl_model_base.hpp"
#include "dl_tensor_base.hpp"
#include "fbs_model.hpp"

static const char *TAG = "AgroStake";

#define IMG_H       96
#define IMG_W       96
#define IMG_C       1
#define NUM_CLASSES 4
#define IMG_SIZE    (IMG_H * IMG_W * IMG_C)

const char *CLASS_NAMES[NUM_CLASSES] = {
    "Black_rot", "Esca", "Healthy", "Leaf_blight"
};

#define UART_NUM    UART_NUM_0
#define BUF_SIZE    (IMG_SIZE + 64)

// Force 16-byte alignment of the embedded binary to silence FbsLoader warnings.
extern const uint8_t model_data_start[] asm("_binary_TinyAgCNN_s8_espdl_start") __attribute__((aligned(16)));
extern const uint8_t model_data_end[]   asm("_binary_TinyAgCNN_s8_espdl_end");

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
    // UART Config
    uart_config_t uart_cfg = {};
    uart_cfg.baud_rate  = 115200;
    uart_cfg.data_bits  = UART_DATA_8_BITS;
    uart_cfg.parity     = UART_PARITY_DISABLE;
    uart_cfg.stop_bits  = UART_STOP_BITS_1;
    uart_cfg.flow_ctrl  = UART_HW_FLOWCTRL_DISABLE;
    uart_cfg.source_clk = UART_SCLK_DEFAULT;

    uart_param_config(UART_NUM, &uart_cfg);
    uart_driver_install(UART_NUM, BUF_SIZE * 2, 0, 0, NULL, 0);

    size_t model_size = model_data_end - model_data_start;
    ESP_LOGI(TAG, "Model size: %d bytes", (int)model_size);

    dl::Model *model = new dl::Model(
        (const char *)model_data_start,
        fbs::MODEL_LOCATION_IN_FLASH_RODATA,
        0,
        dl::MEMORY_MANAGER_GREEDY
    );

    ESP_LOGI(TAG, "Model loaded. Ready.");
    uart_write_bytes(UART_NUM, "READY\n", 6);

    // Keep the large image buffers in PSRAM so internal RAM stays available
    // for the Wi-Fi/FreeRTOS stacks.
    uint8_t *raw = (uint8_t *)heap_caps_aligned_alloc(32, IMG_SIZE, MALLOC_CAP_SPIRAM);
    int8_t *input_int8 = (int8_t *)heap_caps_aligned_alloc(32, IMG_SIZE, MALLOC_CAP_SPIRAM);
    if (!raw || !input_int8) {
        ESP_LOGE(TAG, "PSRAM allocation failed");
        return;
    }

    while (true) {
        uint8_t magic = 0;
        while (magic != 0xAA) {
            uart_read_bytes(UART_NUM, &magic, 1, portMAX_DELAY);
        }

        int received = 0;
        while (received < IMG_SIZE) {
            int n = uart_read_bytes(UART_NUM,
                                    raw + received,
                                    IMG_SIZE - received,
                                    pdMS_TO_TICKS(5000));
            if (n <= 0) break; 
            received += n;
        }

        if (received < IMG_SIZE) {
            uart_write_bytes(UART_NUM, "ERR:TIMEOUT\n", 12);
            continue;
        }

        // The host sends already-shifted INT8 bytes (pixel - 128), matching
        // the exponent -7 input contract.
        memcpy(input_int8, raw, IMG_SIZE);

        dl::TensorBase *input_tensor = new dl::TensorBase(
            {1, IMG_H, IMG_W, IMG_C},
            input_int8,
            -7, 
            dl::DATA_TYPE_INT8
        );

        // Get the actual name of the first input from the model itself
        std::string input_name = model->get_inputs().begin()->first;
        ESP_LOGI(TAG, "Using input name: %s", input_name.c_str());

        std::map<std::string, dl::TensorBase *> inputs = {
            {input_name, input_tensor}
        };

        // Inference Timing
        int64_t start_time = esp_timer_get_time();
        model->run(inputs);
        int64_t end_time = esp_timer_get_time();

        dl::TensorBase *output = model->get_outputs().begin()->second;
        int8_t *scores = (int8_t *)output->get_element_ptr();

        int   max_idx   = 0;
        int8_t max_score = scores[0];
        for (int i = 1; i < NUM_CLASSES; i++) {
            if (scores[i] > max_score) {
                max_score = scores[i];
                max_idx   = i;
            }
        }

        char result[160];
        snprintf(result, sizeof(result),
                 "CLASS:%s SCORE:%d CONF:%d TIME:%lldms\n",
                 CLASS_NAMES[max_idx], (int)max_score,
                 confidence_percent(scores, NUM_CLASSES),
                 (end_time - start_time) / 1000);

        uart_write_bytes(UART_NUM, result, strlen(result));
        ESP_LOGI(TAG, "%s", result);

        delete input_tensor;
    }
}