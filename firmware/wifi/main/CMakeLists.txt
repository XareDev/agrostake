set(srcs "main.cpp")

idf_component_register(
    SRCS "${srcs}"
    INCLUDE_DIRS "."
    PRIV_REQUIRES
        esp-dl
        esp_timer
        esp_hw_support
        esp_wifi
        esp_http_server
        nvs_flash
        esp_netif
        esp_event
        esp_system
        heap
)

# Embed model binary (Flash rodata -> copied to PSRAM at boot)
set(MODEL_FILE "${CMAKE_SOURCE_DIR}/models/TinyAgCNN_v2_s8.espdl")
target_add_binary_data(${COMPONENT_LIB} "${MODEL_FILE}" BINARY)

# Embed the HTML frontend page
# Symbols: _binary_index_html_start / _binary_index_html_end
target_add_binary_data(${COMPONENT_LIB} "${CMAKE_CURRENT_SOURCE_DIR}/index.html" BINARY)