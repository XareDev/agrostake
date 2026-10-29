idf_component_register(
    SRCS "main.cpp"
    INCLUDE_DIRS "."
    PRIV_REQUIRES esp-dl esp_driver_uart esp_driver_gpio esp_timer esp_hw_support esp_system heap
)

# Standard command (if the aligned version is unknown)
set(MODEL_FILE "${CMAKE_SOURCE_DIR}/models/TinyAgCNN_v2_s8.espdl")
target_add_binary_data(${COMPONENT_LIB} "${MODEL_FILE}" BINARY)