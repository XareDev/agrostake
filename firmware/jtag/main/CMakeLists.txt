# Define the source files
set(srcs "main.cpp")

# Register the component
idf_component_register(
    SRCS "${srcs}"
    INCLUDE_DIRS "."
    # Use 'esp_driver_usb_serial_jtag' instead of just 'usb_serial_jtag'
    PRIV_REQUIRES esp-dl esp_driver_usb_serial_jtag esp_timer esp_hw_support esp_system heap
)

# Embed the binary model file
set(MODEL_FILE "${CMAKE_SOURCE_DIR}/models/TinyAgCNN_v2_s8.espdl")
target_add_binary_data(${COMPONENT_LIB} "${MODEL_FILE}" BINARY)