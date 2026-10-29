# Deployment

This document covers the ESP-IDF build, the INT8 model path, and the host
interfaces. It assumes you have a working ESP32-S3 toolchain; the firmware is
not built by this repository's Python tests.

## 1. Prerequisites

| Tool | Version / notes |
|---|---|
| ESP-IDF | ≥ 5.4 (the component manifest declares this) |
| ESP-DL | `espressif/esp-dl ^3.3.2` via `main/idf_component.yml` |
| Python | 3.10+ for the host tools |
| Board | ESP32-S3, 16 MB flash, 8 MB octal PSRAM recommended |

The projects use managed components, so ESP-IDF downloads ESP-DL on the first
build. You do not need to vendor `components/esp-dl` into each firmware folder.

```bash
# Linux/macOS
. $IDF_PATH/export.sh

# Windows PowerShell
. $env:IDF_PATH\export.ps1
```

## 2. Build a firmware variant

Each directory is an independent ESP-IDF project:

```bash
cd firmware/wifi
idf.py set-target esp32s3
idf.py build
idf.py -p COM7 flash monitor
```

Replace `wifi` with `uart-grayscale`, `uart-rgb`, `jtag`, or
`experimental/wifi-serial`. Build directories are ignored by Git.

### Default board profile

`firmware/*/sdkconfig.defaults` sets:

```text
16 MB QIO flash, 80 MHz
8 MB octal PSRAM, 80 MHz
240 MHz CPU
6 MB factory app partition, no OTA
console on USB-Serial-JTAG for UART/Wi-Fi variants
console on UART0 for the JTAG variant
```

If your board has different flash or PSRAM, change the defaults or run
`idf.py menuconfig`. In particular:

- `Component config → ESP PSRAM → SPI RAM config` must match the board
  (octal vs quad, speed).
- `Serial flasher config → Flash size` must match the physical flash.
- If you use a 4 MB board, replace `partitions.csv` with a layout that fits;
  the 6 MB factory partition will not fit.

The v2 firmware variants copy the embedded model into 32-byte-aligned PSRAM at
boot; `uart-grayscale` constructs its smaller model directly from flash rodata.
If PSRAM is missing or misconfigured, the v2 boot log will report an allocation
failure and the model will not run.

### Console routing

| Variant | Image transport | Console |
|---|---|---|
| `uart-grayscale` | UART0 | USB-Serial-JTAG |
| `uart-rgb` | UART0 | USB-Serial-JTAG |
| `jtag` | USB-Serial-JTAG | UART0 |
| `wifi` | Wi-Fi HTTP | USB-Serial-JTAG |
| `experimental/wifi-serial` | USB-Serial-JTAG + Wi-Fi | USB-Serial-JTAG |

This routing avoids the classic "UART image data and console logs fight over
the same pins" failure in the four maintained variants. The experimental
variant deliberately shares USB-Serial-JTAG and uses its `jtag_safe_printf`
wrapper; it is a work-in-progress path, not the default.

## 3. Model embedding

Each `main/CMakeLists.txt` embeds a model with:

```cmake
target_add_binary_data(${COMPONENT_LIB}
    "${CMAKE_SOURCE_DIR}/models/TinyAgCNN_v2_s8.espdl" BINARY)
```

The linker symbol is `_binary_TinyAgCNN_v2_s8_espdl_start`. The v2 firmware
copies that region into PSRAM before constructing `dl::Model`.
`uart-grayscale` uses its smaller model directly from flash rodata.

To swap a model:

1. Put the `.espdl` file in the variant's `models/` directory.
2. Update the `target_add_binary_data` path and the `extern` symbol names in
   `main.cpp` if the filename changes.
3. Rebuild. Do not assume a new `.espdl` is drop-in compatible: input layout,
   class count, and quantisation metadata must match.

### Model-to-firmware map

| Firmware | Model file | SHA-256 |
|---|---|---|
| `uart-grayscale` | `TinyAgCNN_s8.espdl` | `5FDB51D9…A55D609` |
| `uart-rgb` | `TinyAgCNN_v2_s8.espdl` | `E8163BF4…1238614` |
| `wifi` | `TinyAgCNN_v2_s8.espdl` | `E8163BF4…1238614` |
| `experimental/wifi-serial` | `TinyAgCNN_v2_s8.espdl` | `E8163BF4…1238614` |
| `jtag` | `TinyAgCNN_v2_s8.espdl` | `D6778781…F3FDC9F` |

Full hashes are in [`../models/README.md`](../models/README.md) and
[`versions.md`](versions.md).

## 4. INT8 model path

The release contains float ONNX models and already-quantised `.espdl` files.
The original quantisation workspace is not part of this repository, so the exact
ESP-PPQ command is not preserved. The general path is:

```text
PyTorch / Keras checkpoint
        │  export
        ▼
float ONNX (NCHW, static batch 1)
        │  ESP-PPQ calibration on representative images
        ▼
INT8 .espdl + .json quantisation config
        │  embed
        ▼
firmware
```

### What is known about the released files

- The v2 ONNX graph has input `[1,3,96,96]`, output `[1,4]`, and opset 13.
- The grayscale GAP ONNX graphs have input `[1,96,96,1]` (TensorFlow-style
  NHWC, `TinyAgCNN.onnx`) or `[1,1,96,96]` (`TinyAgCNN_s8.onnx`).
- The `.json` files next to each `.espdl` are the preserved quantisation
  configs. They are documentation, not a complete calibration script.
- The firmware expects NHWC input for ESP-DL and uses exponent `-7`
  (`1/128`) for the input tensor.

### Reproducing a quantisation

1. Export the float ONNX:

   ```bash
   python -m training.src.export_onnx \
     --arch v2 \
     --checkpoint models/tinyagcnn_v2_best.pth \
     --out build/TinyAgCNN_v2.onnx
   ```

2. Calibrate with ESP-PPQ using a representative set of 96×96 images and the
   preprocessing contract in [`architecture.md`](architecture.md). For the v2
   model, feed `[-1,1]` data; do not use ImageNet mean/std.
3. Export the `.espdl` and keep the accompanying `.json`.
4. Replace the firmware model and rebuild.
5. Verify a known image on the device. A model that loads but predicts the
   wrong class usually indicates a layout, normalisation, or class-order
   mismatch, not a broken quantiser.

The exact ESP-PPQ API changes between releases; use the documentation for the
installed ESP-DL version rather than copying an old command blindly.

## 5. Serial use

### Wiring

| Variant | ESP32-S3 pins | Host |
|---|---|---|
| UART v1/v2 | UART0 TX/RX (usually GPIO43/44) | USB-UART bridge or board header |
| JTAG | USB-Serial-JTAG D+/D− | USB cable to the native USB port |

The UART variants use 115200 8N1. The host must wait for `READY` before sending
the frame.

### Frame

```text
0xAA + payload
```

| Mode | Payload | Bytes |
|---|---|---:|
| `uart-v1` | 96×96 grayscale, signed INT8 (`pixel - 128`) | 9,216 |
| `uart-v2` | 96×96 RGB, signed INT8 (`pixel - 128`) | 27,648 |
| `jtag` | 96×96 RGB, signed INT8 (`pixel - 128`) | 27,648 |

Response:

```text
CLASS:<name> SCORE:<int8> CONF:<0-100> TIME:<ms>ms
```

Use the CLI:

```bash
python host/serial/send_image.py \
  --port COM7 \
  --mode uart-v2 \
  --image leaf.jpg \
  --json
```

`--roi` crops the largest green region before resizing; it requires
`opencv-python`. `--dry-run` validates the payload without opening the port.
See [`../host/README.md`](../host/README.md).

### If the device never prints READY

- Check the console routing. UART0 is the image transport on `uart-rgb`; the
  console must be on USB-Serial-JTAG.
- Check the baud rate (115200).
- Press reset after opening the port; the host CLI expects the banner.
- On `jtag`, use the native USB port, not the UART bridge.

## 6. Wi-Fi use

1. Flash `firmware/wifi`.
2. Connect to the SoftAP:

   ```text
   SSID:     AgroStake
   password: agrostake2025
   ```

3. Open `http://192.168.4.1` and upload an image, or POST raw bytes:

   ```bash
   curl --request POST \
     --header "Content-Type: application/octet-stream" \
     --data-binary @frame.rgb \
     http://192.168.4.1/detect
   ```

   `frame.rgb` must be exactly 27,648 raw RGB888 bytes, row-major, 96×96×3.
   This is **not** a PNG/JPEG upload; the browser UI does the resize and
   conversion in JavaScript.

Response:

```json
{"class": "Healthy", "confidence": 87, "time_ms": 412}
```

The AP password and open HTTP server are demo defaults. Change the password and
do not expose the device to an untrusted network.

## 7. Host GUI

```bash
cd host/gui
pip install -r requirements.txt
python app.py
```

The GUI lists serial ports, performs a green-region ROI crop, resizes to
96×96 RGB, shifts to signed INT8, and sends the v2 frame. It expects the
`CONF` field when present and falls back to `SCORE` for older firmware.

## 8. Troubleshooting matrix

| Symptom | Likely cause | Check |
|---|---|---|
| No `READY` | wrong console/UART routing, wrong baud, stale reset | `sdkconfig.defaults`, cable, reset |
| Boot loop / allocation failure | PSRAM mode or size mismatch | `idf.py menuconfig` PSRAM settings |
| Model loads, all classes same | normalisation or layout mismatch | `pixel - 128`, NHWC, class order |
| Confidence stuck near 25 % | logits nearly equal or softmax over raw INT8 | inspect `SCORE` values; verify calibration |
| Serial frame timeout | payload size/magic mismatch | v1 9,216 vs v2 27,648; `0xAA` |
| Wi-Fi POST returns 400 | body not exactly 27,648 bytes | send raw RGB888, not JPEG |
| Build cannot find `esp-dl` | managed component not fetched | network access; `idf.py reconfigure` |
| Wrong class names | folder order differs from firmware order | alphabetical mapping in `data/README.md` |

## 9. What is intentionally not here

- A production OTA update path.
- Wi-Fi provisioning or authentication.
- A camera driver; the host or browser supplies the image.
- A field-validation dataset.
- A claim that host-side latency predicts ESP32-S3 latency.

The repository is a reproducible research/engineering snapshot, not a certified
product firmware.
