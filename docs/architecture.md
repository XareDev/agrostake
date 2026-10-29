# Architecture

This document describes the models and the runtime path they share. It is a
map of the code in this repository, not a claim that every variant has been
re-validated on hardware after the repository cleanup.

```text
 image file / camera
        │
        ▼
 host preprocessing (96×96, class contract)
        │  UART / USB-Serial-JTAG / HTTP
        ▼
 ESP32-S3 firmware
   ├─ receive raw pixel frame
   ├─ shift to INT8 (UART/JTAG) or shift in firmware (Wi-Fi)
   ├─ esp-dl model inference
   ├─ class + score + relative confidence + latency
   └─ serial line / JSON response
        │
        ▼
 host CLI / GUI / browser
```

## Models

### RGB v2 — 275,236 parameters

`TinyAgCNNv2` is the deployed RGB model. It is a small VGG-style convolutional
network with global average pooling and a two-layer classifier.

```text
input  96×96×3
Conv3×3 VALID 3→32   → BN → ReLU → MaxPool2   47×47×32
Conv3×3 VALID 32→64  → BN → ReLU → MaxPool2   22×22×64
Conv3×3 VALID 64→128 → BN → ReLU → MaxPool2   10×10×128
Conv3×3 VALID 128→128→ BN → ReLU → MaxPool2    4×4×128
GlobalAveragePool → 128
Linear 128→256 → ReLU → Dropout(0.3) → Linear 256→4
```

- Convolutions have `bias=False`; BatchNorm supplies the bias.
- 275,236 trainable parameters.
- Float checkpoint: `models/tinyagcnn_v2_best.pth`.
- ONNX: `models/TinyAgCNN_v2.onnx`, input `[1,3,96,96]`, output `[1,4]`,
  opset 13.
- INT8: `models/TinyAgCNN_v2_s8.espdl`, 315,808 bytes (308.4 KiB), used by
  UART-RGB, Wi-Fi, and experimental firmware. The JTAG variant embeds a
  315,776-byte export with a different digest; it is documented in
  [`../models/README.md`](../models/README.md).

The PyTorch class was reconstructed from the ONNX graph and checkpoint because
the original v2 training notebook was not preserved. The released checkpoint
loads with `strict=True`, and `tests/test_export_onnx.py` checks ONNX Runtime
parity.

### Grayscale GAP — 27,828 parameters

This is the original v1 model and the architecture of the deployed grayscale
`.keras`/`.tflite`/`.espdl` artefacts.

```text
input  96×96×1
Conv3×3 VALID 1→16  → BN → ReLU → MaxPool2   47×47×16
Conv3×3 VALID 16→32 → BN → ReLU → MaxPool2   22×22×32
Conv3×3 VALID 32→64 → BN → ReLU → MaxPool2   10×10×64
GlobalAveragePool → 64
Dense 64→64 → ReLU → Dropout(0.3) → Dense 64→4
```

- 27,828 trainable parameters; 28,052 weight elements including BatchNorm
  running statistics.
- `TinyAgCNN_best.keras` is the source checkpoint; `best_baseline.keras` and
  `best_kd.keras` are the baseline and KD variants.
- `TinyAgCNN_int8.tflite` is a 36.3 KB TFLite reference; `TinyAgCNN_s8.espdl`
  is the 45.5 KB ESP-DL deployment artefact.
- The 843,364-parameter Flatten variant exists only in
  `training/notebooks/legacy/` and is not deployed.

### Binary gate — 497,226 parameters

`TinyBinaryCNN` is a healthy-vs-diseased experiment:

```text
input 96×96×3
Conv3×3 VALID 3→8  → BN → ReLU → MaxPool2   47×47×8
Conv3×3 VALID 8→16 → BN → ReLU → MaxPool2   22×22×16
Flatten → Linear 64 → ReLU → Dropout(0.3) → Linear 2
```

It is not loaded by any firmware and has no preserved final test report. Its
notebook used ImageNet normalisation, which is not the deployment `[-1,1]`
contract. See [`../models/README.md`](../models/README.md).

## Class order

The four classes are fixed in alphabetical ImageFolder order:

```text
0  Black_rot
1  Esca
2  Healthy
3  Leaf_blight
```

The display names are `Black Rot`, `ESCA`, `Healthy`, `Leaf Blight`. Firmware
strings use underscores to avoid spaces in serial output. If you train with a
different folder order, update the firmware class array and the class-name
mapping in `training/src/data.py`.

## Preprocessing contracts

The two model families were trained under different input conventions. This is
the most common source of “works in the notebook, fails on the board” bugs.

### RGB v2 and binary gate

Training defaults to symmetric normalisation:

```text
x = pixel / 127.5 - 1.0       # [-1, 1]
```

The host converts each channel to a signed byte:

```text
int8 = clip(pixel - 128, -128, 127)
```

The UART v2 and JTAG firmware copy those signed bytes directly into the ESP-DL
input tensor. The Wi-Fi firmware accepts raw RGB888 bytes and performs the
subtraction on the device.

### Grayscale GAP

The original TensorFlow notebook trained on `[0, 1]`. The INT8 TFLite/ESP-DL
metadata uses scale `1/255`, zero point `-128`, so the host again sends
`pixel - 128`. The firmware copies the signed bytes; the quantisation metadata
maps them back to `[0, 1]`. For a new from-scratch PyTorch grayscale model, use
`--normalization unit` to match the notebook, or `--normalization symmetric`
if you deliberately want the `[-1,1]` contract.

No ImageNet mean/std subtraction is used by the deployed firmware, with the
single exception of the unfinished binary-gate experiment described above.

## Serial framing

The maintained serial variants (`uart-grayscale`, `uart-rgb`, and `jtag`) use
the same frame. The experimental variant uses a text command instead; see
[`../firmware/README.md`](../firmware/README.md).

```text
host → device
┌──────────┬──────────────────────────────────────────────┐
│ 0xAA     │ 9,216 bytes (v1) or 27,648 bytes (v2/JTAG)   │
└──────────┴──────────────────────────────────────────────┘

device → host
CLASS:<name> SCORE:<int8> CONF:<0-100> TIME:<milliseconds>ms
```

- `SCORE` is the raw INT8 logit of the winning class.
- `CONF` is a softmax over the four INT8 logits, expressed as a percentage.
  It is a relative ranking score, not a calibrated probability.
- `TIME` is measured on the device with `esp_timer_get_time()`.
- The host must wait for a line containing `READY` before sending the magic
  byte. On UART variants `READY` follows the startup banner on the same port;
  on `jtag` it is written explicitly to USB-Serial-JTAG while the banner stays
  on the UART0 console.

The `CONF` field was added during this repository cleanup. Older firmware only
printed `CLASS`, `SCORE`, and `TIME`; the host CLI and GUI fall back to
`(SCORE + 128) / 2.56` when `CONF` is absent. That fallback is a display
convenience, not a probability.

## Wi-Fi path

The Wi-Fi firmware starts a SoftAP and an HTTP server:

```text
SSID:     AgroStake
password: agrostake2025
address:  http://192.168.4.1
```

The browser UI serves `index.html` from flash. The inference endpoint is:

```http
POST /detect
Content-Type: application/octet-stream

<27,648 raw RGB888 bytes, row-major, 96×96×3>
```

Response:

```json
{"class": "Healthy", "confidence": 87, "time_ms": 412}
```

The firmware copies the request body into a PSRAM buffer, converts each byte to
signed INT8, runs inference in a dedicated task pinned to core 1, and signals
the HTTP handler with a semaphore. `confidence` is the same softmax-over-INT8
score as the serial `CONF` field.

The AP password is compiled into the source. Change it before using the device
outside a bench/demo setting. The HTTP server is intentionally simple and has
no authentication; do not expose it to an untrusted network.

## Memory and flash layout

The recommended board profile is:

- ESP32-S3 with 16 MB quad SPI flash.
- 8 MB octal PSRAM at 80 MHz.
- 240 MHz CPU.

The default partition table reserves a 6 MB factory app and no OTA slot:

```text
nvs       0x9000   0x6000
phy_init  0xf000   0x1000
factory   0x10000  6M
```

The model is linked into the app image with `target_add_binary_data`. The v2
variants (`uart-rgb`, `jtag`, `wifi`, and experimental) copy it at boot into a
32-byte-aligned PSRAM buffer. The v1 grayscale firmware instead constructs the
model directly from flash rodata, which is small enough to fit comfortably. The
image input buffer is allocated in PSRAM in all variants. The Wi-Fi variant
additionally keeps its HTTP receive buffer in PSRAM. These choices keep internal
SRAM available for the Wi-Fi stack, FreeRTOS, and the ESP-DL runtime.

The exact free-heap numbers depend on the ESP-IDF version and board; no
universal memory measurement is claimed here. The firmware logs the allocation
failures explicitly rather than silently continuing.

## Firmware task model

### UART / JTAG variants

The main task initialises the transport, loads the model, prints `READY`, and
blocks on a full-frame read. It then runs inference synchronously and writes
the result line. There is no Wi-Fi and no background inference task.

### Wi-Fi variant

- `app_main` initialises NVS, network interfaces, the SoftAP, and the HTTP
  server.
- The HTTP handler copies the request body into `s_img_buf`, stores the request
  pointer, and gives `s_infer_sem`.
- The inference task waits on `s_infer_sem`, converts the frame to INT8, runs
  ESP-DL, fills `s_result_json`, and gives `s_done_sem`.
- The HTTP handler waits on `s_done_sem` and returns the JSON.

This is a single in-flight-request design. It is appropriate for a bench demo;
it is not a concurrent multi-client server.

### Experimental variant

`firmware/experimental/wifi-serial` combines the serial and Wi-Fi paths and
uses a `jtag_safe_printf` helper. It is a work-in-progress branch captured for
provenance, not a maintained default. See
[`../firmware/README.md`](../firmware/README.md) for the exact caveats.

## Host tools

- `host/serial/send_image.py` is the canonical one-shot CLI. It supports
  `uart-v1`, `uart-v2`, and `jtag`, validates the payload size, and parses both
  the new `CONF` reply and the legacy `SCORE` reply.
- `host/gui/` is the PyQt6 desktop application. It performs a green-region ROI
  crop, resizes to 96×96 RGB, shifts to signed INT8, and sends the v2 frame.
- The GUI and CLI do not share code, but they use the same byte contract. Any
  change to the firmware framing must be mirrored in both.

## Where to change what

| Change | File(s) |
|---|---|
| Model width / classifier | `training/src/models.py`, then retrain and re-export |
| Preprocessing / augmentation | `training/src/data.py` and the host tools |
| Class names/order | `training/src/data.py`, firmware `CLASS_NAMES`, `host/gui/config.ini` |
| Serial framing | all `firmware/*/main/main.cpp`, `host/serial/send_image.py`, `host/gui/processor.py` |
| Wi-Fi endpoint/UI | `firmware/wifi/main/main.cpp`, `firmware/wifi/main/index.html` |
| Board flash/PSRAM | `firmware/*/sdkconfig.defaults` |
| Partition layout | `firmware/*/partitions.csv` |
| Model binary | `firmware/*/models/*.espdl` and the `target_add_binary_data` path |

## Non-goals

- The repository does not include the dataset or a field-validation protocol.
- It does not include OTA updates, authentication, or a production Wi-Fi
  provisioning flow.
- It does not claim that the reported test accuracy transfers to a new camera
  or vineyard without revalidation.
