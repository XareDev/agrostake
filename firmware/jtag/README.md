# AgroStake JTAG firmware

RGB v2 inference over the ESP32-S3 USB-Serial-JTAG peripheral. This is the
native-USB counterpart to [`../uart-rgb`](../uart-rgb); the model and task are
the same, but the transport and the embedded model export are not.

## Hardware and configuration

| Item | Value |
|---|---|
| Board | ESP32-S3 with 16 MB flash and 8 MB octal PSRAM |
| CPU | 240 MHz |
| Image transport | USB-Serial-JTAG (native USB port) |
| Console | UART0, 115200 8N1 |
| Input | 96×96 RGB, signed INT8 (`pixel - 128`) |
| Model | `models/TinyAgCNN_v2_s8.espdl` (315,776 B) |

The model is a separate v2 export from the 315,808-byte shared file used by the
UART and Wi-Fi variants. It has the same architecture and class order but is not
byte-identical; keep the variant's copy. SHA-256:

```text
D677878164AC19185346A021E850CE8A1967C4CB0E71D48992A375DEAF3FDC9F
```

The app copies the embedded model into 32-byte-aligned PSRAM at boot. A board
without working PSRAM will fail before the ready handshake.

## Build and flash

```bash
cd firmware/jtag
idf.py set-target esp32s3
idf.py build
idf.py -p COM7 flash
```

Use ESP-IDF ≥ 5.4; `main/idf_component.yml` fetches `espressif/esp-dl ^3.3.2`.
Flash over the native USB port. Boot/error logs go to UART0; `READY` and the
result line are written explicitly to the USB data channel, so the host does
not need the UART console.

## Protocol

```text
device -> host: READY\n
host   -> device: 0xAA | 27,648 signed INT8 bytes (96×96×3, HWC)
device -> host: CLASS:<name> SCORE:<int8> CONF:<0-100> TIME:<ms>ms\n
```

- `SCORE` is the raw winning INT8 logit.
- `CONF` is a softmax over the four logits, computed on the device. It is a
  relative confidence, not a calibrated probability.
- `TIME` is the span around `model->run()` in milliseconds.
- The frame has no length field or checksum; the firmware reads exactly 27,648
  bytes after `0xAA` and times out after 5 s.

Class order: `Black_rot`, `Esca`, `Healthy`, `Leaf_blight`.

## Host

Use the repository CLI:

```bash
python host/serial/send_image.py \
  --port COM7 --mode jtag --image path/to/leaf.jpg --json
```

It performs the deployment preprocessing (`resize → RGB → pixel - 128`) and
waits for `READY` on the native USB port.

## Provenance

This variant was reconciled with the supplied `JTAG.zip` archive. The archive's
`main.cpp` and 315,776-byte model are the same firmware/model pair already kept
here; the maintained source adds allocation checks, the `CONF` field and an
explicit USB `READY` write. The archive's `send_image.py` applied ImageNet
mean/std normalization, which does not match the v2 deployment contract
(`[-1,1]`, `pixel - 128`), so it was not imported; use
[`../../host/serial/send_image.py`](../../host/serial/send_image.py) instead.
The generated `sdkconfig`, empty `.gitignore` and generic archive README were
not copied. This project was build-checked with ESP-IDF v6.0 and ESP-DL 3.3.2;
hardware timing still needs board validation.
