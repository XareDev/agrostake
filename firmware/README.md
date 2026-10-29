# Firmware variants

Five independent ESP-IDF projects are kept under `firmware/`. They are real
variants from the original workspace, not five copies of one build. Each has
its own `main/`, model binary, partition table, `sdkconfig.defaults`, and
`idf_component.yml`.

| Directory | Model | Transport | Status |
|---|---|---|---|
| `uart-grayscale/` | `TinyAgCNN_s8.espdl` (27,828 params) | UART0, 115200 8N1 | v1 baseline |
| `uart-rgb/` | `TinyAgCNN_v2_s8.espdl` (275,236 params) | UART0, 115200 8N1 | v2 |
| `jtag/` | v2 `.espdl` (separate export) | USB-Serial-JTAG | v2 |
| `wifi/` | `TinyAgCNN_v2_s8.espdl` | SoftAP + HTTP | v2 |
| `experimental/wifi-serial/` | `TinyAgCNN_v2_s8.espdl` | USB-Serial-JTAG + HTTP | WIP, not maintained |

## Common build

```bash
cd firmware/<variant>
idf.py set-target esp32s3
idf.py build
idf.py -p COM7 flash monitor
```

- ESP-IDF ≥ 5.4.
- `main/idf_component.yml` pulls `espressif/esp-dl ^3.3.2`.
- `partitions.csv` reserves a 6 MB factory app; no OTA.
- `sdkconfig.defaults` assumes 16 MB QIO flash and 8 MB octal PSRAM at 80 MHz.
- The model is linked into the app with `target_add_binary_data`. The v2
  variants copy it into 32-byte-aligned PSRAM at boot; `uart-grayscale` loads
  its smaller model directly from flash rodata.
- Build products (`build/`, `sdkconfig`, `managed_components/`, `.bin`, `.elf`)
  are ignored by Git.

If your board uses different PSRAM or flash, change `sdkconfig.defaults` or run
`idf.py menuconfig` before building. A model that fails to allocate PSRAM will
not run; the boot log reports the failure.

## Console routing

This is the most common bring-up issue. In the four maintained variants the
image transport and the console are deliberately separate. The experimental
variant shares USB-Serial-JTAG and is a work-in-progress exception.

| Variant | Image transport | Console |
|---|---|---|
| `uart-grayscale` | UART0 | USB-Serial-JTAG |
| `uart-rgb` | UART0 | USB-Serial-JTAG |
| `jtag` | USB-Serial-JTAG | UART0 |
| `wifi` | Wi-Fi HTTP | USB-Serial-JTAG |
| `experimental/wifi-serial` | USB-Serial-JTAG + HTTP | USB-Serial-JTAG |

## Serial protocol

### `uart-grayscale` (v1)

```text
host -> device: 0xAA | 9,216 signed INT8 bytes (96×96×1, pixel - 128)
device -> host: CLASS:<name> SCORE:<int8> CONF:<0-100> TIME:<ms>ms
```

- Model: `models/TinyAgCNN_s8.espdl`.
- The host sends already-shifted INT8 bytes; the firmware copies them directly.
- Wait for `READY` before sending.
- The grayscale model was trained on `[0,1]`; its INT8 metadata maps the
  `pixel - 128` bytes back to that range.

### `uart-rgb` (v2)

```text
host -> device: 0xAA | 27,648 signed INT8 bytes (96×96×3, pixel - 128)
device -> host: CLASS:<name> SCORE:<int8> CONF:<0-100> TIME:<ms>ms
```

- Model: shared `TinyAgCNN_v2_s8.espdl` (SHA-256 `E8163BF4…1238614`).
- Input is NHWC in the ESP-DL tensor; exponent `-7` means a 1/128 scale.
- The host must not send raw 0–255 bytes in this mode.

### `jtag` (v2)

Same payload and reply as `uart-rgb`, but over the USB-Serial-JTAG peripheral:

```text
host -> device: 0xAA | 27,648 signed INT8 bytes
device -> host: CLASS:<name> SCORE:<int8> CONF:<0-100> TIME:<ms>ms
```

- Console is on UART0, so use the native USB port for the image.
- The embedded model is a separate 315,776-byte export with SHA-256
  `D6778781…F3FDC9F`. It is intentionally kept in this directory.

### `experimental/wifi-serial`

This variant does **not** use the `0xAA` frame. Its serial command path is:

```text
host -> device: "IMAGE\n"
device -> host: "READY_FOR_IMAGE\n"
host -> device: 27,648 raw bytes (converted to INT8 on the device)
device -> host: RESULT:class=<name>,confidence=<0-100>,time=<ms>
```

It also exposes the Wi-Fi `/detect` endpoint. It is a work-in-progress
combination of two code paths, not a drop-in replacement for `wifi`. The
`jtag_safe_printf` wrapper exists to avoid host-driver hangs when the USB
Serial/JTAG FIFO is flushed. Treat it as provenance / a starting point.

## Wi-Fi protocol

`firmware/wifi/` starts:

```text
SSID:     AgroStake
password: agrostake2025
address:  http://192.168.4.1
```

Endpoints:

| Method | Path | Body | Response |
|---|---|---|---|
| GET | `/` | — | HTML UI |
| POST | `/detect` | exactly 27,648 raw RGB888 bytes | JSON |

```json
{"class": "Healthy", "confidence": 87, "time_ms": 412}
```

The browser UI reads the selected file, draws it to a 96×96 canvas, extracts
RGB bytes, and POSTs the raw buffer. It does not upload a JPEG/PNG. If you use
`curl`, build the raw frame yourself:

```bash
curl -X POST \
  -H "Content-Type: application/octet-stream" \
  --data-binary @frame.rgb \
  http://192.168.4.1/detect
```

The AP password is compiled into the source and the HTTP server has no
authentication. Change both before using the device outside a bench demo.

## Confidence semantics

`CONF` and the Wi-Fi `confidence` field are computed as a softmax over the four
INT8 logits:

```text
conf = 100 * exp(score[win] - max) / sum(exp(score[i] - max))
```

This is a **relative** confidence. It is not calibrated and it is not a
probability that the plant is correctly diagnosed. A value near 25 % means the
four logits are close, not that the model is “25 % sure” in a statistical
sense. The raw `SCORE` is included so you can inspect the quantised logits.

The `CONF` field was added during repository cleanup. Older host code can parse
`SCORE` and `TIME`; the included CLI and GUI fall back to a linear display
mapping when `CONF` is absent.

## Class order

All firmware variants use:

```c
"Black_rot", "Esca", "Healthy", "Leaf_blight"
```

This is alphabetical order of the ImageFolder directories. If you rename the
dataset folders, update the firmware array and the host class display together.

## Files and hashes

| File | SHA-256 | Size |
|---|---|---:|
| `uart-grayscale/models/TinyAgCNN_s8.espdl` | `5FDB51D9B4321515516858E501CBB4AE697289C60FE1F9CFDC3819DD9A55D609` | 46,576 B |
| `uart-rgb/models/TinyAgCNN_v2_s8.espdl` | `E8163BF47B8EBC6EEB3CA2980A696ED89E6F4D116A2E82C5D538930211238614` | 315,808 B |
| `wifi/models/TinyAgCNN_v2_s8.espdl` | `E8163BF47B8EBC6EEB3CA2980A696ED89E6F4D116A2E82C5D538930211238614` | 315,808 B |
| `experimental/wifi-serial/models/TinyAgCNN_v2_s8.espdl` | `E8163BF47B8EBC6EEB3CA2980A696ED89E6F4D116A2E82C5D538930211238614` | 315,808 B |
| `jtag/models/TinyAgCNN_v2_s8.espdl` | `D677878164AC19185346A021E850CE8A1967C4CB0E71D48992A375DEAF3FDC9F` | 315,776 B |

The shared v2 export differs from the JTAG export by 32 bytes. Do not
“normalise” them without re-testing the JTAG build.

## Not verified here

The repository cleanup did not recompile the firmware because the local
ESP-IDF installation was incomplete. The source, component manifest, partition
tables, and defaults are provided so the projects can be built with a clean
ESP-IDF 5.4+ installation. Validate on your own board, especially:

- PSRAM mode/speed,
- console routing,
- model hash after build,
- actual inference latency,
- Wi-Fi AP behaviour under concurrent requests.

## Related documents

- [`../docs/architecture.md`](../docs/architecture.md) — task model and data flow.
- [`../docs/deployment.md`](../docs/deployment.md) — build, flash, INT8 path, troubleshooting.
- [`../models/README.md`](../models/README.md) — model provenance and hashes.
- [`../host/README.md`](../host/README.md) — CLI and GUI.
