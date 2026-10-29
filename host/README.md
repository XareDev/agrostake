# Host tools

Two host interfaces are included:

| Tool | Path | Purpose |
|---|---|---|
| CLI | `host/serial/send_image.py` | Send one image to UART v1, UART v2, or JTAG firmware |
| GUI | `host/gui/` | PyQt6 desktop app for the RGB v2 firmware |

Both use the same deployment preprocessing: resize to 96×96, then convert to
signed INT8 with `pixel - 128`. No ImageNet mean/std subtraction is applied.

## CLI

### Install

From the repository root:

```bash
pip install pyserial pillow numpy
# optional, only for --roi:
pip install opencv-python
```

Or install the full environment with `pip install -r requirements.txt`.

### Run

```bash
python host/serial/send_image.py \
  --port COM7 \
  --mode uart-v2 \
  --image path/to/leaf.jpg \
  --json
```

| Option | Meaning |
|---|---|
| `--port` | Serial port, e.g. `COM7` or `/dev/ttyACM0` |
| `--mode` | `uart-v1` (9,216 B grayscale), `uart-v2` (27,648 B RGB), or `jtag` (27,648 B RGB) |
| `--image` | Input image (JPEG/PNG/BMP, etc.) |
| `--baud` | Default 115200 |
| `--timeout` | Seconds to wait for READY and the result |
| `--roi` | Crop the largest green region before resizing (requires OpenCV) |
| `--json` | Print the parsed result as JSON |
| `--dry-run` | Validate preprocessing and payload size without opening the port |

The CLI waits for a line containing `READY`, sends `0xAA` plus the payload, and
parses:

```text
CLASS:Healthy SCORE:91 CONF:87 TIME:412ms
```

If the firmware is older and omits `CONF`, the CLI falls back to
`(SCORE + 128) / 2.56` for display. That fallback is not a calibrated
probability; it exists only so old devices remain usable.

### Mode contract

| Mode | Payload | Firmware |
|---|---:|---|
| `uart-v1` | 96×96×1 signed INT8 | `firmware/uart-grayscale` |
| `uart-v2` | 96×96×3 signed INT8 | `firmware/uart-rgb` |
| `jtag` | 96×96×3 signed INT8 | `firmware/jtag` |

The experimental firmware uses a different text command (`IMAGE\n`) and is not
covered by this CLI. See [`../firmware/README.md`](../firmware/README.md).

### Examples

```bash
# v1 grayscale, no ROI crop
python host/serial/send_image.py \
  --port COM7 --mode uart-v1 --image leaf.jpg

# v2 with green-region crop
python host/serial/send_image.py \
  --port COM7 --mode uart-v2 --image leaf.jpg --roi --json

# JTAG firmware on the native USB port
python host/serial/send_image.py \
  --port COM9 --mode jtag --image leaf.jpg

# Check payload size without hardware
python host/serial/send_image.py \
  --port COM7 --mode uart-v2 --image leaf.jpg --dry-run
```

## GUI

```bash
cd host/gui
pip install -r requirements.txt
python app.py
```

The GUI:

1. Lists available serial ports.
2. Opens an image and detects the largest green region (HSV threshold).
3. Resizes the ROI to 96×96 RGB.
4. Converts to signed INT8 (`pixel - 128`).
5. Sends the `0xAA` frame and displays class, confidence, and latency.

`config.ini` contains the serial defaults and target dimensions. It is set for
the RGB v2 firmware (`IMAGE_CHANNELS = 3`, `IMAGE_DATA_SIZE = 27648`). The GUI
does not implement the grayscale v1 mode; use the CLI for `uart-v1`.

The GUI's result parser accepts both the new `CONF` field and the legacy
`SCORE`-only reply.

## Why two tools?

They are intentionally small and separate:

- The CLI is easy to script and is the canonical protocol reference.
- The GUI demonstrates the ROI workflow and is convenient at the bench.

They duplicate the protocol parser instead of sharing a package so each can be
read and debugged independently. If you change the frame, update both
`host/serial/send_image.py` and `host/gui/processor.py`.

## Troubleshooting

| Problem | Fix |
|---|---|
| `device did not send READY` | Check console routing, baud, and reset the board |
| `pyserial` missing | `pip install pyserial` |
| `--roi` fails | `pip install opencv-python` |
| Wrong class / low confidence | Check `pixel - 128`, RGB vs grayscale mode, and class order |
| GUI cannot find port | Install the board's USB-UART or USB-JTAG driver; close other serial monitors |
| Timeout after sending | The payload may be the wrong size or the firmware may expect the experimental `IMAGE\n` command |

## Tests

```bash
python -m pytest tests/test_host_serial.py -q
```

This checks the 9,216 / 27,648-byte payloads, signed-byte range, and both reply
formats.
