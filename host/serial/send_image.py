#!/usr/bin/env python3
"""Send one image to an AgroStake ESP32-S3 firmware over USB serial.

Modes
-----
uart-v1 : 96x96x1 grayscale, 9,216-byte payload (uart-grayscale firmware)
uart-v2 : 96x96x3 RGB, 27,648-byte payload (uart-rgb firmware)
jtag    : 96x96x3 RGB, 27,648-byte payload (jtag firmware)

The UART/JTAG firmware expects already-shifted INT8 bytes (``pixel - 128``);
the magic byte ``0xAA`` precedes the payload.  The Wi-Fi firmware is separate:
it accepts raw RGB888 over HTTP, so no serial framing is involved.

Example::

    python host/serial/send_image.py --port COM7 --mode uart-v2 \
        --image leaf.jpg --json

The script intentionally mirrors the firmware contract rather than training
normalisation: no ImageNet mean/std is subtracted.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

MAGIC = 0xAA
MODES = {
    "uart-v1": {"channels": 1, "size": 96 * 96},
    "uart-v2": {"channels": 3, "size": 96 * 96 * 3},
    "jtag": {"channels": 3, "size": 96 * 96 * 3},
}
RESULT_RE = re.compile(r"CLASS:(?P<cls>\S+)(?:\s+SCORE:(?P<score>-?\d+))?(?:\s+CONF:(?P<conf>\d+))?(?:\s+TIME:(?P<time>\d+)ms)?")


def roi_crop(image: Image.Image) -> Image.Image:
    """Crop the largest green region, matching the desktop GUI heuristic."""
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("--roi requires opencv-python") from exc
    rgb = np.asarray(image.convert("RGB"))
    hsv = cv2.cvtColor(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([30, 35, 35]), np.array([90, 255, 255]))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    pad_x, pad_y = int(w * 0.10), int(h * 0.10)
    box = (
        max(0, x - pad_x),
        max(0, y - pad_y),
        min(image.width, x + w + pad_x),
        min(image.height, y + h + pad_y),
    )
    return image.crop(box)


def preprocess(path: str | Path, mode: str, use_roi: bool = False) -> bytes:
    spec = MODES[mode]
    image = Image.open(path)
    if use_roi:
        image = roi_crop(image)
    if spec["channels"] == 1:
        image = image.convert("L").resize((96, 96), Image.Resampling.LANCZOS)
    else:
        image = image.convert("RGB").resize((96, 96), Image.Resampling.LANCZOS)
    arr = np.asarray(image, dtype=np.int16)
    shifted = np.clip(arr - 128, -128, 127).astype(np.int8)
    payload = shifted.tobytes()
    assert len(payload) == spec["size"], (len(payload), spec["size"])
    return payload


def parse_result(line: str) -> dict | None:
    match = RESULT_RE.search(line)
    if not match:
        return None
    score = int(match.group("score")) if match.group("score") is not None else None
    if match.group("conf") is not None:
        confidence = int(match.group("conf"))
    elif score is not None:
        # Backward-compatible display fallback for older firmware.
        confidence = max(0, min(100, (score + 128) / 2.56))
    else:
        confidence = None
    return {
        "class": match.group("cls"),
        "score": score,
        "confidence": confidence,
        "time_ms": int(match.group("time")) if match.group("time") else None,
    }


def send(port: str, baud: int, payload: bytes, timeout: float) -> dict:
    try:
        import serial
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("pyserial is required: pip install pyserial") from exc

    with serial.Serial(port, baud, timeout=0.1) as ser:
        ser.reset_input_buffer()
        deadline = time.time() + timeout
        ready = False
        while time.time() < deadline:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if line:
                print(f"[device] {line}")
                if "READY" in line.upper():
                    ready = True
                    break
        if not ready:
            raise SystemExit("device did not send READY")
        ser.write(bytes([MAGIC]))
        ser.write(payload)
        ser.flush()
        deadline = time.time() + timeout
        while time.time() < deadline:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            print(f"[device] {line}")
            parsed = parse_result(line)
            if parsed:
                return parsed
    raise SystemExit("timed out waiting for CLASS: result")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", required=True, help="e.g. COM7 or /dev/ttyACM0")
    p.add_argument("--mode", choices=sorted(MODES), default="uart-v2")
    p.add_argument("--image", required=True)
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--timeout", type=float, default=10.0)
    p.add_argument("--roi", action="store_true", help="crop the largest green region first")
    p.add_argument("--json", action="store_true", help="print the parsed result as JSON")
    p.add_argument("--dry-run", action="store_true", help="validate preprocessing without opening the port")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = preprocess(args.image, args.mode, args.roi)
    if args.dry_run:
        print(f"mode={args.mode} bytes={len(payload)} magic=0x{MAGIC:02X}")
        return 0
    result = send(args.port, args.baud, payload, args.timeout)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(
            f"{result['class']}  confidence={result['confidence']}  "
            f"time={result['time_ms']} ms"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
