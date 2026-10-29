"""Qt worker that sends an image to an AgroStake ESP32-S3 over USB serial.

This mirrors ``host/serial/send_image.py`` for the UART v2 / JTAG RGB firmware:
the payload is 96x96 RGB888 shifted to signed INT8 (``pixel - 128``) and
prefixed with the ``0xAA`` magic byte.
"""
from __future__ import annotations

import time

import cv2
import numpy as np
import serial
from PIL import Image
from PyQt6.QtCore import QThread, pyqtSignal


class SerialProcessor(QThread):
    log_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)
    result_signal = pyqtSignal(dict)
    progress_signal = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self.port: str | None = None
        self.image_path: str | None = None
        self.ser: serial.Serial | None = None
        self.is_running = False

    def set_parameters(self, port: str, image_path: str) -> None:
        self.port = port
        self.image_path = image_path

    def run(self) -> None:
        try:
            self.is_running = True
            self.progress_signal.emit("Processing image...")
            payload = self._load_and_preprocess_image_rgb()
            if payload is None:
                return

            self.progress_signal.emit("Connecting to ESP32...")
            if not self._open_serial():
                return

            self.progress_signal.emit("Waiting for ESP32 READY...")
            if not self._wait_for_ready():
                self.error_signal.emit("Timeout: ESP32 never signalled READY")
                return

            self.progress_signal.emit("Sending image data...")
            self._send_image(payload)

            self.progress_signal.emit("Waiting for inference...")
            self._read_result()
        except Exception as exc:  # pragma: no cover - GUI error path
            self.error_signal.emit(f"Error: {exc}")
        finally:
            self._close_serial()
            self.is_running = False
            self.progress_signal.emit("Ready")

    def _roi_box(self, image: Image.Image) -> tuple[int, int, int, int]:
        rgb = np.asarray(image)
        hsv = cv2.cvtColor(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([30, 35, 35]), np.array([90, 255, 255]))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
            pad_x, pad_y = int(w * 0.10), int(h * 0.10)
            return (
                max(0, x - pad_x),
                max(0, y - pad_y),
                min(image.width, x + w + pad_x),
                min(image.height, y + h + pad_y),
            )
        self.log_signal.emit("No leaf ROI detected; using the full image.")
        return (0, 0, image.width, image.height)

    def _load_and_preprocess_image_rgb(self) -> bytes | None:
        try:
            image = Image.open(self.image_path).convert("RGB")
            box = self._roi_box(image)
            roi = image.crop(box).resize((96, 96), Image.Resampling.LANCZOS)
            arr = np.asarray(roi, dtype=np.int16)
            shifted = np.clip(arr - 128, -128, 127).astype(np.int8)
            return shifted.tobytes()
        except Exception as exc:  # pragma: no cover - GUI error path
            self.error_signal.emit(f"Preprocessing error: {exc}")
            return None

    def _open_serial(self) -> bool:
        try:
            self.ser = serial.Serial(self.port, 115200, timeout=0.2)
            self.ser.setDTR(False)
            time.sleep(0.1)
            self.ser.setDTR(True)
            return True
        except Exception as exc:  # pragma: no cover - GUI error path
            self.error_signal.emit(f"Serial error: {exc}")
            return False

    def _wait_for_ready(self, timeout: float = 10.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.ser and self.ser.in_waiting:
                line = self.ser.readline().decode("utf-8", errors="ignore").strip()
                if line:
                    self.log_signal.emit(f"[ESP32] {line}")
                    if "READY" in line.upper():
                        return True
            time.sleep(0.01)
        return False

    def _send_image(self, payload: bytes) -> None:
        if not self.ser:
            return
        self.ser.write(bytes([0xAA]))
        chunk = 2048
        for start in range(0, len(payload), chunk):
            self.ser.write(payload[start:start + chunk])
            time.sleep(0.001)
        self.ser.flush()
        self.log_signal.emit(f"Sent {len(payload)} bytes.")

    def _read_result(self) -> None:
        deadline = time.time() + 10.0
        while self.is_running and time.time() < deadline:
            if self.ser and self.ser.in_waiting:
                line = self.ser.readline().decode("utf-8", errors="ignore").strip()
                if line:
                    self.log_signal.emit(f"[ESP32] {line}")
                    result = self._parse_result(line)
                    if result:
                        self.result_signal.emit(result)
                        return
            time.sleep(0.01)

    @staticmethod
    def _parse_result(line: str) -> dict | None:
        if "CLASS:" not in line:
            return None
        parts = {p.split(":", 1)[0]: p.split(":", 1)[1] for p in line.split() if ":" in p}
        try:
            score = int(parts.get("SCORE", 0))
            if "CONF" in parts:
                confidence = int(parts["CONF"])
            else:
                confidence = max(0, min(100, (score + 128) / 2.56))
            return {
                "class_name": parts.get("CLASS", "Unknown"),
                "score": score,
                "confidence": confidence,
                "time_ms": int(parts.get("TIME", "0").replace("ms", "")),
            }
        except (TypeError, ValueError):
            return None

    def _close_serial(self) -> None:
        if self.ser and self.ser.is_open:
            self.ser.close()
