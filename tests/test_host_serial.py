from pathlib import Path

import numpy as np
from PIL import Image

from host.serial.send_image import MODES, parse_result, preprocess


def test_payload_sizes(tmp_path: Path):
    path = tmp_path / "leaf.png"
    Image.fromarray(np.full((120, 140, 3), 128, dtype=np.uint8)).save(path)
    assert len(preprocess(path, "uart-v1")) == MODES["uart-v1"]["size"] == 9216
    assert len(preprocess(path, "uart-v2")) == MODES["uart-v2"]["size"] == 27648
    assert len(preprocess(path, "jtag")) == MODES["jtag"]["size"] == 27648


def test_preprocess_shift_is_signed_int8(tmp_path: Path):
    path = tmp_path / "leaf.png"
    Image.fromarray(np.zeros((96, 96, 3), dtype=np.uint8)).save(path)
    payload = preprocess(path, "uart-v2")
    arr = np.frombuffer(payload, dtype=np.int8)
    assert arr.min() == -128
    assert arr.max() == -128


def test_parse_new_and_legacy_replies():
    new = parse_result("CLASS:Healthy SCORE:91 CONF:87 TIME:412ms")
    assert new == {"class": "Healthy", "score": 91, "confidence": 87, "time_ms": 412}
    legacy = parse_result("CLASS:Esca SCORE:64 TIME:700ms")
    assert legacy["class"] == "Esca"
    assert legacy["confidence"] == 75
