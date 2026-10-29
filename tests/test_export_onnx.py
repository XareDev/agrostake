from pathlib import Path

import numpy as np
import pytest
import torch

from training.src.export_onnx import main as export_main
from training.src.models import build_model

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "models" / "tinyagcnn_v2_best.pth"


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="released checkpoint not present")
def test_v2_onnx_export_round_trip(tmp_path: Path):
    out = tmp_path / "v2.onnx"
    # argparse reads sys.argv; call the CLI entry point with an explicit list
    assert export_main([
        "--arch", "v2",
        "--checkpoint", str(CHECKPOINT),
        "--out", str(out),
    ]) == 0
    assert out.exists() and out.stat().st_size > 100_000

    import onnxruntime as ort

    session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    x = np.random.randn(1, 3, 96, 96).astype(np.float32)
    got = session.run(None, {"input": x})[0]
    model = build_model("v2").eval()
    model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu", weights_only=True))
    expected = model(torch.from_numpy(x)).detach().numpy()
    np.testing.assert_allclose(got, expected, atol=1e-4, rtol=1e-4)
