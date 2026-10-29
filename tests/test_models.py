from pathlib import Path

import pytest
import torch

from training.src.models import build_model, total_parameters, trainable_parameters

ROOT = Path(__file__).resolve().parents[1]


def test_parameter_counts_match_released_artifacts():
    assert trainable_parameters(build_model("v2")) == 275_236
    assert trainable_parameters(build_model("gray")) == 27_828
    assert trainable_parameters(build_model("binary")) == 497_226
    assert total_parameters(build_model("gray")) == 27_828


def test_forward_shapes():
    assert build_model("v2")(torch.zeros(1, 3, 96, 96)).shape == (1, 4)
    assert build_model("gray")(torch.zeros(1, 1, 96, 96)).shape == (1, 4)
    assert build_model("binary")(torch.zeros(1, 3, 96, 96)).shape == (1, 2)


@pytest.mark.parametrize("arch,channels", [("v2", 3), ("binary", 3)])
def test_released_pytorch_checkpoints_load(arch, channels):
    checkpoint = {
        "v2": ROOT / "models" / "tinyagcnn_v2_best.pth",
        "binary": ROOT / "models" / "TinyBinaryCNN_v1_best.pth",
    }[arch]
    if not checkpoint.exists():
        pytest.skip(f"missing {checkpoint}")
    model = build_model(arch)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    assert model(torch.zeros(1, channels, 96, 96)).shape == (1, 4 if arch == "v2" else 2)


def test_unknown_arch_rejected():
    with pytest.raises(ValueError):
        build_model("mobilenet")
