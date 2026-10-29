"""Model definitions for the AgroStake edge classifiers.

The classes here are reconstructed from the exported artefacts in ``models/``
and from the original training notebooks. They are intentionally small: the
deployed models are 27.8k (grayscale), 275.2k (RGB v2) and 497.2k (binary gate)
trainable parameters.

All models use NCHW tensors in PyTorch. The firmware consumes NHWC INT8 tensors,
which is an export/quantisation concern, not a training concern.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv3x3 (VALID) -> BatchNorm -> ReLU -> MaxPool2."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TinyAgCNNv2(nn.Module):
    """RGB 96x96 four-convolution classifier used by the v2 firmware.

    Architecture (96x96x3 input)::

        Conv 3->32 3x3 VALID -> BN -> ReLU -> MaxPool2   47x47x32
        Conv 32->64 3x3 VALID -> BN -> ReLU -> MaxPool2  22x22x64
        Conv 64->128 3x3 VALID -> BN -> ReLU -> MaxPool2 10x10x128
        Conv 128->128 3x3 VALID -> BN -> ReLU -> MaxPool2 4x4x128
        GlobalAveragePool -> 128
        Linear 128->256 -> ReLU -> Dropout(0.3) -> Linear 256->4

    The state-dict keys match ``models/tinyagcnn_v2_best.pth`` exactly
    (``features.N.block.*`` and ``classifier.{1,4}.*``).
    """

    def __init__(self, num_classes: int = 4, dropout: float = 0.3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 128),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gap(self.features(x))
        return self.classifier(x)


class TinyAgCNNGap(nn.Module):
    """Grayscale 96x96 three-convolution classifier (deployed grayscale model).

    This is the architecture of ``models/TinyAgCNN_best.keras``: three
    Conv3x3 VALID blocks (16/32/64), global average pooling, Dense(64), Dense(4).
    It has 28,052 weight elements, 27,828 of them trainable (the difference is
    the BatchNorm running statistics).

    The legacy Flatten variant in ``notebooks/legacy/`` has 843,364 parameters
    and is not the deployed artefact.
    """

    def __init__(self, num_classes: int = 4, dropout: float = 0.3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 16),
            ConvBlock(16, 32),
            ConvBlock(32, 64),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gap(self.features(x))
        return self.classifier(x)


class TinyBinaryCNN(nn.Module):
    """Healthy-vs-diseased gate model from ``06_binary_gate.ipynb``.

    This model is an experiment; it is not wired into the current firmware.
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0.3, img_size: int = 96) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        with torch.no_grad():
            flat = self.features(torch.zeros(1, 3, img_size, img_size)).flatten(1).shape[1]
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


ARCHITECTURES = {
    "v2": TinyAgCNNv2,
    "gray": TinyAgCNNGap,
    "binary": TinyBinaryCNN,
}

# The default input normalisation that matches the INT8 deployment contract.
# ``symmetric`` maps [0, 255] to [-1, 1] (pixel - 128 over 128), which is what
# the host tools send. ``unit`` keeps [0, 1] and is the original grayscale
# TensorFlow notebook's contract.
DEFAULT_NORMALIZATION = {
    "v2": "symmetric",
    "gray": "unit",
    "binary": "symmetric",
}


def build_model(arch: str, num_classes: int | None = None, dropout: float = 0.3) -> nn.Module:
    """Build a model by short architecture name."""
    if arch not in ARCHITECTURES:
        raise ValueError(f"unknown arch {arch!r}; choose from {sorted(ARCHITECTURES)}")
    if num_classes is None:
        num_classes = 2 if arch == "binary" else 4
    return ARCHITECTURES[arch](num_classes=num_classes, dropout=dropout)


def trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def total_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
