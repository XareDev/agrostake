"""Dataset and preprocessing helpers for AgroStake.

The original experiments used an ImageFolder layout::

    augmented-dataset/
        train/
            Black Rot/
            ESCA/
            Healthy/
            Leaf Blight/
        test/
            ...

``train`` contains the 14,444-image train+validation pool (12,999 / 1,445)
and ``test`` contains the 3,610 held-out images.  ``data/splits/split_indices.npz``
stores the exact train/validation split used by the notebooks, so the same
hold-out is reproducible without copying files.

Preprocessing is deliberately explicit.  The firmware receives ``pixel - 128``
for every channel (a [-1, 1] tensor after the INT8 input exponent is applied),
which corresponds to ``symmetric`` normalisation.  The original grayscale
TensorFlow model was trained on [0, 1] and is reproduced with ``unit``.
``imagenet`` is kept for the old MobileNetV2/teacher experiments and for the
legacy binary-gate checkpoint.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

IMG_SIZE = 96
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
NORMALIZATIONS = ("unit", "symmetric", "imagenet")

# The names used by the firmware, in the alphabetical order produced by
# torchvision's ImageFolder for the four class directories.
FIRMWARE_CLASSES = ("Black_rot", "Esca", "Healthy", "Leaf_blight")
DISPLAY_CLASSES = ("Black Rot", "ESCA", "Healthy", "Leaf Blight")


def _norm_name(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


_NAME_TO_FIRMWARE = {
    _norm_name("Black Rot"): "Black_rot",
    _norm_name("Black_rot"): "Black_rot",
    _norm_name("ESCA"): "Esca",
    _norm_name("Esca"): "Esca",
    _norm_name("Healthy"): "Healthy",
    _norm_name("Leaf Blight"): "Leaf_blight",
    _norm_name("Leaf_blight"): "Leaf_blight",
}


def class_name_map(class_names: Sequence[str]) -> list[str]:
    """Map ImageFolder class names to the firmware's stable names."""
    mapped = []
    for name in class_names:
        key = _norm_name(name)
        if key not in _NAME_TO_FIRMWARE:
            raise ValueError(f"unrecognised class directory {name!r}")
        mapped.append(_NAME_TO_FIRMWARE[key])
    if len(set(mapped)) != len(mapped):
        raise ValueError(f"duplicate classes after mapping: {mapped}")
    return mapped


def build_transform(
    arch: str,
    normalization: str = "auto",
    train: bool = True,
    img_size: int = IMG_SIZE,
) -> transforms.Compose:
    """Build the preprocessing pipeline for an architecture.

    Parameters
    ----------
    arch:
        ``v2`` (RGB), ``gray`` (single-channel) or ``binary`` (RGB).
    normalization:
        ``auto`` uses the deployment-consistent default for the architecture,
        or one of ``unit``, ``symmetric``, ``imagenet``.
    """
    if normalization == "auto":
        normalization = "unit" if arch == "gray" else "symmetric"
    if normalization not in NORMALIZATIONS:
        raise ValueError(f"normalization must be one of {NORMALIZATIONS} or 'auto'")

    ops: list = []
    if arch == "gray":
        ops.append(transforms.Grayscale(num_output_channels=1))
    if train:
        ops.extend(
            [
                transforms.RandomResizedCrop(
                    img_size, scale=(0.70, 1.0), ratio=(0.80, 1.25)
                ),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
            ]
        )
        if arch == "gray":
            ops.append(transforms.ColorJitter(brightness=0.20, contrast=0.20))
        else:
            ops.append(
                transforms.ColorJitter(
                    brightness=0.20, contrast=0.20, saturation=0.20, hue=0.03
                )
            )
    else:
        ops.extend(
            [
                transforms.Resize(img_size),
                transforms.CenterCrop(img_size),
            ]
        )

    ops.append(transforms.ToTensor())
    channels = 1 if arch == "gray" else 3
    if normalization == "symmetric":
        ops.append(transforms.Normalize([0.5] * channels, [0.5] * channels))
    elif normalization == "imagenet":
        if channels != 3:
            raise ValueError("ImageNet normalisation requires an RGB model")
        ops.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    return transforms.Compose(ops)


class IndexedImageFolder(ImageFolder):
    """ImageFolder that also returns the dataset index (for cached logits)."""

    def __getitem__(self, index: int):
        image, label = super().__getitem__(index)
        return image, label, index


class BinaryLabelDataset(Dataset):
    """Wrap a 4-class dataset and collapse it to healthy vs diseased."""

    def __init__(self, base: Dataset, healthy_index: int) -> None:
        self.base = base
        self.healthy_index = healthy_index

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        item = self.base[index]
        image, label = item[0], item[1]
        binary = 0 if int(label) == self.healthy_index else 1
        if len(item) == 3:
            return image, binary, item[2]
        return image, binary

    @property
    def class_names(self) -> list[str]:
        return ["Healthy", "Diseased"]

    @property
    def class_to_idx(self) -> dict[str, int]:
        return {"Healthy": 0, "Diseased": 1}


def _stratified_split(
    targets: Sequence[int], val_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    targets_arr = np.asarray(targets)
    for label in np.unique(targets_arr):
        idx = np.flatnonzero(targets_arr == label)
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_fraction)))
        val_parts.append(idx[:n_val])
        train_parts.append(idx[n_val:])
    train_idx = np.sort(np.concatenate(train_parts))
    val_idx = np.sort(np.concatenate(val_parts))
    return train_idx, val_idx


def _load_split_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    if "train_idx" not in data or "val_idx" not in data:
        raise ValueError(f"{path} must contain train_idx and val_idx")
    return data["train_idx"].astype(np.int64), data["val_idx"].astype(np.int64)


def build_datasets(
    data_root: str | Path,
    arch: str,
    normalization: str = "auto",
    split_file: str | Path | None = None,
    val_fraction: float = 0.10,
    seed: int = 42,
    img_size: int = IMG_SIZE,
    binary: bool | None = None,
    indexed: bool = False,
) -> tuple[Dataset, Dataset, Dataset, list[str], dict]:
    """Build train/val/test datasets and return metadata.

    Returns ``(train, val, test, class_names, metadata)``.  ``class_names`` are
    the original ImageFolder names, in label order; ``metadata['firmware_classes']``
    gives the corresponding firmware names.
    """
    root = Path(data_root)
    train_dir = root / "train"
    if not train_dir.is_dir():
        raise FileNotFoundError(f"expected ImageFolder split at {train_dir}")
    if binary is None:
        binary = arch == "binary"

    folder_cls = IndexedImageFolder if indexed else ImageFolder
    train_tf = build_transform(arch, normalization, train=True, img_size=img_size)
    eval_tf = build_transform(arch, normalization, train=False, img_size=img_size)

    base_train = folder_cls(str(train_dir), transform=train_tf)
    class_names = list(base_train.classes)
    fw_classes = class_name_map(class_names)
    if len(class_names) != 4:
        raise ValueError(f"expected 4 classes, found {class_names}")

    # Prefer an explicit val/ directory; otherwise reproduce the notebook split.
    val_dir = root / "val"
    if val_dir.is_dir():
        train_ds = base_train
        val_ds = folder_cls(str(val_dir), transform=eval_tf)
        split_source = "directory"
    else:
        if split_file is None:
            split_file = (
                Path(__file__).resolve().parents[2]
                / "data"
                / "splits"
                / "split_indices.npz"
            )
        split_path = Path(split_file)
        if split_path.exists():
            train_idx, val_idx = _load_split_file(split_path)
            split_source = str(split_path)
        else:
            train_idx, val_idx = _stratified_split(base_train.targets, val_fraction, seed)
            split_source = f"stratified:{val_fraction}:{seed}"
        if max(int(train_idx.max()), int(val_idx.max())) >= len(base_train):
            raise ValueError(
                f"split indices exceed dataset length {len(base_train)}; "
                "is this the 14,444-image train pool?"
            )
        train_ds = Subset(base_train, train_idx.tolist())
        val_ds = Subset(folder_cls(str(train_dir), transform=eval_tf), val_idx.tolist())

    test_dir = root / "test"
    test_ds: Dataset
    if test_dir.is_dir():
        test_ds = folder_cls(str(test_dir), transform=eval_tf)
    else:
        test_ds = val_ds  # explicit fallback; metadata records it
        split_source += "; test=val"

    healthy_idx = class_names.index(next(n for n in class_names if _norm_name(n) == "healthy"))
    if binary:
        train_ds = BinaryLabelDataset(train_ds, healthy_idx)
        val_ds = BinaryLabelDataset(val_ds, healthy_idx)
        test_ds = BinaryLabelDataset(test_ds, healthy_idx)
        out_classes = ["Healthy", "Diseased"]
        fw_classes = ["Healthy", "Diseased"]
    else:
        out_classes = class_names

    metadata = {
        "data_root": str(root.resolve()),
        "split_source": split_source,
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "test_size": len(test_ds),
        "class_names": out_classes,
        "firmware_classes": fw_classes,
        "normalization": normalization if normalization != "auto" else ("unit" if arch == "gray" else "symmetric"),
        "arch": arch,
        "img_size": img_size,
    }
    return train_ds, val_ds, test_ds, out_classes, metadata


def build_loaders(
    train_ds: Dataset,
    val_ds: Dataset,
    test_ds: Dataset,
    batch_size: int = 32,
    workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    pin = torch.cuda.is_available()
    common = dict(num_workers=workers, pin_memory=pin, persistent_workers=workers > 0)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **common)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **common)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **common)
    return train_loader, val_loader, test_loader


def save_split_metadata(metadata: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
