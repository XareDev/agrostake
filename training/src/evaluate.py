"""Evaluate a trained AgroStake checkpoint on a split."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from .data import build_datasets
    from .models import build_model
    from .train import confusion_matrix, metrics_from_confusion, resolve_device
except ImportError:  # pragma: no cover
    from data import build_datasets
    from models import build_model
    from train import confusion_matrix, metrics_from_confusion, resolve_device


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arch", choices=["v2", "gray", "binary"], required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root", default="data/augmented-dataset")
    p.add_argument("--split-file", default=None)
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--normalization", choices=["auto", "unit", "symmetric", "imagenet"], default="auto")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default=None, help="output directory (default: beside checkpoint)")
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args(argv)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        y_true.extend(targets.tolist())
        y_pred.extend(logits.argmax(dim=1).detach().cpu().tolist())
    return np.asarray(y_true), np.asarray(y_pred)


@torch.no_grad()
def benchmark(model, arch: str, device, runs: int = 50) -> dict:
    model.eval()
    channels = 1 if arch == "gray" else 3
    x = torch.randn(1, channels, 96, 96, device=device)
    for _ in range(5):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return {
        "device": str(device),
        "runs": runs,
        "ms_per_image": round(1000.0 * elapsed / runs, 3),
        "images_per_second": round(runs / elapsed, 2),
        "note": "host-side PyTorch timing; not an ESP32-S3 measurement",
    }


def save_confusion_plot(cm: np.ndarray, class_names: list[str], path: Path, title: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"matplotlib unavailable ({exc}); skipping plot")
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
    threshold = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(int(cm[i, j]), "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
            )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = resolve_device(args.device)
    out_dir = Path(args.out) if args.out else Path(args.checkpoint).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, test_ds, class_names, meta = build_datasets(
        data_root=args.data_root,
        arch=args.arch,
        normalization=args.normalization,
        split_file=args.split_file,
    )
    ds = {"train": train_ds, "val": val_ds, "test": test_ds}[args.split]
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    model = build_model(args.arch, num_classes=len(class_names)).to(device)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    y_true, y_pred = predict(model, loader, device)
    cm = confusion_matrix(y_true, y_pred, len(class_names))
    metrics = metrics_from_confusion(cm, class_names)
    metrics["checkpoint"] = str(Path(args.checkpoint).resolve())
    metrics["split"] = args.split
    metrics["dataset"] = meta
    metrics["benchmark"] = benchmark(model, args.arch, device)

    name = Path(args.checkpoint).stem
    (out_dir / f"eval_{name}_{args.split}.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        f"checkpoint: {metrics['checkpoint']}",
        f"split: {args.split} ({metrics['support']} images)",
        f"accuracy: {metrics['accuracy']:.4f}",
        f"macro precision: {metrics['macro_precision']:.4f}",
        f"macro recall: {metrics['macro_recall']:.4f}",
        f"macro F1: {metrics['macro_f1']:.4f}",
        "",
        "class                 precision  recall  f1      support",
    ]
    for c in metrics["per_class"]:
        lines.append(
            f"{c['class']:<20} {c['precision']:9.4f} {c['recall']:7.4f} "
            f"{c['f1']:7.4f} {c['support']:7d}"
        )
    lines.append("")
    lines.append("confusion matrix (rows=true, cols=pred):")
    lines.append(str(cm))
    report = "\n".join(lines) + "\n"
    (out_dir / f"eval_{name}_{args.split}.txt").write_text(report, encoding="utf-8")
    print(report)
    if not args.no_plot:
        save_confusion_plot(
            cm,
            class_names,
            out_dir / f"eval_{name}_{args.split}_cm.png",
            f"AgroStake {args.arch} — {args.split}",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
