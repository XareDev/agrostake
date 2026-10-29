"""Train the AgroStake edge classifiers.

Examples
--------
RGB v2 model (deployed firmware model)::

    python -m training.src.train --arch v2 --data-root data/augmented-dataset \
        --epochs 40 --batch-size 32 --out-dir runs/v2

Grayscale GAP model (matches ``models/TinyAgCNN_best.keras``)::

    python -m training.src.train --arch gray --normalization unit \
        --epochs 30 --out-dir runs/gray

Binary healthy/diseased gate (experiment)::

    python -m training.src.train --arch binary --epochs 30 --out-dir runs/binary
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

try:  # works both as ``python -m training.src.train`` and a direct script
    from .data import build_datasets, build_loaders, save_split_metadata
    from .models import build_model, trainable_parameters
except ImportError:  # pragma: no cover
    from data import build_datasets, build_loaders, save_split_metadata
    from models import build_model, trainable_parameters


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n: int) -> np.ndarray:
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true.astype(int), y_pred.astype(int)):
        cm[t, p] += 1
    return cm


def metrics_from_confusion(cm: np.ndarray, class_names: list[str]) -> dict:
    n = cm.shape[0]
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    diag = np.diag(cm).astype(np.float64)
    per_class = []
    for i in range(n):
        tp = float(diag[i])
        rec = tp / float(support[i]) if support[i] else 0.0
        prec = tp / float(predicted[i]) if predicted[i] else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class.append(
            {
                "class": class_names[i],
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "support": int(support[i]),
            }
        )
    return {
        "accuracy": float(diag.sum() / max(1, cm.sum())),
        "macro_precision": float(np.mean([c["precision"] for c in per_class])),
        "macro_recall": float(np.mean([c["recall"] for c in per_class])),
        "macro_f1": float(np.mean([c["f1"] for c in per_class])),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "support": int(cm.sum()),
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader, criterion, device: torch.device, n_classes: int):
    model.eval()
    total_loss = 0.0
    all_true: list[int] = []
    all_pred: list[int] = []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        total_loss += float(criterion(logits, targets).item()) * images.size(0)
        all_true.extend(targets.detach().cpu().tolist())
        all_pred.extend(logits.argmax(dim=1).detach().cpu().tolist())
    y_true = np.asarray(all_true, dtype=np.int64)
    y_pred = np.asarray(all_pred, dtype=np.int64)
    cm = confusion_matrix(y_true, y_pred, n_classes)
    out = metrics_from_confusion(cm, [f"class_{i}" for i in range(n_classes)])
    out["loss"] = total_loss / max(1, len(loader.dataset))
    return out, y_true, y_pred


def train_one_epoch(model, loader, criterion, optimizer, device: torch.device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, targets in tqdm(loader, leave=False, desc="train"):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * images.size(0)
        correct += int((logits.argmax(dim=1) == targets).sum().item())
        total += images.size(0)
    return total_loss / max(1, total), correct / max(1, total)


def class_weights_from_dataset(dataset, n_classes: int) -> torch.Tensor:
    """Compute inverse-frequency weights, including for binary wrappers."""
    raw = None
    if hasattr(dataset, "dataset") and hasattr(dataset, "indices"):
        raw = [dataset.dataset.targets[i] for i in dataset.indices]
    elif hasattr(dataset, "base") and hasattr(dataset, "healthy_index"):
        # BinaryLabelDataset wrapping an ImageFolder or Subset.
        base = dataset.base
        healthy = int(dataset.healthy_index)
        if hasattr(base, "dataset") and hasattr(base, "indices"):
            raw = [base.dataset.targets[i] for i in base.indices]
        elif hasattr(base, "targets"):
            raw = base.targets
        if raw is not None:
            raw = [0 if int(t) == healthy else 1 for t in raw]
    elif hasattr(dataset, "targets"):
        raw = dataset.targets
    if raw is None:
        return torch.ones(n_classes)
    counts = np.bincount(np.asarray(raw, dtype=np.int64), minlength=n_classes)
    weights = counts.sum() / np.maximum(counts, 1) / n_classes
    return torch.tensor(weights, dtype=torch.float32)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arch", choices=["v2", "gray", "binary"], default="v2")
    p.add_argument("--data-root", default="data/augmented-dataset")
    p.add_argument("--split-file", default=None, help="npz with train_idx/val_idx")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="runs/agrostake")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--normalization", choices=["auto", "unit", "symmetric", "imagenet"], default="auto")
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--balanced", action="store_true", help="weight the loss by inverse class frequency")
    p.add_argument("--patience", type=int, default=10, help="early-stopping patience; 0 disables")
    p.add_argument("--resume", default=None)
    p.add_argument("--no-test", action="store_true", help="skip the final test evaluation")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    set_seed(args.seed)
    device = resolve_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, test_ds, class_names, meta = build_datasets(
        data_root=args.data_root,
        arch=args.arch,
        normalization=args.normalization,
        split_file=args.split_file,
        seed=args.seed,
    )
    meta["seed"] = args.seed
    meta["batch_size"] = args.batch_size
    meta["epochs"] = args.epochs
    save_split_metadata(meta, out_dir / "dataset_metadata.json")
    train_loader, val_loader, test_loader = build_loaders(
        train_ds, val_ds, test_ds, batch_size=args.batch_size, workers=args.workers
    )

    n_classes = len(class_names)
    model = build_model(args.arch, num_classes=n_classes, dropout=args.dropout).to(device)
    n_params = trainable_parameters(model)
    print(f"arch={args.arch} classes={class_names} trainable_params={n_params:,}")
    print(f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    weights = None
    if args.balanced:
        weights = class_weights_from_dataset(train_ds, n_classes).to(device)
        print("class weights:", [round(float(x), 4) for x in weights.cpu()])
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=1e-6)

    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        print(f"resumed weights from {args.resume}")

    history: list[dict] = []
    best_recall = -1.0
    best_epoch = 0
    stale = 0
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()
        val_raw, _, _ = evaluate(model, val_loader, criterion, device, n_classes)
        val_metrics = metrics_from_confusion(
            np.asarray(val_raw["confusion_matrix"], dtype=np.int64), class_names
        )
        record = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_raw["loss"],
            "val_acc": val_metrics["accuracy"],
            "val_macro_recall": val_metrics["macro_recall"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        history.append(record)
        print(
            f"epoch {epoch:03d}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_raw['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} "
            f"val_recall={val_metrics['macro_recall']:.4f} val_f1={val_metrics['macro_f1']:.4f}"
        )
        if val_metrics["macro_recall"] > best_recall + 1e-6:
            best_recall = val_metrics["macro_recall"]
            best_epoch = epoch
            stale = 0
            torch.save(model.state_dict(), out_dir / "best.pth")
            (out_dir / "best_metrics.json").write_text(
                json.dumps({"epoch": epoch, **val_metrics}, indent=2) + "\n", encoding="utf-8"
            )
        else:
            stale += 1
        torch.save(model.state_dict(), out_dir / "last.pth")
        if args.patience and stale >= args.patience:
            print(f"early stopping at epoch {epoch} (best epoch {best_epoch})")
            break

    summary = {
        "arch": args.arch,
        "args": vars(args),
        "dataset": meta,
        "trainable_params": n_params,
        "best_epoch": best_epoch,
        "best_val_macro_recall": best_recall,
        "elapsed_seconds": round(time.time() - started, 2),
        "history": history,
    }

    if not args.no_test and (out_dir / "best.pth").exists():
        model.load_state_dict(torch.load(out_dir / "best.pth", map_location="cpu", weights_only=True))
        test_raw, y_true, y_pred = evaluate(model, test_loader, criterion, device, n_classes)
        test_metrics = metrics_from_confusion(
            np.asarray(test_raw["confusion_matrix"], dtype=np.int64), class_names
        )
        test_metrics["loss"] = test_raw["loss"]
        summary["test"] = test_metrics
        np.savez(out_dir / "test_predictions.npz", y_true=y_true, y_pred=y_pred)
        print(
            f"test: acc={test_metrics['accuracy']:.4f} "
            f"recall={test_metrics['macro_recall']:.4f} f1={test_metrics['macro_f1']:.4f}"
        )

    (out_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {out_dir / 'training_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
