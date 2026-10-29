# Documentation index

The top-level [`README.md`](../README.md) is the entry point. The documents here
go deeper into the implementation and the evidence behind the released models.

| Document | Read it for |
|---|---|
| [`architecture.md`](architecture.md) | Model layers, preprocessing contracts, serial/Wi-Fi data flow, firmware task model, memory layout |
| [`versions.md`](versions.md) | Which firmware/model came from which variant, hashes, and the GAP vs Flatten provenance issue |
| [`results.md`](results.md) | Full test metrics, confusion matrices, per-class numbers, negative results, and caveats |
| [`deployment.md`](deployment.md) | ESP-IDF build, board settings, model embedding, INT8 path, serial and Wi-Fi use, troubleshooting |
| [`references.md`](references.md) | Dataset, MobileNetV2, distillation, quantisation, and upstream tool citations |
| [`archive/README.md`](archive/README.md) | Caveat for the early design PDF |

## Figures

| Figure | Description |
|---|---|
| [`figures/v2_confusion_matrix.png`](figures/v2_confusion_matrix.png) | RGB v2 held-out confusion matrix |
| [`figures/v2_training_curves.png`](figures/v2_training_curves.png) | RGB v2 training curves |
| [`figures/cm_Baseline_CE.png`](figures/cm_Baseline_CE.png) | Grayscale baseline CE confusion matrix |
| [`figures/cm_KD.png`](figures/cm_KD.png) | Grayscale KD student confusion matrix |
| [`figures/training_curves.png`](figures/training_curves.png) | Grayscale baseline/KD training curves |

## Evidence hierarchy

When documents disagree, use this order:

1. A released model file plus `models/SHA256SUMS` (actual bytes).
2. The preserved notebook output / confusion matrix (`training/notebooks/` and
   `docs/figures/`).
3. The reproducible scripts in `training/src/`.
4. Narrative documents (including this one).
5. The early design PDF in `archive/`, which is historical only.

## Scope

These documents describe the repository contents. They do not claim that the
firmware has been re-flashed or re-measured after the cleanup, and they do not
claim field-level generalisation beyond the provided test split. The
[`results.md`](results.md) scope note is the authoritative statement on that
point.
