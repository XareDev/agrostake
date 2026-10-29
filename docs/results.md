# Results

All four-class numbers below use the same held-out test set of **3,610 images**
with class supports:

| Class | Support |
|---|---:|
| Black Rot | 944 |
| ESCA | 960 |
| Healthy | 846 |
| Leaf Blight | 860 |
| **Total** | **3,610** |

The numbers are copied from the preserved notebook outputs and confusion-matrix
figures. They were not re-estimated during repository cleanup, because the
image dataset is not included.

## Headline table

| Model | Input | Trainable params | Accuracy | Macro precision | Macro recall | Macro F1 |
|---|---|---:|---:|---:|---:|---:|
| RGB v2 (`tinyagcnn_v2_best.pth`) | 96×96×3 | 275,236 | **98.61 %** | **98.62 %** | **98.66 %** | **98.64 %** |
| Grayscale baseline CE (`best_baseline.keras`) | 96×96×1 | 27,828 | 80.47 % | 82.00 %* | 80.70 % | 81.18 % |
| Grayscale KD student (`best_kd.keras`) | 96×96×1 | 27,828 | 60.00 % | 61.00 %* | 60.76 % | 59.00 % |
| Binary gate (`TinyBinaryCNN_v1_best.pth`) | 96×96×3 | 497,226 | not recorded | — | — | — |

\* The preserved printed report rounds macro precision to two decimals; the
exact unrounded value is not recoverable from the saved output. The other
columns come from the printed report or the confusion matrix.

## RGB v2

Source: `docs/figures/v2_confusion_matrix.png` and the corresponding preserved
run output. Confusion matrix (rows = true, columns = predicted):

```text
                 pred Black  pred ESCA  pred Healthy  pred Leaf
true Black Rot          915         13             0         16
true ESCA                14        946             0          0
true Healthy              0          0           846          0
true Leaf Blight          0          0             7        853
```

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Black Rot | 98.49 % | 96.93 % | 97.70 % | 944 |
| ESCA | 98.64 % | 98.54 % | 98.59 % | 960 |
| Healthy | 99.18 % | 100.00 % | 99.59 % | 846 |
| Leaf Blight | 98.16 % | 99.19 % | 98.67 % | 860 |
| **Macro** | **98.62 %** | **98.66 %** | **98.64 %** | **3,610** |

Accuracy = 3,560 / 3,610 = **98.61 %**.

### Interpretation

This is the strongest model in the repository and the only one deployed in all
four maintained RGB firmware variants. The errors are concentrated in Black
Rot (31 errors) and Leaf Blight (23 errors), with a small Black Rot ↔ Leaf
Blight confusion. ESCA and Healthy are nearly saturated on this split.

The caveat is important: the test split belongs to the same augmented image
pool as training. The result demonstrates that the model, preprocessing, and
quantisation path are internally consistent; it does **not** establish
field-level generalisation across vineyards, cameras, seasons, or lighting.
No external test set is included.

## Grayscale baseline CE

Source: `training/notebooks/05_tinyagcnn_kd_tensorflow.ipynb`, preserved test
output.

```text
Accuracy : 0.8047
F1 Macro : 0.8118
Recall   : 0.8070
```

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Black Rot | 0.64 | 0.72 | 0.68 | 944 |
| ESCA | 0.79 | 0.81 | 0.80 | 960 |
| Healthy | 0.95 | 0.91 | 0.93 | 846 |
| Leaf Blight | 0.90 | 0.78 | 0.84 | 860 |
| **Macro avg** | **0.82** | **0.81** | **0.81** | **3,610** |
| Weighted avg | 0.81 | 0.80 | 0.81 | 3,610 |

The baseline is weakest on Black Rot and Leaf Blight, the same two classes that
dominate the v2 errors. The deployed grayscale artefact is this 27,828-parameter
GAP model, not the 843,364-parameter Flatten variant from the legacy notebook.

## Grayscale KD student

Source: same notebook, preserved `KD — Test Set` output.

```text
Accuracy : 0.60
Macro Recall: 0.6076
```

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Black Rot | 0.49 | 0.45 | 0.47 | 944 |
| ESCA | 0.77 | 0.39 | 0.52 | 960 |
| Healthy | 0.60 | 0.96 | 0.74 | 846 |
| Leaf Blight | 0.60 | 0.63 | 0.62 | 860 |
| **Macro avg** | **0.61** | **0.61** | **0.59** | **3,610** |
| Weighted avg | 0.62 | 0.60 | 0.58 | 3,610 |

The KD student collapsed toward the majority-ish Healthy class and lost most
ESCA recall. It is a **negative result** in the preserved record: distillation
did not improve the small grayscale student under this recipe. `best_kd.keras`
is included so the result can be reproduced or diagnosed, not because it is a
recommended deployment model.

## Binary gate

The healthy/diseased gate (`TinyBinaryCNN`, 497,226 parameters) was an
experiment. The preserved notebook output stops at epoch 14, before a final
test report, so no held-out accuracy is claimed here. The checkpoint is kept
under `models/TinyBinaryCNN_v1_best.pth`.

Two further caveats:

- The notebook trained with ImageNet mean/std normalisation, while the
  deployment serial contract is `pixel - 128` (`[-1,1]`). The checkpoint is
  therefore not directly compatible with the v2 host preprocessing.
- No firmware variant loads this checkpoint.

To turn the idea into a deployable gate, retrain with
`python -m training.src.train --arch binary` (which uses symmetric `[-1,1]`
normalisation) and add an explicit firmware path.

## Quantisation and deployment artefacts

| Model | Float checkpoint | ONNX | INT8 deployable | Deployable size |
|---|---|---|---:|---:|
| RGB v2 | 1.06 MB `.pth` | 1.05 MB | `.espdl` | 308.4 KB |
| Grayscale GAP | 167.0 KB `.keras` | 112.5 KB / 111.7 KB | `.espdl` | 45.5 KB |
| Grayscale GAP | 167.5 KB `.keras` (baseline/KD) | — | TFLite 36.3 KB | 36.3 KB |

The `.espdl` files are the deployment artefacts embedded in firmware. Their
SHA-256 digests differ between the shared v2 export and the JTAG export; see
[`../models/README.md`](../models/README.md) and
[`../firmware/README.md`](../firmware/README.md).

No on-device latency benchmark is included. The firmware measures inference
time with `esp_timer_get_time()` and returns it in the `TIME` field; that value
depends on the board, PSRAM speed, flash mode, and whether the model is cached.
Do not treat host-side PyTorch timing from `evaluate.py` as an ESP32-S3 number.

## Figure inventory

| Figure | Contents |
|---|---|
| `docs/figures/v2_confusion_matrix.png` | RGB v2 held-out confusion matrix |
| `docs/figures/v2_training_curves.png` | RGB v2 training/validation curves |
| `docs/figures/cm_Baseline_CE.png` | Grayscale baseline CE confusion matrix |
| `docs/figures/cm_KD.png` | Grayscale KD student confusion matrix |
| `docs/figures/training_curves.png` | Grayscale baseline/KD training curves |

## Reproducibility notes

- The split indices are in `data/splits/split_indices.npz`.
- The grayscale metrics come from the GAP notebook, not the Flatten legacy
  notebook.
- The v2 metrics are derived from the preserved confusion matrix; the
  per-class percentages above are rounded to two decimals.
- The binary gate has no final test metrics because the preserved run stopped
  early. Do not infer a number from the checkpoint.
