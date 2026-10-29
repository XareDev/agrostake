# Training

This directory contains the original notebooks and a small, reproducible
PyTorch implementation of the model family. The notebooks are the historical
record; `src/` is the code path intended for new runs.

## What is here

```text
training/
├── notebooks/
│   ├── 01_mobilenetv2_transfer_learning.ipynb
│   ├── 02_mobilenetv2_finetuning.ipynb
│   ├── 03_mobilenetv2_v2.ipynb
│   ├── 04_tinyagcnn_grayscale_pytorch.ipynb
│   ├── 05_tinyagcnn_kd_tensorflow.ipynb
│   ├── 06_binary_gate.ipynb
│   └── legacy/
│       └── 05_tinyagcnn_kd_flatten_legacy.ipynb
└── src/
    ├── models.py
    ├── data.py
    ├── train.py
    ├── evaluate.py
    └── export_onnx.py
```

## Notebook inventory

| Notebook | Purpose | Model | Status |
|---|---|---|---|
| `01_mobilenetv2_transfer_learning.ipynb` | MobileNetV2 transfer learning at 224×224 | MobileNetV2 | Phase-1 experiment |
| `02_mobilenetv2_finetuning.ipynb` | Fine-tuning / continued MobileNetV2 training | MobileNetV2 | Phase-2 experiment |
| `03_mobilenetv2_v2.ipynb` | MobileNetV2 v2 classifier experiments | MobileNetV2 | Baseline for teacher |
| `04_tinyagcnn_grayscale_pytorch.ipynb` | Early PyTorch grayscale skeleton | Flatten TinyAgCNN | Not the deployed model |
| `05_tinyagcnn_kd_tensorflow.ipynb` | Grayscale CE + KD, GAP student | **27,828-param GAP** | **Matches deployed `.keras`** |
| `06_binary_gate.ipynb` | Healthy vs diseased gate | TinyBinaryCNN | Experiment; no final test report |
| `legacy/05_...flatten...` | Original Flatten variant | 843,364-param Flatten | Historical; not deployed |

### Why there are two “05” notebooks

The deployed grayscale `.keras` files have this structure:

```text
Conv 16 → BN → ReLU → Pool
Conv 32 → BN → ReLU → Pool
Conv 64 → BN → ReLU → Pool
GlobalAveragePool → Dense(64) → Dense(4)
```

That is the GAP model with 27,828 trainable parameters. The older notebook in
`legacy/` uses `Flatten → Dense(...)`, which has 843,364 parameters. The
Flatten variant appears in `04_tinyagcnn_grayscale_pytorch.ipynb` as well. The
released `.keras`, `.tflite`, ONNX, and ESP-DL grayscale artefacts all use the
GAP architecture. Keeping both notebooks documents the change rather than
pretending it did not happen.

## Reproducible PyTorch path

Install the environment from the repository root:

```bash
pip install -r requirements.txt
```

### RGB v2

```bash
python -m training.src.train \
  --arch v2 \
  --data-root data/augmented-dataset \
  --epochs 40 \
  --batch-size 32 \
  --lr 3e-4 \
  --out-dir runs/v2
```

The v2 architecture and state-dict keys were reconstructed from
`models/TinyAgCNN_v2.onnx` and `models/tinyagcnn_v2_best.pth`. The released
checkpoint loads with `strict=True`. The v2 training notebook itself was not
preserved in the original workspace; the model card records this explicitly.

### Grayscale GAP

```bash
python -m training.src.train \
  --arch gray \
  --normalization unit \
  --data-root data/augmented-dataset \
  --epochs 30 \
  --batch-size 32 \
  --lr 1e-3 \
  --out-dir runs/gray
```

`--normalization unit` keeps the notebook's `[0,1]` input. If you are training
a model that will consume the firmware's `pixel - 128` bytes directly, use
`--normalization symmetric` and validate the exported quantisation metadata.

### Binary gate

```bash
python -m training.src.train \
  --arch binary \
  --data-root data/augmented-dataset \
  --epochs 30 \
  --batch-size 32 \
  --lr 3e-4 \
  --out-dir runs/binary
```

This uses the deployment-consistent `[-1,1]` normalisation, unlike the
preserved checkpoint, which used ImageNet normalisation. The gate is not wired
into the firmware.

## Knowledge distillation recipe (preserved notebook)

`05_tinyagcnn_kd_tensorflow.ipynb` implements cross-framework KD:

- Teacher: PyTorch MobileNetV2 (`agroStake-0.1-p2-finetuned.pth`), 224×224 RGB,
  ImageNet normalisation.
- Student: TensorFlow grayscale GAP TinyAgCNN, 96×96, `[0,1]` input.
- Teacher logits are extracted once and cached as `teacher_logits_cache.npy`.
- KD loss: `alpha * CE(student, label) + (1 - alpha) * KL(teacher || student) * T²`.
- `alpha = 0.5`, `T = 4.0`, 30 epochs, AdamW, lr `1e-3`.

The preserved run produced 60.00 % accuracy / 0.59 macro F1 for the KD student
versus 80.47 % / 0.81 for the baseline CE student. This is a negative result in
the preserved record. It is documented, not hidden.

## Dataset and splits

See [`../data/README.md`](../data/README.md). The loader uses
`data/splits/split_indices.npz` when present, so training and evaluation share
the exact same 12,999 / 1,445 split.

## Outputs

`train.py` writes to `--out-dir`:

| File | Contents |
|---|---|
| `best.pth` | raw `state_dict` of the best validation macro-recall epoch |
| `last.pth` | last epoch |
| `training_summary.json` | config, dataset metadata, history, test metrics |
| `dataset_metadata.json` | split source, sizes, class names, normalisation |
| `best_metrics.json` | validation metrics for the best epoch |
| `test_predictions.npz` | `y_true`, `y_pred` for the final test evaluation |

The checkpoints are raw state dictionaries so `evaluate.py` and
`export_onnx.py` can load them without a wrapper class.

## Evaluation and export

```bash
python -m training.src.evaluate \
  --arch v2 \
  --checkpoint models/tinyagcnn_v2_best.pth \
  --data-root data/augmented-dataset \
  --split test

python -m training.src.export_onnx \
  --arch v2 \
  --checkpoint models/tinyagcnn_v2_best.pth \
  --out models/TinyAgCNN_v2.onnx
```

`evaluate.py` writes a JSON report, a text report, and a confusion-matrix PNG.
It also reports host-side PyTorch latency; that number is not an ESP32-S3
measurement.

## Tests

From the repository root:

```bash
python -m pytest tests -q
```

The tests check the released parameter counts, checkpoint loading, ONNX export
parity, and the serial payload sizes/protocol parser. They do **not** compile
the firmware.

## Notebook portability

The notebooks contain absolute/relative paths from the original machine
(`../0_data/augmented-dataset`, `../4_models/...`). They also install packages
inside cells. Treat them as provenance, not as one-click reproductions. Use
`training/src/` for new runs and `docs/deployment.md` for the INT8 export.
