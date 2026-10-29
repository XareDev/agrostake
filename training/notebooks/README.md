# Notebook inventory

The notebooks are the historical experiment record. They contain original
absolute paths, package installs, and cell outputs; they are not one-click
reproductions. Use `training/src/` for new runs.

| Notebook | Purpose | Role |
|---|---|---|
| `01_mobilenetv2_transfer_learning.ipynb` | MobileNetV2 transfer learning at 224×224 | teacher / phase-1 experiment |
| `02_mobilenetv2_finetuning.ipynb` | Continued MobileNetV2 training | teacher / phase-2 experiment |
| `03_mobilenetv2_v2.ipynb` | MobileNetV2 v2 classifier experiments | baseline / teacher |
| `04_tinyagcnn_grayscale_pytorch.ipynb` | Early PyTorch grayscale skeleton | historical; Flatten head |
| `05_tinyagcnn_kd_tensorflow.ipynb` | Grayscale CE + KD, GAP student | **matches released grayscale artefacts** |
| `06_binary_gate.ipynb` | Healthy/diseased gate | experiment; no final test report |
| `legacy/05_tinyagcnn_kd_flatten_legacy.ipynb` | Original Flatten variant | historical; 843,364 params, not deployed |

## The important notebook

`05_tinyagcnn_kd_tensorflow.ipynb` is the source of the released grayscale
metrics:

- Baseline CE: 80.47 % accuracy, 0.8118 macro F1.
- KD student: 60.00 % accuracy, 0.59 macro F1.
- Both use the 27,828-parameter GAP architecture.

The older Flatten notebook has a different head (`Flatten → Dense`) and 843,364
parameters. It does **not** describe the released `.keras`, `.tflite`, ONNX, or
ESP-DL files. It is kept under `legacy/` so the evolution is visible.

## Paths and outputs

The notebooks reference paths such as:

```text
../0_data/augmented-dataset
../4_models/tinyagcnn/...
teacher_logits_cache.npy
```

Those paths reflect the original workspace. To reuse a notebook, point the data
and output cells at your own copy and keep the split from
`../data/splits/split_indices.npz` if you want to compare with the reported
numbers.

## Notebook-specific caveats

- The grayscale KD notebook uses TensorFlow/Keras; the reusable scripts use
  PyTorch. The two are not numerically identical.
- The binary-gate notebook uses ImageNet normalisation, while the firmware
  contract is `pixel - 128`. The checkpoint is not directly deployable without
  matching preprocessing.
- The v2 RGB training notebook was not preserved. The v2 PyTorch class in
  `training/src/models.py` was reconstructed from the checkpoint and ONNX graph.
- Notebook outputs may include figures that are also copied to `docs/figures/`.
