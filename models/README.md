# Model inventory

All released model binaries are in this directory. The SHA-256 digest of every
file is in [`SHA256SUMS`](SHA256SUMS); the firmware copies are listed separately
in [`../firmware/README.md`](../firmware/README.md).

## Summary

| File | Architecture | Input | Trainable params | Size | Status |
|---|---|---|---:|---:|---|
| `tinyagcnn_v2_best.pth` | RGB v2, 4×Conv + GAP | 3×96×96 | 275,236 | 1.06 MB | deployed source checkpoint |
| `TinyAgCNN_v2.onnx` | RGB v2 | 1×3×96×96 | 275,236 | 1.05 MB | float ONNX |
| `TinyAgCNN_v2_s8.espdl` | RGB v2 INT8 | NHWC 96×96×3 | 275,236 | 308.4 KB | deployed by UART-RGB / Wi-Fi / experimental |
| `TinyAgCNN_v2_s8.json` | RGB v2 quantisation config | — | — | 28.3 KB | ESP-PPQ config |
| `TinyAgCNN_best.keras` | grayscale GAP | 96×96×1 | 27,828 | 167.0 KB | grayscale source checkpoint |
| `best_baseline.keras` | grayscale GAP, baseline CE | 96×96×1 | 27,828 | 167.5 KB | 80.47 % test accuracy |
| `best_kd.keras` | grayscale GAP, KD student | 96×96×1 | 27,828 | 167.5 KB | 60.00 % test accuracy |
| `TinyAgCNN.onnx` | grayscale GAP | NHWC 1×96×96×1 | 27,828 | 112.5 KB | float ONNX (TensorFlow-style layout) |
| `TinyAgCNN_s8.onnx` | grayscale GAP | NCHW 1×1×96×96 | 27,828 | 111.7 KB | float/quant-ready ONNX |
| `TinyAgCNN_int8.tflite` | grayscale GAP INT8 | 1×96×96×1 | 27,828 | 36.3 KB | TFLite reference |
| `TinyAgCNN_s8.espdl` | grayscale GAP INT8 | NHWC 1×96×96×1 | 27,828 | 45.5 KB | deployed by UART-grayscale |
| `TinyAgCNN_s8.json` | grayscale quantisation config | — | — | 23.4 KB | ESP-PPQ config |
| `TinyBinaryCNN_v1_best.pth` | healthy/diseased gate | 3×96×96 | 497,226 | 1.90 MB | experiment, not deployed |

“Trainable params” counts parameters updated by the optimiser. Keras/PyTorch
state dictionaries also contain BatchNorm running means/variances; the
grayscale model has 28,052 weight elements in total, of which 27,828 are
trainable.

## RGB v2 model

`TinyAgCNN_v2` is a custom four-block convolutional network:

```text
Input 96×96×3
  Conv 3→32, 3×3, VALID, bias=False → BN → ReLU → MaxPool2   47×47×32
  Conv 32→64, 3×3, VALID, bias=False → BN → ReLU → MaxPool2  22×22×64
  Conv 64→128, 3×3, VALID, bias=False → BN → ReLU → MaxPool2 10×10×128
  Conv 128→128, 3×3, VALID, bias=False → BN → ReLU → MaxPool2 4×4×128
  GlobalAveragePool → 128
  Linear 128→256 → ReLU → Dropout(0.3) → Linear 256→4
```

- Source checkpoint: `tinyagcnn_v2_best.pth`.
- `TinyAgCNN_v2_s8.espdl` is the INT8 model embedded in
  `firmware/uart-rgb`, `firmware/wifi`, and
  `firmware/experimental/wifi-serial`.
- The JTAG firmware uses a separately exported 315,776-byte ESP-DL file
  (`D677878164AC19185346A021E850CE8A1967C4CB0E71D48992A375DEAF3FDC9F`).
  It has the same architecture and task but is not byte-identical to the
  315,808-byte export used by the other variants. Keep the two files with
  their respective firmware trees.
- The ONNX file uses NCHW input and opset 13. The PyTorch model's state-dict
  keys match the released checkpoint exactly; this is covered by
  `tests/test_models.py`.

The v2 training notebook was not preserved as a separate file in the original
workspace. The PyTorch class in `training/src/models.py` was reconstructed from
the exported ONNX graph and the `tinyagcnn_v2_best.pth` state dictionary; it
loads the released checkpoint with `strict=True` and reproduces the ONNX graph.
The notebook that is preserved (`05_tinyagcnn_kd_tensorflow.ipynb`) is the
grayscale KD experiment, not the v2 trainer.

## Grayscale GAP model

The deployed grayscale model is the **GAP** architecture:

```text
Input 96×96×1
  Conv 1→16, 3×3, VALID, bias=False → BN → ReLU → MaxPool2  47×47×16
  Conv 16→32, 3×3, VALID, bias=False → BN → ReLU → MaxPool2 22×22×32
  Conv 32→64, 3×3, VALID, bias=False → BN → ReLU → MaxPool2 10×10×64
  GlobalAveragePool → 64
  Dense 64→64 → ReLU → Dropout(0.3) → Dense 64→4
```

`TinyAgCNN_best.keras`, `best_baseline.keras`, and `best_kd.keras` all use this
architecture. The Flatten variant in
`training/notebooks/legacy/05_tinyagcnn_kd_flatten_legacy.ipynb` has 843,364
parameters and is a different experiment; it does **not** correspond to these
deployable artefacts. This mismatch is documented rather than silently
“fixed” because the Flatten notebook is part of the original record.

`TinyAgCNN_s8.espdl` is the grayscale artefact embedded by
`firmware/uart-grayscale`. There is no KD-specific `.espdl` in the preserved
workspace; `best_kd.keras` is a Keras checkpoint only. Do not assume that the
released `.espdl` is the KD student.

### Grayscale input encoding

The original TensorFlow notebook trained on `[0, 1]` input. The TFLite INT8
converter uses scale `1/255` and zero point `-128`, so the host sends
`pixel - 128`. The ESP-DL grayscale firmware expects those already-shifted
INT8 bytes. This is why the training notebook and the firmware byte values
look different even though they describe the same image.

## Binary gate experiment

`TinyBinaryCNN_v1_best.pth` is the healthy-vs-diseased gate from
`training/notebooks/06_binary_gate.ipynb`:

```text
Input 96×96×3
  Conv 3→8, 3×3, VALID → BN → ReLU → MaxPool2  47×47×8
  Conv 8→16, 3×3, VALID → BN → ReLU → MaxPool2 22×22×16
  Flatten 7,744 → Linear 64 → ReLU → Dropout(0.3) → Linear 2
```

It has 497,226 trainable parameters. Important caveats:

- The preserved notebook uses ImageNet mean/std normalisation for this model,
  while the deployment contract used by the firmware is `pixel - 128`
  (approximately `[-1, 1]`). The checkpoint is therefore **not** directly
  compatible with the v2 host preprocessing.
- The preserved training output stops before a final test report; no held-out
  accuracy is claimed here.
- No firmware variant loads this checkpoint. It is an experiment, not part of
  the deployed path.

If you want to use the gate, retrain it with
`python -m training.src.train --arch binary` (which defaults to symmetric
`[-1,1]` normalisation) and add it to a firmware variant explicitly.

## `.json` and `.info` files

`TinyAgCNN_s8.json` and `TinyAgCNN_v2_s8.json` are ESP-PPQ quantisation
configuration files exported alongside the `.espdl` models. They describe the
per-tensor quantisation policies and are kept because they document how the
INT8 artefacts were produced. The original `.info` files were much larger
profiling dumps and are not required for deployment; they are not included.

## Reproducing the exports

```bash
python -m training.src.export_onnx \
  --arch v2 \
  --checkpoint models/tinyagcnn_v2_best.pth \
  --out models/TinyAgCNN_v2.onnx
```

For the grayscale model, the released checkpoints are Keras files, so use the
TensorFlow notebook (`training/notebooks/05_tinyagcnn_kd_tensorflow.ipynb`) or
train an equivalent PyTorch model with:

```bash
python -m training.src.train --arch gray --normalization unit --out-dir runs/gray
python -m training.src.export_onnx --arch gray --checkpoint runs/gray/best.pth --out runs/gray/model.onnx
```

The ONNX export is NCHW; ESP-PPQ converts it for the ESP-DL runtime. The exact
ESP-DL quantisation commands are documented in
[`../docs/deployment.md`](../docs/deployment.md).

## Integrity

Run this from the repository root:

```bash
# Linux/macOS
cd models && sha256sum -c SHA256SUMS
```

```powershell
# Windows PowerShell, from the repository root
Get-Content models\SHA256SUMS | ForEach-Object {
  $hash, $file = $_ -split '\s+', 2
  if ((Get-FileHash (Join-Path models $file.Trim()) -Algorithm SHA256).Hash -ne $hash) {
    throw "hash mismatch: $file"
  }
}
```
