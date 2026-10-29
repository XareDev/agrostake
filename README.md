# AgroStake

**Edge AI for grapevine disease detection on the ESP32-S3.**

AgroStake classifies a 96×96 grapevine-leaf image into one of four classes —
**Black Rot**, **ESCA**, **Healthy**, or **Leaf Blight** — and runs entirely on
an ESP32-S3. The repository contains the training code, the exported PyTorch /
ONNX / TensorFlow Lite / ESP-DL models, five firmware variants, and the host
tools used to drive them.

The project started as a grapevine-disease classifier and grew into a study of
the full TinyML deployment path: dataset splitting, knowledge distillation,
INT8 post-training quantisation, ESP-DL model export, UART and Wi-Fi protocols,
and the gap between notebook accuracy and on-device behaviour. The released
artefacts are kept as they were measured; the documentation records what is
known and what is still open.

## Results

All four-class numbers below use the same held-out test set of **3,610 images**
(class supports: 944 / 960 / 846 / 860). They are copied from the preserved
notebook outputs and confusion matrices, not re-estimated during repository
cleanup.

| Model | Input | Trainable params | Test accuracy | Macro F1 | Released artefact |
|---|---|---:|---:|---:|---|
| **TinyAgCNN-v2** (RGB, deployed) | 96×96×3 | 275,236 | **98.61 %** | **98.64 %** | `TinyAgCNN_v2_s8.espdl` (308 KB) |
| TinyAgCNN grayscale, baseline CE | 96×96×1 | 27,828 | 80.47 % | 81.18 % | `TinyAgCNN_s8.espdl` (45.5 KB) |
| TinyAgCNN grayscale, KD student | 96×96×1 | 27,828 | 60.00 % | 59.00 % | `best_kd.keras` (167.5 KB) |
| TinyBinaryCNN gate (experiment) | 96×96×3 | 497,226 | not recorded | not recorded | `TinyBinaryCNN_v1_best.pth` (1.90 MB) |

The RGB v2 model is the strongest model and is used by the UART-RGB, JTAG,
Wi-Fi, and experimental firmware. The grayscale model is the original v1 path
used by `firmware/uart-grayscale`. The KD student underperformed its baseline
in the preserved run, so it is included for reproducibility rather than
recommended for deployment. The binary gate was an experiment that was not
wired into the firmware and whose training run did not reach a recorded test
evaluation.

Per-class details are in [`docs/results.md`](docs/results.md), and the raw
figures are in [`docs/figures/`](docs/figures/).

> **Scope note:** the v2 score is measured on the provided test split, which
> comes from the same augmented image pool as training. It is not an external
> field-validation result. Treat it as an upper-bound / reproducibility number
> until the model is tested on images from a different vineyard, device, and
> season.

## Repository layout

```text
agrostake/
├── training/
│   ├── notebooks/           # original experiments (01–06, plus a legacy Flatten notebook)
│   └── src/                 # reproducible PyTorch training, evaluation and ONNX export
├── models/                  # released checkpoints, Keras/TFLite/ONNX/ESP-DL artefacts
├── firmware/
│   ├── uart-grayscale/      # v1 grayscale over UART0
│   ├── uart-rgb/            # v2 RGB over UART0
│   ├── jtag/                # v2 RGB over USB-Serial-JTAG
│   ├── wifi/                # v2 RGB SoftAP + browser UI
│   └── experimental/        # WIP serial + Wi-Fi variant
├── host/
│   ├── serial/send_image.py # one-shot CLI for all serial firmware
│   └── gui/                 # PyQt6 desktop application
├── data/splits/             # exact train/validation indices (no images)
├── docs/                    # architecture, versions, results, deployment, references
├── tests/                   # model, export and host-protocol smoke tests
├── LICENSE                  # MIT for code
└── LICENSE-DOCS             # CC BY-NC-SA 4.0 for documentation and released metadata
```

## Quick start

### 1. Train a model

The dataset is not included. Put an ImageFolder-style dataset at
`data/augmented-dataset/` with `train/` and `test/` splits (see
[`data/README.md`](data/README.md)). Then:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate

pip install -r requirements.txt

python -m training.src.train \
  --arch v2 \
  --data-root data/augmented-dataset \
  --epochs 40 \
  --batch-size 32 \
  --out-dir runs/v2
```

`training/src/train.py` uses the exact split stored in
`data/splits/split_indices.npz` when it is present. It writes `best.pth`,
`last.pth`, `training_summary.json`, and test predictions. The released
`models/tinyagcnn_v2_best.pth` can be evaluated directly:

```bash
python -m training.src.evaluate \
  --arch v2 \
  --checkpoint models/tinyagcnn_v2_best.pth \
  --data-root data/augmented-dataset \
  --split test
```

### 2. Export to ONNX

```bash
python -m training.src.export_onnx \
  --arch v2 \
  --checkpoint models/tinyagcnn_v2_best.pth \
  --out models/TinyAgCNN_v2.onnx
```

The export uses a static batch of 1, NCHW input, opset 13, and verifies the
result with ONNX Runtime. ESP-PPQ / ESP-DL performs the NHWC runtime layout
conversion during quantisation. See [`docs/deployment.md`](docs/deployment.md)
for the INT8 ESP-DL workflow.

### 3. Flash firmware

Each firmware variant is a standalone ESP-IDF project:

```bash
cd firmware/wifi
idf.py set-target esp32s3
idf.py build
idf.py -p COM7 flash monitor
```

The projects use `espressif/esp-dl ^3.3.2` through `main/idf_component.yml`, so
ESP-IDF downloads the matching component on the first build. The defaults
assume a 16 MB quad-flash board with 8 MB octal PSRAM at 80 MHz. Adjust
`sdkconfig.defaults` if your board differs.

| Variant | Transport | Console | Model |
|---|---|---|---|
| `uart-grayscale` | UART0, 115200 8N1 | USB-Serial-JTAG | `TinyAgCNN_s8.espdl` |
| `uart-rgb` | UART0, 115200 8N1 | USB-Serial-JTAG | `TinyAgCNN_v2_s8.espdl` |
| `jtag` | USB-Serial-JTAG | UART0 | `TinyAgCNN_v2_s8.espdl` |
| `wifi` | SoftAP `AgroStake`, HTTP `192.168.4.1` | USB-Serial-JTAG | `TinyAgCNN_v2_s8.espdl` |
| `experimental/wifi-serial` | USB-Serial-JTAG + SoftAP | USB-Serial-JTAG | `TinyAgCNN_v2_s8.espdl` |

### 4. Send an image from the host

```bash
python host/serial/send_image.py \
  --port COM7 \
  --mode uart-v2 \
  --image path/to/leaf.jpg \
  --json
```

Modes: `uart-v1` (9,216 bytes), `uart-v2` (27,648 bytes), and `jtag`
(27,648 bytes). The firmware expects the magic byte `0xAA` followed by
already-shifted INT8 pixels (`pixel - 128`). The Wi-Fi firmware is driven from
a browser instead; see [`firmware/README.md`](firmware/README.md).

## Models

| File | What it is |
|---|---|
| `models/tinyagcnn_v2_best.pth` | RGB v2 PyTorch checkpoint (275,236 params), source of the deployed v2 model |
| `models/TinyAgCNN_v2.onnx` | RGB v2 ONNX, input `[1,3,96,96]`, output `[1,4]` |
| `models/TinyAgCNN_v2_s8.espdl` | INT8 ESP-DL model used by UART-RGB, Wi-Fi and experimental firmware; JTAG embeds a separate v2 export |
| `models/TinyAgCNN_best.keras` | Grayscale GAP model (27,828 params) matching the deployed grayscale artefacts |
| `models/best_baseline.keras` | Grayscale baseline CE checkpoint (80.47 % accuracy) |
| `models/best_kd.keras` | Grayscale KD student checkpoint (60.00 % accuracy) |
| `models/TinyAgCNN_s8.espdl` | INT8 grayscale ESP-DL model (45.5 KB) |
| `models/TinyAgCNN_int8.tflite` | Grayscale TFLite INT8 model (36.3 KB) |
| `models/TinyBinaryCNN_v1_best.pth` | Healthy/diseased gate experiment (497,226 params) |

Every released binary has a SHA-256 entry in
[`models/SHA256SUMS`](models/SHA256SUMS). The per-file provenance, input
shapes, and known quirks are documented in [`models/README.md`](models/README.md).

The original experiments are preserved in `training/notebooks/`. The primary
grayscale notebook is the GAP architecture that matches the `.keras`
artefacts; the Flatten variant is retained under
`training/notebooks/legacy/` and is **not** the deployed model. This
distinction matters because the Flatten model has 843,364 parameters while the
deployed GAP model has 27,828.

## Protocols

The serial firmware uses a simple binary frame:

```text
host -> device: 0xAA | 9,216 or 27,648 signed INT8 bytes
device -> host: CLASS:<name> SCORE:<int8> CONF:<0-100> TIME:<ms>ms
```

`SCORE` is the raw INT8 logit. `CONF` is a softmax over the four INT8 logits,
provided as a relative confidence and **not** a calibrated probability. A host
that predates `CONF` can still parse `SCORE` and `TIME`.

The Wi-Fi firmware exposes `POST /detect` with a raw 27,648-byte RGB888 body and
returns JSON:

```json
{"class": "Healthy", "confidence": 87, "time_ms": 412}
```

The full protocol and wiring notes are in
[`firmware/README.md`](firmware/README.md) and
[`docs/architecture.md`](docs/architecture.md).

## Known limitations

- **Dataset not included.** The augmented grape-leaf dataset is large and has
  its own upstream terms; only the split indices are shipped.
- **No field validation.** The 98.61 % v2 result is on the provided held-out
  split, not on new field images.
- **KD is a negative result here.** The preserved KD student reached 60.00 %
  accuracy versus 80.47 % for the baseline CE model. The KD notebook is kept
  for reproducibility, not as a recommended recipe.
- **Binary gate is incomplete.** The notebook's training run stopped before a
  final test report, and its ImageNet-normalised preprocessing does not match
  the deployment `[-1,1]` contract. It is not wired into any firmware.
- **Firmware build status is partial.** The `firmware/jtag` variant was
  build-checked during integration with ESP-IDF v6.0 and ESP-DL 3.3.2; the other
  variants were not recompiled. Build and validate on your board before relying
  on timing numbers. The original on-device logs are not part of this
  repository.
- **The experimental firmware is WIP.** It combines serial and Wi-Fi paths and
  is intentionally isolated from the four maintained variants.

## Documentation

| Document | Contents |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | Models, data flow, firmware tasks, memory layout |
| [`docs/versions.md`](docs/versions.md) | Branch/firmware/model provenance map |
| [`docs/results.md`](docs/results.md) | Full metrics, confusion matrices, figure notes |
| [`docs/deployment.md`](docs/deployment.md) | ESP-IDF, ESP-DL, quantisation, serial and Wi-Fi use |
| [`docs/references.md`](docs/references.md) | Dataset, MobileNetV2, KD, ESP-DL references |
| [`training/README.md`](training/README.md) | Notebook inventory and reproducible scripts |
| [`models/README.md`](models/README.md) | Per-artefact model card |
| [`data/README.md`](data/README.md) | Expected dataset layout and split semantics |
| [`host/README.md`](host/README.md) | CLI and GUI usage |
| [`firmware/README.md`](firmware/README.md) | Variant matrix, wiring, build, protocols |
| [`firmware/jtag/README.md`](firmware/jtag/README.md) | USB-Serial-JTAG variant, protocol and archive reconciliation |

## Citation

If you use this work, please cite the repository using
[`CITATION.cff`](CITATION.cff):

```text
Harmouni, Y. (2026). AgroStake: Edge AI for Grapevine Disease
Detection on the ESP32-S3 (Version 0.1.1) [Computer software].
https://github.com/XareDev/agrostake
```

## License

- **Code** (`training/src/`, `host/`, `firmware/`, `tests/`): MIT — see
  [`LICENSE`](LICENSE).
- **Documentation, figures, notebooks, and released model weights/metadata**:
  CC BY-NC-SA 4.0 — see [`LICENSE-DOCS`](LICENSE-DOCS). The weights are shared
  for research and non-commercial use; contact the authors for other terms.
- **Dataset**: not included. PlantVillage and any derived augmented dataset
  remain subject to their upstream terms; if you redistribute a derived
  dataset, keep the CC BY-NC-SA 4.0 notice and the upstream attribution.

## Acknowledgements

This work was carried out at IGEE (ex. INELEC), University of Boumerdes,
Algeria. It builds on the PlantVillage grape-leaf images, MobileNetV2,
knowledge distillation, TensorFlow Lite, and Espressif's ESP-DL runtime. See
[`docs/references.md`](docs/references.md) for the full list.
