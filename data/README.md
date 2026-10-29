# Dataset and splits

The image dataset is **not included** in this repository. It is large, and its
upstream terms are separate from the code licence. What is included is the
exact train/validation index file used by the experiments:
[`splits/split_indices.npz`](splits/split_indices.npz).

## Expected layout

The training scripts use `torchvision.datasets.ImageFolder`, so each class is a
directory:

```text
data/augmented-dataset/
├── train/
│   ├── Black Rot/
│   ├── ESCA/
│   ├── Healthy/
│   └── Leaf Blight/
└── test/
    ├── Black Rot/
    ├── ESCA/
    ├── Healthy/
    └── Leaf Blight/
```

`train/` is the **14,444-image train+validation pool** (not the final training
split). `test/` is the 3,610-image held-out set. The split file indexes
`train/`:

| File | Key | Shape | Meaning |
|---|---|---|---|
| `splits/split_indices.npz` | `train_idx` | `(12999,)` | indices into the 14,444-image `train/` pool |
| `splits/split_indices.npz` | `val_idx` | `(1445,)` | held-out validation indices from the same pool |

The final test set is never used for model selection. The split is
stratified by class in the original notebooks; the index file preserves the
exact rows rather than relying on a fresh random seed.

If `data/augmented-dataset/val/` exists, the loader uses that directory instead
of `split_indices.npz`. If neither is available, it creates a stratified 90/10
split with `--seed` and records the choice in `dataset_metadata.json`.

## Class order

ImageFolder sorts directory names alphabetically. The label order is therefore:

| Label | ImageFolder name | Firmware name |
|---:|---|---|
| 0 | `Black Rot` | `Black_rot` |
| 1 | `ESCA` | `Esca` |
| 2 | `Healthy` | `Healthy` |
| 3 | `Leaf Blight` | `Leaf_blight` |

The firmware class arrays use the same order. `training/src/data.py` validates
this mapping and refuses unknown directory names. If you rename the folders,
keep the alphabetical order or update both the firmware array and the mapping.

## Test-set composition

The preserved evaluation reports use 3,610 test images:

| Class | Support |
|---|---:|
| Black Rot | 944 |
| ESCA | 960 |
| Healthy | 846 |
| Leaf Blight | 860 |
| **Total** | **3,610** |

The v2 confusion matrix and the grayscale baseline/KD reports in
[`../docs/results.md`](../docs/results.md) use these supports.

## Preprocessing contracts

There are two distinct contracts in the project. They are easy to confuse, so
they are stated explicitly here.

### RGB v2 / binary gate (`v2`, `binary`)

- Resize/crop to 96×96 RGB.
- Training: random resized crop (70–100 %), horizontal flip, ±15° rotation,
  mild colour jitter.
- Validation/test: resize + centre crop.
- Normalisation: `pixel / 127.5 - 1`, i.e. `[-1, 1]`, implemented as
  `Normalize(0.5, 0.5)`.
- Firmware/host: send `pixel - 128` as signed INT8.

This is the contract used by the v2 firmware and by
`host/serial/send_image.py --mode uart-v2|jtag`.

### Grayscale GAP (`gray`)

- Convert to single-channel grayscale, resize to 96×96.
- Training: random resized crop, horizontal flip, ±15° rotation, mild
  brightness/contrast jitter.
- Original TensorFlow notebook: `[0, 1]` input (no mean/std subtraction).
- Firmware/host: the TFLite INT8 model uses scale `1/255` and zero point
  `-128`, so the host sends `pixel - 128` and the model's quantisation metadata
  maps it back to `[0, 1]`.

Use `--normalization unit` to reproduce the notebook's `[0, 1]` training
contract, or `--normalization symmetric` if you intentionally train a
deployment-style `[-1, 1]` grayscale model.

### Legacy binary checkpoint

`models/TinyBinaryCNN_v1_best.pth` was trained in
`06_binary_gate.ipynb` with ImageNet mean/std normalisation. That is **not**
the `pixel - 128` contract used by the firmware. The checkpoint is kept as an
experiment; the current firmware does not load it. Retrain with
`--arch binary` for a deployment-consistent gate.

## Augmentation and leakage

The source dataset is an augmented grape-leaf dataset. The notebooks use
random crops, flips, rotations, and colour jitter. Because the test split comes
from the same augmented pool, the reported numbers should be read as
reproducibility metrics on that pool, not as field-generalisation estimates.
See the scope note in the top-level README.

## Obtaining the images

The classes correspond to the grape subset of PlantVillage:

- Hughes, D. P., & Salathé, M. (2015). *An open access repository of images on
  plant health for the development of mobile and web applications.*
  <https://doi.org/10.5281/zenodo.1204916>
- Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016). *Using deep learning
  for image-based plant disease detection.* Frontiers in Plant Science, 7,
  1419. <https://doi.org/10.3389/fpls.2016.01419>

Download the upstream images, apply your own augmentation, and arrange them in
the layout above. Do not commit the images to this repository.

## Licence

The split indices and documentation are released under CC BY-NC-SA 4.0 (see
[`../LICENSE-DOCS`](../LICENSE-DOCS)). The upstream images remain under their
own terms. If you redistribute a derived dataset, keep the upstream
attribution and the CC BY-NC-SA 4.0 notice.
