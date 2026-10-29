# Version and provenance map

This repository is a **consolidation**, not a replay of every Git branch from
the original private workspace. The original work lived in several branches and
working trees; the public repository maps each functional variant to a stable
directory and records where it came from.

## Repository version

| Item | Value |
|---|---|
| Public release | `0.1.0` |
| Release date | 2026-10-29 |
| Primary model | RGB v2, 275,236 parameters |
| Dataset split | 12,999 train / 1,445 validation / 3,610 test |

The version number is for the public packaging. The underlying experiments span
several iterations and were not individually tagged.

## Firmware variants

| Public directory | Functional variant | Model | Original source (as preserved) |
|---|---|---|---|
| `firmware/uart-grayscale` | v1 grayscale over UART0 | `TinyAgCNN_s8.espdl` (27,828 params) | Original grayscale inference firmware |
| `firmware/uart-rgb` | v2 RGB over UART0 | `TinyAgCNN_v2_s8.espdl` (275,236 params) | `feature/stable-opi-80mhz-uart` |
| `firmware/jtag` | v2 RGB over USB-Serial-JTAG | separately exported v2 `.espdl` | USB-Serial-JTAG variant |
| `firmware/wifi` | v2 RGB SoftAP + browser UI | `TinyAgCNN_v2_s8.espdl` | Wi-Fi workspace (`WI-FI/`) |
| `firmware/experimental/wifi-serial` | v2 serial + Wi-Fi, WIP | `TinyAgCNN_v2_s8.espdl` | `unresolved` / `folder_unresolved` worktree |

Not every original branch name can be recovered from the preserved workspace.
Where a name is known it is shown above; where it is not, the functional variant
is the source of truth. The public repository deliberately does **not** recreate
one Git branch per hardware experiment, because that would duplicate hundreds of
near-identical files and make the maintained path harder to find.

### Firmware file hashes

| File | SHA-256 | Size |
|---|---|---:|
| `models/TinyAgCNN_s8.espdl` (grayscale) | `5FDB51D9B4321515516858E501CBB4AE697289C60FE1F9CFDC3819DD9A55D609` | 46,576 B |
| `models/TinyAgCNN_v2_s8.espdl` (v2 shared) | `E8163BF47B8EBC6EEB3CA2980A696ED89E6F4D116A2E82C5D538930211238614` | 315,808 B |
| `firmware/jtag/models/TinyAgCNN_v2_s8.espdl` | `D677878164AC19185346A021E850CE8A1967C4CB0E71D48992A375DEAF3FDC9F` | 315,776 B |

The two v2 files have the same architecture and task but are not byte-identical.
The JTAG firmware must keep its own copy.

## Model families

| Name | Parameters | Input | Purpose | Status |
|---|---:|---|---|---|
| `TinyAgCNN` (GAP) | 27,828 | 96×96×1 | grayscale baseline | deployed v1 |
| `TinyAgCNN` (KD GAP) | 27,828 | 96×96×1 | distillation student | preserved; underperformed |
| `TinyAgCNN` (Flatten) | 843,364 | 96×96×1 | early notebook variant | **not deployed** |
| `TinyAgCNN_v2` | 275,236 | 96×96×3 | RGB classifier | deployed v2 |
| `TinyBinaryCNN` | 497,226 | 96×96×3 | healthy/diseased gate | experiment only |

### What changed between v1 and v2

- **Input:** grayscale 96×96×1 → RGB 96×96×3.
- **Capacity:** 27,828 → 275,236 trainable parameters.
- **Head:** `GAP → Dense(64) → Dense(4)` → `GAP → Linear(256) → Linear(4)`.
- **Transport:** UART-only → UART, USB-Serial-JTAG, and Wi-Fi.
- **Preprocessing:** grayscale `[0,1]` training contract → RGB `[-1,1]`
  deployment contract.
- **Result:** 80.47 % → 98.61 % accuracy on the provided test split.

The v2 training notebook was not preserved. The PyTorch class in
`training/src/models.py` was reconstructed from the exported ONNX graph and the
checkpoint state dictionary; it loads the released checkpoint exactly. This is
a provenance limitation, not a hidden “reproduced from scratch” claim.

## Grayscale notebook versions

The original workspace contains two different grayscale notebooks:

| Notebook | Head | Parameters | Relationship to released artefacts |
|---|---|---:|---|
| `05_tinyagcnn_kd_tensorflow.ipynb` | GAP + Dense(64) + Dense(4) | 27,828 | **matches** `.keras`, TFLite, ONNX, ESP-DL |
| `legacy/05_tinyagcnn_kd_flatten_legacy.ipynb` | Flatten + Dense(128) | 843,364 | historical; no released artefact |
| `04_tinyagcnn_grayscale_pytorch.ipynb` | Flatten skeleton | 843,364 | historical; no released artefact |

The released `.keras` files (`TinyAgCNN_best.keras`, `best_baseline.keras`,
`best_kd.keras`) all have the GAP structure. The Flatten notebook is kept for
honesty about the earlier design, not as a deployment recipe.

## Dataset split

`data/splits/split_indices.npz` contains:

| Key | Length | Meaning |
|---|---:|---|
| `train_idx` | 12,999 | rows in the 14,444-image train pool |
| `val_idx` | 1,445 | held-out validation rows in the same pool |

The separate `test/` directory has 3,610 images. The split file indexes only the
train pool; it does not contain test labels. The same split is used by the
grayscale notebook and the v2 evaluation.

## Artifact timeline (as reconstructed)

```text
MobileNetV2 224×224 experiments
        │
        ├─ teacher checkpoint (not released)
        │
        ▼
TinyAgCNN grayscale, Flatten (843k)      ← legacy notebooks
        │
        ├─ architecture change to GAP (27.8k)
        │
        ├─ baseline CE (80.47 %)
        └─ KD student (60.00 %)          ← negative result
        │
        ▼
TinyAgCNN_v2 RGB (275k)                  ← v2 checkpoints / ONNX / ESP-DL
        │
        ├─ UART RGB firmware
        ├─ USB-Serial-JTAG firmware
        ├─ Wi-Fi firmware
        └─ experimental serial + Wi-Fi
        │
        ▼
binary gate experiment (497k)            ← not completed, not deployed
```

The ordering above is a reconstruction from filenames, notebook outputs, and
checkpoints. It is not a commit-by-commit history.

## Release integrity

`models/SHA256SUMS` covers every released model binary and JSON config. The
firmware copies are covered by the hash table above. If a hash differs, do not
assume it is a harmless re-export: check the size and the architecture first.
