# Fruit Ripeness Grading — Documentation

Banana ripeness classification with **ResNet152**, comparing a standard multiclass classifier against an **ordinal regression** model.

- **Source:** `Fruit_Ripeness_Ordinal_ResNet152_TransferLearning_Evaluation_Updated (1).ipynb` (Google Colab notebook, TensorFlow/Keras)
- **Dataset:** BananaID (Mendeley Data), original ripeness images only
- **Task:** predict one of four ordered stages: `Unripe < Half-ripe < Ripe < Overripe`

---

## 1. Project overview

Ripeness is an *ordered* quantity, so predicting "Overripe" for an Unripe banana is a worse mistake than predicting "Half-ripe". A softmax classifier treats all errors equally. This project tests whether a model that respects the ordering does better.

Both models share the same backbone, preprocessing, data split and training budget, so the only difference is the output head and loss.

### Headline results (test set, 294 images)

| Model | Accuracy | MAE (stages) |
|---|---|---|
| ResNet152 Categorical | 0.6973 | 0.3333 |
| ResNet152 Ordinal | **0.7687** | **0.2449** |

Ordinal wins on both metrics, with fewer 1-stage and 2-stage errors (see §5).

---

## 2. Dataset

| Class | Label | Images | Train | Val | Test |
|---|---|---|---|---|---|
| unripe | 0 | 408 | 286 | 61 | 61 |
| half-ripe | 1 | 488 | 341 | 73 | 74 |
| ripe | 2 | 616 | 431 | 93 | 92 |
| overripe | 3 | 448 | 314 | 67 | 67 |
| **Total** | | **1,960** | **1,372** | **294** | **294** |

- Source folder layout in the zip: `Banana Ripeness Classification Dataset/{Unripe, Half_Ripe, Ripe, Overripe}`.
- The loader walks the directory recursively and maps the parent folder name to a label. Accepted aliases: `unripe`, `green`, `half_ripe`, `half-ripe`, `half ripe`, `ripe`, `overripe`, `over-ripe`, `over_ripe`, `over ripe` (case-insensitive; `-` and spaces are normalised to `_`).
- Accepted extensions: `.jpg .jpeg .png .bmp .webp`.
- **Use the original images, not the augmented copies.** Augmentation is done in the training pipeline so the validation/test sets stay clean.
- Split: stratified 70/15/15, `random_state=42`, done *before* augmentation.

---

## 3. Pipeline

1. **Load** – `tf.io.read_file` → `decode_image` (3 channels) → resize to 224×224.
2. **Augment (train only)** – horizontal flip, rotation ±0.08, zoom 0.10, contrast 0.10.
3. **Preprocess** – `keras.applications.resnet.preprocess_input`.
4. **Batch** – batch size 16, `prefetch(AUTOTUNE)`. Training set is reshuffled every epoch.

Target encoding differs per model:

| Class | Categorical target | Ordinal target |
|---|---|---|
| Unripe | `0` | `[0, 0, 0]` |
| Half-ripe | `1` | `[1, 0, 0]` |
| Ripe | `2` | `[1, 1, 0]` |
| Overripe | `3` | `[1, 1, 1]` |

The ordinal target is `[y > 0, y > 1, y > 2]`.

---

## 4. Models

Both models use the same body:

```
Input (224×224×3)
→ ResNet152 (include_top=False, weights=None)   # 7×7×2048, 58.37M params
→ GlobalAveragePooling2D                         # 2048
→ Dense(256, ReLU)                               # 524,544 params
→ Dropout(0.5)
→ <head>
```

> **The backbone is trained from scratch (`weights=None`), not ImageNet-pretrained.** This is a deliberate choice in the notebook, and likely the main reason for the modest accuracy and slow early epochs.

### 4.1 Categorical model — `ResNet152_Categorical`

| | |
|---|---|
| Head | `Dense(4, softmax)` |
| Loss | `SparseCategoricalCrossentropy` |
| Class weights | balanced, from training set: unripe 1.199, half-ripe 1.006, ripe 0.796, overripe 1.092 |
| Params | 58,896,516 total (58,745,092 trainable) |
| Checkpoint | `val_accuracy` (max) → `saved_models/resnet152_categorical_best.keras` |
| ReduceLROnPlateau | monitor `val_loss`, factor 0.2, patience 3, min 1e-7 |
| EarlyStopping | monitor **`accuracy`** (training), patience 6, restore best weights |
| Prediction | `argmax` of the softmax |

### 4.2 Ordinal model — `ResNet152_Ordinal`

| | |
|---|---|
| Head | `Dense(3, sigmoid)`: outputs `P(y>Unripe)`, `P(y>Half-ripe)`, `P(y>Ripe)` |
| Loss | `ordinal_bce_loss`: binary cross-entropy averaged over the 3 thresholds (predictions clipped to [1e-7, 1−1e-7]) |
| Class weights | none |
| Params | 58,896,259 total (58,744,835 trainable) |
| Checkpoint | `val_loss` (min) → `saved_models/resnet152_ordinal_best.keras` |
| ReduceLROnPlateau | monitor `val_loss`, factor 0.2, patience 3, min 1e-7 |
| EarlyStopping | monitor `val_loss`, patience 7, restore best weights |
| Prediction | number of thresholds with probability ≥ 0.5: `int((probs >= 0.5).sum())` |

### 4.3 Shared training settings

| Setting | Value |
|---|---|
| Optimizer | Adam, lr 1e-4 |
| Max epochs | 40 |
| Batch size | 16 |
| Image size | 224×224 |
| Seed | 42 (Python, NumPy, TensorFlow) |
| Environment | TensorFlow 2.21.0 on Colab |

### 4.4 Saved artifacts (written to `saved_models/` on Colab)

- `resnet152_categorical_best.keras`, `resnet152_ordinal_best.keras` – best checkpoints
- `resnet152_categorical_final.keras`, `resnet152_ordinal_final.keras` – final models
- `model_comparison.csv` – accuracy / MAE table

**None of these files are in this project folder yet.** Download them from the Colab session if you want to reuse the models.

---

## 5. Results

### Per-class metrics (test set)

| Class | Categorical P / R / F1 | Ordinal P / R / F1 |
|---|---|---|
| unripe | 0.750 / 0.885 / 0.812 | 0.815 / 0.869 / 0.841 |
| half-ripe | 0.619 / 0.527 / 0.569 | 0.723 / 0.635 / 0.676 |
| ripe | 0.684 / 0.707 / 0.695 | 0.735 / 0.783 / 0.758 |
| overripe | 0.734 / 0.702 / 0.718 | 0.818 / 0.806 / 0.812 |
| macro avg F1 | 0.699 | 0.772 |

Half-ripe is the hardest class for both models, as expected for the middle of a continuum.

### Error distance (absolute stages off, count of test images)

| Distance | Categorical | Ordinal |
|---|---|---|
| 0 (correct) | 205 | 226 |
| 1 | 80 | 64 |
| 2 | 9 | 4 |
| 3 | 0 | 0 |

### Caveats

- Single run, single seed, 294 test images. A 7-point accuracy gap is suggestive, not statistically established. Repeat with several seeds or use cross-validation before drawing firm conclusions.
- Categorical training was slow to take off (validation accuracy stuck at 0.248 for the first 3 epochs), consistent with training from scratch.

---

## 6. How to run

### On Colab (as written)

1. Upload `Banana Ripeness Dataset.zip` to `/content/`.
2. Run the notebook top to bottom. It unzips to `/content/dataset`, trains both models, evaluates, and saves to `/content/saved_models`.
3. The final cell uses `google.colab.files.upload()` to run both models on an image you upload.

Training time is roughly 1–2 minutes per epoch on a Colab GPU (the first epoch takes about 5 minutes).

### Locally

The notebook hard-codes Colab paths. To run it locally:

- Replace `/content/Banana Ripeness Dataset.zip` and `/content/dataset` with local paths.
- Remove the `from google.colab import files` import and the upload cell.
- Requirements: `tensorflow`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `pillow`. A GPU is strongly recommended for ResNet152.

### Inference with saved models

```python
import numpy as np
from PIL import Image
from tensorflow import keras
from tensorflow.keras.applications.resnet import preprocess_input

CLASS_NAMES = ["unripe", "half-ripe", "ripe", "overripe"]

# The ordinal model uses a custom loss, so load without compiling.
ordinal = keras.models.load_model("saved_models/resnet152_ordinal_final.keras", compile=False)

img = Image.open("banana.jpg").convert("RGB").resize((224, 224))
x = preprocess_input(np.asarray(img, dtype=np.float32))[None]

probs = ordinal.predict(x, verbose=0)[0]       # 3 threshold probabilities
stage = int((probs >= 0.5).sum())
print(CLASS_NAMES[stage], probs)
```

For the categorical model, use `CLASS_NAMES[np.argmax(model.predict(x)[0])]`.

---

## 7. Known issues and suggestions

These are things I noticed while reading the notebook. None were changed.

1. **Epoch count mismatch.** Code uses `EPOCHS = 40`; the final checklist says 50.
2. **Categorical EarlyStopping watches training `accuracy`**, not validation. It will rarely trigger on overfitting, whereas the ordinal model correctly monitors `val_loss`. This makes the comparison slightly uneven. Consider using `val_loss` for both.
3. **Backbone called with `training=True`** in both builders, so BatchNorm stays in training mode, including at inference through the Keras model. For from-scratch training this is usually harmless, but it is unusual.
4. **Disagreement on the sample image.** For the uploaded test image, the categorical model says Overripe (99.6%) while the ordinal model says Ripe (`P(>Ripe)` = 0.001). Both thresholds below `Ripe` are ~1.0, so the ordinal output is confident the other way. Worth inspecting that image.
5. **No pretrained weights.** Trying `weights="imagenet"` (with a frozen-then-unfrozen schedule) would likely raise accuracy for both models substantially, and gives a fairer picture of ordinal vs categorical.
6. **Ordinal output can be non-monotonic** (e.g. `P(>Half-ripe) < P(>Ripe)`). Counting thresholds ≥ 0.5 tolerates this, but CORAL/CORN-style heads enforce consistency.
7. **Dataset path inconsistency.** Setup cell defines `./dataset`; later cells override with `/content/dataset`. The reproducibility notes also reference `./dataset`.
8. **Stray text** `citeturn0search1` at the end of the last markdown cell is a leftover citation artifact and can be deleted.
9. **Dataset name:** the unzipped folder is named `Half_Ripe`; the markdown cell says `Half-ripe`. Both are handled by the alias map.

---

## 8. Citation

Dataset: *BananaID: An image dataset of banana varieties and ripeness stages in Indonesia designed for classification*, Mendeley Data.
