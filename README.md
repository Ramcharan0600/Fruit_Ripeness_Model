# Fruit Ripeness Grading — Ordinal Regression with ResNet152

Classifies banana ripeness into four ordered stages (**Unripe → Half-ripe → Ripe → Overripe**) and compares a standard softmax classifier against an ordinal regression model that respects the ordering.

## Results

Test set, 294 images:

| Model | Accuracy | MAE (stages) |
|---|---|---|
| ResNet152 Categorical | 0.6973 | 0.3333 |
| ResNet152 Ordinal | **0.7687** | **0.2449** |

The ordinal model is more accurate and makes fewer large errors. This is a single run with one seed, so treat the gap as suggestive.

## What's in here

| File | Purpose |
|---|---|
| `Fruit_Ripeness_Ordinal_ResNet152_Banana_Classification (1).ipynb` | Full experiment: data loading, both models, training, evaluation, single-image inference |
| `DOCUMENTATION.md` | Detailed docs: dataset, pipeline, model specs, results, known issues |

## Dataset

[BananaID](https://data.mendeley.com/) (Mendeley Data): 1,960 original ripeness images across four classes. Use the **original** images, not the augmented copies. Expected layout:

```
dataset/
├── Unripe/
├── Half_Ripe/
├── Ripe/
└── Overripe/
```

Folder names are matched case-insensitively, and `half-ripe`, `half ripe` and `over_ripe` style variants also work.

## Quick start (Google Colab)

1. Upload `Banana Ripeness Dataset.zip` to `/content/`.
2. Open the notebook and run all cells.
3. Trained models are saved to `/content/saved_models/`. Download them before the session ends.

The last section lets you upload a banana photo and see both models' predictions.

## Running locally

The notebook uses Colab paths, so edit the zip and dataset paths and remove the `google.colab` upload cell. Requirements:

```
tensorflow
numpy
pandas
matplotlib
seaborn
scikit-learn
pillow
```

A GPU is strongly recommended.

## How it works

Both models share a ResNet152 backbone (trained from scratch, no pretrained weights), global average pooling, `Dense(256)` and `Dropout(0.5)`. They differ only in the head:

- **Categorical:** `Dense(4, softmax)` with cross-entropy loss and class weights.
- **Ordinal:** `Dense(3, sigmoid)` predicting `P(y>Unripe)`, `P(y>Half-ripe)` and `P(y>Ripe)`, trained with binary cross-entropy. The predicted class is the number of thresholds with probability ≥ 0.5.

Training settings: 224×224 input, batch size 16, Adam at 1e-4, up to 40 epochs, stratified 70/15/15 split, seed 42.

See [DOCUMENTATION.md](DOCUMENTATION.md) for full details.

## Citation

*BananaID: An image dataset of banana varieties and ripeness stages in Indonesia designed for classification*, Mendeley Data.
