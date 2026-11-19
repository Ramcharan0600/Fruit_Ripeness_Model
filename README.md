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
| `Fruit_Ripeness_Ordinal_ResNet152_TransferLearning_Evaluation_Updated (1).ipynb` | Full experiment: data loading, both models, training, evaluation, single-image inference |
| `DOCUMENTATION.md` | Detailed docs: dataset, pipeline, model specs, results, known issues |
| `backend/server.py` | Flask API that loads the models and serves predictions |
| `frontend/` | React (Vite) dashboard |

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

## Web app (Vercel + ngrok)

The React dashboard in `frontend/` talks to the Flask API in `backend/server.py`. The models are about 1.7 GB, too large for free hosting tiers, so the API runs on your own PC and is exposed with a free ngrok tunnel. The frontend is hosted on Vercel.

### 1. Put the models in place

Place the trained `.keras` / `.h5` files in `models/` (excluded from Git by `.gitignore`).

### 2. Run the backend

```
pip install -r requirements.txt
python backend/server.py
```

The API listens on port 5000. Check `http://localhost:5000/api/health`.

### 3. Expose it with ngrok

Free ngrok accounts need agent version 3.30.0 or newer (`ngrok update`), and an authtoken (`ngrok config add-authtoken <token>`).

```
ngrok http 5000
```

Copy the `https://...ngrok-free.dev` URL. Open `<url>/api/models` in a browser once; you should see JSON listing your models. The frontend sends the `ngrok-skip-browser-warning` header, which stops free-tier ngrok from returning its HTML warning page instead of JSON.

### 4. Deploy the frontend

1. Push the repository to GitHub.
2. On Vercel, import the repository and set the project root to `frontend`. Build command: `npm run build`, output directory: `dist`.
3. Set the environment variable `VITE_API_URL` to your ngrok URL, with no trailing slash and no `/api`.
4. Redeploy. Vite bakes the value in at build time, so changing it requires a new build.

For local development, `npm run dev` inside `frontend/` uses `http://localhost:5000` when `VITE_API_URL` is not set.

Notes:

- Your PC, the Flask server and ngrok must all be running for the site to work.
- If your ngrok URL changes, update `VITE_API_URL` and redeploy. A free static domain from the ngrok dashboard avoids this.
- The API allows all origins (CORS `*`). Restrict this to your frontend's domain for anything beyond a demo.

## How it works

Both models share a ResNet152 backbone (trained from scratch, no pretrained weights), global average pooling, `Dense(256)` and `Dropout(0.5)`. They differ only in the head:

- **Categorical:** `Dense(4, softmax)` with cross-entropy loss and class weights.
- **Ordinal:** `Dense(3, sigmoid)` predicting `P(y>Unripe)`, `P(y>Half-ripe)` and `P(y>Ripe)`, trained with binary cross-entropy. The predicted class is the number of thresholds with probability ≥ 0.5.

Training settings: 224×224 input, batch size 16, Adam at 1e-4, up to 40 epochs, stratified 70/15/15 split, seed 42.

See [DOCUMENTATION.md](DOCUMENTATION.md) for full details.

## Citation

*BananaID: An image dataset of banana varieties and ripeness stages in Indonesia designed for classification*, Mendeley Data.
