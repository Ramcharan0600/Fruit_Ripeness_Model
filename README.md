# 🍌 Banana Ripeness Grading — ResNet152, Ordinal Regression and a Web App

Classifies banana ripeness into four ordered stages (**Unripe → Half-ripe → Ripe → Overripe**) from a photo. The project trains and compares three ResNet152 models and serves them through a React web app backed by a Flask API.

**What's in the project**

- A **research part**: a notebook that trains a standard softmax classifier and an ordinal regression model that respects the ordering of the stages, plus a transfer-learning model.
- A **web app**: upload a banana photo, pick a model (or compare all of them) and get the predicted stage, confidence, probabilities and each model's benchmark scores.

## Results

Test set, 294 images:

| Model | Accuracy | MAE (stages) |
|---|---|---|
| ResNet152 Categorical (from scratch) | 0.6973 | 0.3333 |
| ResNet152 Ordinal (from scratch) | 0.7687 | 0.2449 |
| ResNet152 Transfer Learning | **0.9271** | n/a |

- Between the two from-scratch models, the ordinal one is more accurate and makes fewer large errors, since it knows that Unripe → Overripe is a worse mistake than Unripe → Half-ripe. This is a single run with one seed, so treat that gap as suggestive.
- The transfer-learning model scores highest. Its MAE was not recorded, and its accuracy is the figure stored in the app's benchmark table (`backend/server.py`).

## How it works

```
Browser (React, hosted on Vercel)
        │  HTTPS
        ▼
   ngrok tunnel  ───►  Flask API (your PC, port 5000)  ───►  .keras models in models/
```

1. The page loads the model list from `GET /api/models`.
2. You upload a banana photo and choose a model, or compare all of them.
3. The API resizes the image to 224×224, applies ResNet's `preprocess_input`, and runs the model.
4. It returns the predicted stage, confidence, per-class (or per-threshold) probabilities, a short description of the stage and the model's benchmark accuracy and MAE.

Models are loaded on first use and cached in memory, so the first prediction is slower than later ones.

### API

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/health` | GET | Status check and list of available models |
| `/api/models` | GET | Available model files |
| `/api/predict` | POST | `file` + `model`: prediction from one model |
| `/api/compare` | POST | `file` (+ optional `groundTruth`): prediction from every model |

## How I built and ran this project

1. **Trained the models on Google Colab** (GPU). The notebook loads the BananaID dataset, builds the categorical, ordinal and transfer-learning ResNet152 models, trains and evaluates them, and saves the final `.keras` files.
2. **Wrote a Flask API** (`backend/server.py`) that loads those model files and exposes the prediction endpoints above.
3. **Built a React (Vite) dashboard** (`frontend/`) for uploading an image, choosing a model and comparing results.
4. **Deployed the frontend on Vercel**, straight from this GitHub repository.
5. **Exposed the backend with a free ngrok tunnel.** The three models total about 1.7 GB, which is more memory than free hosting tiers provide (I tried hosted backends first). So the API runs on my own PC, and ngrok gives it a public HTTPS URL. That URL is set as `VITE_API_URL` in Vercel.
6. **Added the `ngrok-skip-browser-warning` header** to the frontend's requests. Without it, free-tier ngrok returns an HTML warning page instead of JSON and the app reports that it cannot connect.

So the live site works only while my PC, the Flask server and ngrok are running. See [Run it yourself](#run-it-yourself) to run your own copy.

## Run it yourself

### Prerequisites

- Python 3.10+ and Node.js 18+
- The trained model files (`.keras` or `.h5`). They are too large for GitHub, so they are not in this repository. Train them with the notebook, or use your own, and put them in a `models/` folder at the repository root.
- A free [ngrok](https://ngrok.com) account (only needed if you want to put the site online)

### 1. Backend

```
pip install -r requirements.txt
python backend/server.py
```

The API listens on port 5000. Check `http://localhost:5000/api/health`.

### 2. Frontend (local)

```
cd frontend
npm install
npm run dev
```

Open the URL Vite prints. With no `VITE_API_URL` set, the app talks to `http://localhost:5000`.

### 3. Put it online (optional)

1. Update ngrok to version 3.30.0 or newer (`ngrok update`) and add your token (`ngrok config add-authtoken <token>`).
2. Start the tunnel: `ngrok http 5000`.
3. Open `<your-ngrok-url>/api/models` in a browser. You should see JSON listing your models. If a warning page appears, click "Visit Site".
4. On Vercel, import the repository and set the project root to `frontend`. Build command: `npm run build`, output directory: `dist`.
5. Set the environment variable `VITE_API_URL` to your ngrok URL, with no trailing slash and no `/api`, then redeploy. Vite bakes the value in at build time.

Tips:

- A free static domain from the ngrok dashboard (Domains) keeps the URL the same between restarts: `ngrok http --domain=your-name.ngrok-free.dev 5000`.
- The API allows all origins (CORS `*`). Restrict this to your frontend's domain for anything beyond a demo.

### Troubleshooting

| Symptom | Fix |
|---|---|
| "Could not connect to the prediction server" | Check that `python backend/server.py` and `ngrok` are both running, and that `VITE_API_URL` matches the current ngrok URL (redeploy after changing it). |
| ngrok error `ERR_NGROK_121` (agent too old) | Run `ngrok update`. |
| Model list is empty | Put `.keras` / `.h5` files in `models/`. |
| Works locally but not on Vercel | `VITE_API_URL` must be an `https://` URL with no trailing slash and no `/api`. |
| Warning page when opening the ngrok URL in a browser | Expected. Click "Visit Site". The app skips it with a request header. |

## Project structure

| Path | Purpose |
|---|---|
| `Fruit_Ripeness_Ordinal_ResNet152_TransferLearning_Evaluation_Updated (1).ipynb` | Full experiment: data loading, models, training, evaluation, single-image inference |
| `backend/server.py` | Flask API that loads the models and serves predictions |
| `frontend/` | React (Vite) dashboard |
| `requirements.txt` | Python dependencies for the API |
| `DOCUMENTATION.md` | Detailed docs: dataset, pipeline, model specs, results, known issues |
| `models/` | Trained model files (not in Git) |

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

## Training the models

### On Google Colab

1. Upload `Banana Ripeness Dataset.zip` to `/content/`.
2. Open the notebook and run all cells.
3. Trained models are saved to `/content/saved_models/`. Download them before the session ends and place them in `models/`.

The last section lets you upload a banana photo and see the models' predictions.

### Locally

The notebook uses Colab paths, so edit the zip and dataset paths and remove the `google.colab` upload cell. Needs `tensorflow`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn` and `pillow`. A GPU is strongly recommended.

## Model details

The from-scratch models share a ResNet152 backbone (no pretrained weights), global average pooling, `Dense(256)` and `Dropout(0.5)`. They differ only in the head:

- **Categorical:** `Dense(4, softmax)` with cross-entropy loss and class weights.
- **Ordinal:** `Dense(3, sigmoid)` predicting `P(y>Unripe)`, `P(y>Half-ripe)` and `P(y>Ripe)`, trained with binary cross-entropy. The predicted class is the number of thresholds with probability ≥ 0.5.

Training settings: 224×224 input, batch size 16, Adam at 1e-4, up to 40 epochs, stratified 70/15/15 split, seed 42.

See [DOCUMENTATION.md](DOCUMENTATION.md) for full details.

## Citation

*BananaID: An image dataset of banana varieties and ripeness stages in Indonesia designed for classification*, Mendeley Data.
