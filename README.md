# Fruit Ripeness Grading — Ordinal Regression with ResNet152

Classifies banana ripeness into four ordered stages (**Unripe → Half-ripe → Ripe → Overripe**) and compares a standard softmax classifier against an ordinal regression model that respects the ordering.

## Results

The dashboard's configured benchmark values are:

| Model | Accuracy | MAE (stages) |
|---|---|---|
| ResNet152 Categorical | 0.6973 | 0.3333 |
| ResNet152 Ordinal | **0.7687** | **0.2449** |
| ResNet152 Transfer Learning | 0.9271 | N/A |

The ordinal model outperforms the categorical model in this benchmark. The saved transfer-learning notebook reports a held-out test accuracy of 0.8707, which differs from the dashboard's configured value of 0.9271; do not treat these values as a directly comparable test-set ranking until that discrepancy is reconciled.

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

## Hosting the web app

The repository includes two web interfaces:

- The React dashboard in `frontend/` uses the Flask API in `backend/server.py`.
- `app.py` is a separate Streamlit app and does not use the React dashboard.

### React dashboard (Vercel + Render)

Deploy the Flask API as a Render Web Service and the React frontend as a Vercel project. The root `.python-version` file pins Render to Python 3.12, which has a TensorFlow 2.21.0 Linux wheel. Render's newer default Python 3.14 does not.

1. Push the project to GitHub.
2. On Render, create a Web Service for the repository. Keep the service's root directory at the repository root so Render reads `.python-version`. Use:
   - Build command: `pip install -r requirements.txt && pip install gunicorn`
   - Start command: `gunicorn --bind 0.0.0.0:$PORT backend.server:app`
3. On Vercel, import the repository and set the project root to `frontend`. Use `npm run build` as the build command and `dist` as the output directory. In the Vercel project's environment variables, set `VITE_API_URL` to the public URL of your Render service (for example, `https://your-service.onrender.com`, without `/api`). Set it for the environments you use, then redeploy so the value is included in the frontend build.
4. Verify the backend at `https://your-service.onrender.com/api/health`. Opening the service root (`https://your-service.onrender.com/`) returns backend information; the API itself is under `/api/`. Then open the Vercel URL and try a prediction.

The backend needs trained `.keras` or `.h5` model files under the repository's `models/` directory (or project root). Model files are excluded from Git by `.gitignore`, so they will not be present in a normal deployment from GitHub. Provide them to the deployed service separately, for example by downloading them from model or object storage during deployment. Choose a hosting plan with sufficient memory for TensorFlow and the models. For production, configure the API's CORS policy to allow only your frontend's domain.

### Streamlit app

To host only the Streamlit interface, deploy `app.py` with Streamlit Community Cloud and use the repository's `requirements.txt`. It also requires trained model files in `models/` (or the project root), so make those available to the deployment as described above.

## How it works

Both models share a ResNet152 backbone (trained from scratch, no pretrained weights), global average pooling, `Dense(256)` and `Dropout(0.5)`. They differ only in the head:

- **Categorical:** `Dense(4, softmax)` with cross-entropy loss and class weights.
- **Ordinal:** `Dense(3, sigmoid)` predicting `P(y>Unripe)`, `P(y>Half-ripe)` and `P(y>Ripe)`, trained with binary cross-entropy. The predicted class is the number of thresholds with probability ≥ 0.5.

Training settings: 224×224 input, batch size 16, Adam at 1e-4, up to 40 epochs, stratified 70/15/15 split, seed 42.

See [DOCUMENTATION.md](DOCUMENTATION.md) for full details.

## Citation

*BananaID: An image dataset of banana varieties and ripeness stages in Indonesia designed for classification*, Mendeley Data.
