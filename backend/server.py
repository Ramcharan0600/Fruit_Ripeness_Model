import glob
import io
import os
from typing import List

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.resnet import preprocess_input

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_CACHE: dict[str, tf.keras.Model] = {}
MODEL_BENCHMARKS = {
    "resnet152_categorical_final.keras": {
        "type": "categorical",
        "accuracy": 0.6973,
        "mae": 0.3333,
        "label": "ResNet152 Categorical / Softmax",
    },
    "resnet152_ordinal_final.keras": {
        "type": "ordinal",
        "accuracy": 0.7687,
        "mae": 0.2449,
        "label": "ResNet152 Ordinal / Thresholds",
    },
    "resnet152_transfer_learning_final.keras": {
        "type": "categorical",
        "accuracy": 0.9271,
        "mae": None,
        "label": "ResNet152 Transfer Learning",
    },
}

APP_TITLE = "Banana Ripeness AI"
CLASS_NAMES = ["Unripe", "Half-ripe", "Ripe", "Overripe"]
STAGE_INFO = {
    "Unripe": {
        "emoji": "🟢",
        "description": "Green and firm. High in starch, low in sugar. Great for cooking or green smoothies.",
        "class_css": "stage-unripe",
    },
    "Half-ripe": {
        "emoji": "🟡",
        "description": "Yellow with green tips. Mild sweetness with a firmer texture.",
        "class_css": "stage-half-ripe",
    },
    "Ripe": {
        "emoji": "🍌",
        "description": "Fully yellow with slight sugar speckles. Peak sweetness and flavor for direct eating.",
        "class_css": "stage-ripe",
    },
    "Overripe": {
        "emoji": "🟤",
        "description": "Heavy brown spots or fully brown. Soft and very sweet. Ideal for banana bread, pancakes, or freezing.",
        "class_css": "stage-overripe",
    },
}
ORDINAL_THRESHOLDS = ["P(> Unripe)", "P(> Half-ripe)", "P(> Ripe)"]

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})


def list_models() -> List[dict]:
    paths = (
        glob.glob(os.path.join(MODELS_DIR, "*.keras"))
        + glob.glob(os.path.join(MODELS_DIR, "*.h5"))
        + glob.glob(os.path.join(BASE_DIR, "*.keras"))
        + glob.glob(os.path.join(BASE_DIR, "*.h5"))
    )
    unique_paths = sorted(set(paths))
    output = []
    for path in unique_paths:
        name = os.path.basename(path)
        benchmark = MODEL_BENCHMARKS.get(name, {})
        output.append({
            "name": name,
            "label": benchmark.get("label", name),
            "type": benchmark.get("type", "categorical"),
            "accuracy": benchmark.get("accuracy"),
            "mae": benchmark.get("mae"),
        })
    return output


def get_model_abs_path(model_name: str) -> str:
    candidates = [
        os.path.join(MODELS_DIR, model_name),
        os.path.join(BASE_DIR, model_name),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"Model file not found: {model_name}")


def load_model(model_name: str):
    if model_name not in MODEL_CACHE:
        model_path = get_model_abs_path(model_name)
        MODEL_CACHE[model_name] = tf.keras.models.load_model(model_path, compile=False)
    return MODEL_CACHE[model_name]


def preprocess_image(image_bytes: bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((224, 224))
    img_array = np.array(image, dtype=np.float32)
    preprocessed = preprocess_input(img_array)
    return np.expand_dims(preprocessed, axis=0)


def predict_stage(raw_output: np.ndarray):
    if raw_output.shape[0] == 3:
        pred_idx = int((raw_output >= 0.5).sum())
        if pred_idx >= len(CLASS_NAMES):
            pred_idx = len(CLASS_NAMES) - 1
        pred_stage = CLASS_NAMES[pred_idx]
        return {
            "type": "ordinal",
            "predicted_class": pred_stage,
            "confidence": float(np.max(raw_output)) if raw_output.size else 0.0,
            "probabilities": [float(value) for value in raw_output],
            "labels": ORDINAL_THRESHOLDS,
            "description": STAGE_INFO[pred_stage]["description"],
            "class_css": STAGE_INFO[pred_stage]["class_css"],
            "emoji": STAGE_INFO[pred_stage]["emoji"],
        }

    pred_idx = int(np.argmax(raw_output))
    pred_stage = CLASS_NAMES[pred_idx]
    confidence = float(raw_output[pred_idx])
    return {
        "type": "categorical",
        "predicted_class": pred_stage,
        "confidence": confidence,
        "probabilities": [float(value) for value in raw_output],
        "labels": CLASS_NAMES,
        "description": STAGE_INFO[pred_stage]["description"],
        "class_css": STAGE_INFO[pred_stage]["class_css"],
        "emoji": STAGE_INFO[pred_stage]["emoji"],
    }


@app.get("/api/health")
def health():
    return {"status": "ok", "title": APP_TITLE, "models": list_models()}


@app.get("/api/models")
def get_models():
    return {"models": list_models()}


@app.post("/api/predict")
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No image uploaded."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No image selected."}), 400

    model_name = request.form.get("model")
    if not model_name:
        return jsonify({"error": "Model name is required."}), 400

    try:
        model = load_model(model_name)
        image_bytes = file.read()
        batch = preprocess_image(image_bytes)
        raw_output = model.predict(batch, verbose=0)[0]
        result = predict_stage(np.asarray(raw_output, dtype=np.float32))
        benchmark = MODEL_BENCHMARKS.get(model_name, {})
        result["benchmark"] = {
            "accuracy": benchmark.get("accuracy"),
            "mae": benchmark.get("mae"),
            "type": benchmark.get("type"),
            "label": benchmark.get("label"),
        }
        return {
            "model": model_name,
            "architecture": "ordinal" if result["type"] == "ordinal" else "categorical",
            "result": result,
        }
    except Exception as exc:  # pragma: no cover
        return jsonify({"error": str(exc)}), 500


@app.post("/api/compare")
def compare_models():
    if "file" not in request.files:
        return jsonify({"error": "No image uploaded."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No image selected."}), 400

    ground_truth = request.form.get("groundTruth", "").strip()
    try:
        image_bytes = file.read()
        batch = preprocess_image(image_bytes)
        comparison = []
        for model_name in [m["name"] for m in list_models()]:
            model = load_model(model_name)
            raw_output = model.predict(batch, verbose=0)[0]
            result = predict_stage(np.asarray(raw_output, dtype=np.float32))
            benchmark = MODEL_BENCHMARKS.get(model_name, {})
            result["benchmark"] = {
                "accuracy": benchmark.get("accuracy"),
                "mae": benchmark.get("mae"),
                "type": benchmark.get("type"),
                "label": benchmark.get("label"),
            }
            comparison.append({
                "model": model_name,
                "label": benchmark.get("label", model_name),
                "type": result["type"],
                "predicted_class": result["predicted_class"],
                "confidence": result["confidence"],
                "probabilities": result["probabilities"],
                "labels": result["labels"],
                "benchmark": result["benchmark"],
            })

        sample_accuracy = None
        if ground_truth:
            ground_truth_index = CLASS_NAMES.index(ground_truth)
            sample_accuracy = 1 if max(comparison, key=lambda x: x["confidence"])["predicted_class"] == ground_truth else 0
            if ground_truth_index is not None:
                for item in comparison:
                    item["ground_truth"] = ground_truth
                    item["sample_correct"] = (item["predicted_class"] == ground_truth)

        return {
            "comparison": comparison,
            "ground_truth": ground_truth,
            "sample_accuracy": sample_accuracy,
            "benchmark_table": [
                {
                    "model": item["model"],
                    "label": item["label"],
                    "type": item["type"],
                    "accuracy": item["benchmark"]["accuracy"],
                    "mae": item["benchmark"]["mae"],
                }
                for item in comparison
            ],
        }
    except Exception as exc:  # pragma: no cover
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
