"""
Task 4: Developing a Flask API for Deep Learning Models
==========================================================
Exposes the CNN digit-classification model trained in Task 1 (a from-scratch
NumPy CNN, saved as cnn_digits_model.pkl) as a REST API using Flask.

Endpoints:
  GET  /              - API information
  GET  /health        - health check
  POST /predict       - predict the digit for a single 8x8 grayscale image
  POST /predict/batch - predict digits for multiple images in one request

Run with:  python app.py
Then send requests to http://127.0.0.1:5000/
"""

import os
import pickle
import traceback
import numpy as np
from flask import Flask, request, jsonify

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # __file__ is not defined when this code runs inside a notebook cell
    # (Jupyter/Colab) rather than as a standalone script - fall back to the
    # current working directory in that case.
    BASE_DIR = os.getcwd()
MODEL_PATH = os.path.join(BASE_DIR, "cnn_digits_model.pkl")

app = Flask(__name__)

# ----------------------------------------------------------------------
# Model definition (must match the architecture used to train and save
# cnn_digits_model.pkl in Task 1) and inference-only forward pass.
# ----------------------------------------------------------------------


def im2col(x, kh, kw, stride=1, pad=0):
    N, H, W, C = x.shape
    if pad > 0:
        x = np.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)))
    out_h = (H + 2 * pad - kh) // stride + 1
    out_w = (W + 2 * pad - kw) // stride + 1
    cols = np.zeros((N, out_h, out_w, kh, kw, C), dtype=x.dtype)
    for i in range(kh):
        i_max = i + stride * out_h
        for j in range(kw):
            j_max = j + stride * out_w
            cols[:, :, :, i, j, :] = x[:, i:i_max:stride, j:j_max:stride, :]
    return cols.reshape(N, out_h, out_w, kh * kw * C), out_h, out_w


def conv_forward(x, W, b, stride=1, pad=1):
    N, H, Wd, C = x.shape
    k = W.shape[0]
    out_ch = W.shape[-1]
    cols, out_h, out_w = im2col(x, k, k, stride, pad)
    W_col = W.reshape(-1, out_ch)
    out = cols.reshape(N * out_h * out_w, -1) @ W_col + b
    return out.reshape(N, out_h, out_w, out_ch)


def relu(x):
    return np.maximum(0, x)


def maxpool_forward(x, size=2, stride=2):
    N, H, W, C = x.shape
    out_h, out_w = H // stride, W // stride
    x = x[:, :out_h * stride, :out_w * stride, :]
    out = np.zeros((N, out_h, out_w, C), dtype=x.dtype)
    for i in range(out_h):
        for j in range(out_w):
            window = x[:, i*stride:i*stride+size, j*stride:j*stride+size, :]
            out[:, i, j, :] = window.max(axis=(1, 2))
    return out


def softmax(logits):
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


class DigitCNN:
    """Loads the Task 1 CNN weights and runs inference only (no training)."""

    def __init__(self, weights_path):
        with open(weights_path, "rb") as f:
            self.w = pickle.load(f)

    def predict_proba(self, x):
        """x: numpy array of shape (N, 8, 8, 1), pixel values already in [0, 1]."""
        w = self.w
        x = conv_forward(x, w["conv1_W"], w["conv1_b"], stride=1, pad=1)
        x = relu(x)
        x = maxpool_forward(x, 2, 2)
        x = conv_forward(x, w["conv2_W"], w["conv2_b"], stride=1, pad=1)
        x = relu(x)
        x = maxpool_forward(x, 2, 2)
        x = x.reshape(x.shape[0], -1)
        x = x @ w["fc1_W"] + w["fc1_b"]
        x = relu(x)
        logits = x @ w["fc2_W"] + w["fc2_b"]
        return softmax(logits)


# Load the model once at startup, not on every request.
model = DigitCNN(MODEL_PATH)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def parse_image(payload):
    """
    Validates and converts a single image payload into a (1, 8, 8, 1)
    float32 array normalized to [0, 1]. Raises ValueError with a clear
    message on any problem, which the route handlers turn into a 400.
    """
    if payload is None:
        raise ValueError("Missing 'image' field in request body.")

    arr = np.array(payload, dtype=np.float64)

    if arr.size != 64:
        raise ValueError(
            f"Expected an 8x8 (64-value) grayscale image, got {arr.size} values."
        )

    arr = arr.reshape(8, 8)

    if np.isnan(arr).any():
        raise ValueError("Image contains non-numeric or missing values.")

    # Accept either raw 0-16 pixel scale (like the original digits dataset)
    # or already-normalized 0-1 scale, and normalize consistently to [0, 1].
    if arr.max() > 1.0:
        if arr.max() > 16.0 or arr.min() < 0.0:
            raise ValueError("Pixel values must be within [0, 16] (or already normalized to [0, 1]).")
        arr = arr / 16.0
    elif arr.min() < 0.0:
        raise ValueError("Pixel values must not be negative.")

    return arr.reshape(1, 8, 8, 1).astype(np.float32)


def format_prediction(probs_row):
    probs_row = probs_row.tolist()
    pred_digit = int(np.argmax(probs_row))
    return {
        "predicted_digit": pred_digit,
        "confidence": round(probs_row[pred_digit], 4),
        "probabilities": {str(i): round(p, 4) for i, p in enumerate(probs_row)},
    }


# ----------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "name": "Digit Classification API",
        "description": "REST API serving the Task 1 from-scratch CNN digit classifier.",
        "endpoints": {
            "GET /health": "Health check.",
            "POST /predict": "Predict the digit for one 8x8 grayscale image. "
                              "Body: {\"image\": [64 numbers, 0-16 or 0-1, row-major 8x8]}",
            "POST /predict/batch": "Predict digits for multiple images. "
                                    "Body: {\"images\": [[64 numbers], [64 numbers], ...]}",
        },
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True)
        if data is None:
            return jsonify({"error": "Request body must be valid JSON."}), 400

        image = parse_image(data.get("image"))
        probs = model.predict_proba(image)[0]
        return jsonify(format_prediction(probs)), 200

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        app.logger.error("Unexpected error in /predict: %s\n%s", e, traceback.format_exc())
        return jsonify({"error": "Internal server error while generating prediction."}), 500


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    try:
        data = request.get_json(silent=True)
        if data is None:
            return jsonify({"error": "Request body must be valid JSON."}), 400

        images_payload = data.get("images")
        if not isinstance(images_payload, list) or len(images_payload) == 0:
            return jsonify({"error": "'images' must be a non-empty list of 64-value images."}), 400
        if len(images_payload) > 100:
            return jsonify({"error": "Batch size limited to 100 images per request."}), 400

        results = []
        for idx, img_payload in enumerate(images_payload):
            try:
                image = parse_image(img_payload)
                probs = model.predict_proba(image)[0]
                results.append(format_prediction(probs))
            except ValueError as e:
                results.append({"error": f"image[{idx}]: {e}"})

        return jsonify({"count": len(results), "results": results}), 200

    except Exception as e:
        app.logger.error("Unexpected error in /predict/batch: %s\n%s", e, traceback.format_exc())
        return jsonify({"error": "Internal server error while generating predictions."}), 500


# ----------------------------------------------------------------------
# Error handlers for common HTTP errors
# ----------------------------------------------------------------------

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found. See GET / for a list of available endpoints."}), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed for this endpoint."}), 405


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error."}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
