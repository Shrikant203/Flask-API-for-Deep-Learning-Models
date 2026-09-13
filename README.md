# Digit Classification API — Documentation

A Flask REST API that serves the CNN digit-classification model trained from
scratch in Task 1 (`cnn_digits_model.pkl`). The API accepts an 8x8 grayscale
image (the same format as the scikit-learn Digits dataset used in Task 1) and
returns the predicted digit along with a full probability distribution.

## Running the API

```bash
pip install flask numpy
python app.py
```

The server starts on `http://127.0.0.1:5000/`.

## Endpoints

### `GET /`
Returns basic API information and a list of available endpoints.

### `GET /health`
Health check. Returns whether the model loaded successfully.

**Response**
```json
{"status": "ok", "model_loaded": true}
```

### `POST /predict`
Predicts the digit for a single image.

**Request body**
```json
{"image": [64 numbers]}
```
- `image` must contain exactly 64 numbers (an 8x8 image, given row-major).
- Pixel values may be on the original 0–16 scale (as in the raw Digits
  dataset) or already normalized to 0–1 — both are accepted and handled
  consistently.

**Successful response — `200 OK`**
```json
{
  "predicted_digit": 3,
  "confidence": 0.9998,
  "probabilities": {"0": 0.0, "1": 0.0, "2": 0.0, "3": 0.9998, "4": 0.0,
                     "5": 0.0, "6": 0.0, "7": 0.0, "8": 0.0, "9": 0.0001}
}
```

**Error responses**
| Status | Cause |
|---|---|
| `400` | Body is not valid JSON |
| `400` | `image` field missing |
| `400` | `image` does not contain exactly 64 numeric values |
| `400` | Pixel values out of the expected range, or non-numeric |
| `500` | Unexpected server-side error |

### `POST /predict/batch`
Predicts digits for multiple images in a single request (maximum 100 images
per request).

**Request body**
```json
{"images": [[64 numbers], [64 numbers], ...]}
```

**Successful response — `200 OK`**
```json
{
  "count": 2,
  "results": [
    {"predicted_digit": 3, "confidence": 0.9998, "probabilities": {...}},
    {"predicted_digit": 7, "confidence": 0.9827, "probabilities": {...}}
  ]
}
```
If an individual image in the batch is invalid, its entry in `results`
contains an `"error"` message instead of a prediction, so one bad image does
not fail the whole batch.

## Error Handling Summary
- **404** — unknown endpoint (a helpful message points back to `GET /`).
- **405** — endpoint called with the wrong HTTP method (e.g. `GET /predict`).
- **500** — any unexpected internal error; the full traceback is logged
  server-side, but only a generic message is returned to the client.
- All validation errors return **structured JSON** (`{"error": "..."}`)
  rather than raw stack traces, so client applications can parse failures
  reliably.

## Example Usage (Python `requests`)
```python
import requests

resp = requests.post(
    "http://127.0.0.1:5000/predict",
    json={"image": [0,0,5,13,9,1,0,0, 0,0,13,15,10,15,5,0,
                     0,3,15,2,0,11,8,0, 0,4,12,0,0,8,8,0,
                     0,5,8,0,0,9,8,0, 0,4,11,0,1,12,7,0,
                     0,2,14,5,10,12,0,0, 0,0,6,13,10,0,0,0]}
)
print(resp.json())
```

## Example Usage (curl)
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"image": [0,0,5,13,9,1,0,0, 0,0,13,15,10,15,5,0, 0,3,15,2,0,11,8,0, 0,4,12,0,0,8,8,0, 0,5,8,0,0,9,8,0, 0,4,11,0,1,12,7,0, 0,2,14,5,10,12,0,0, 0,0,6,13,10,0,0,0]}'
```
