"""
Sample client script demonstrating requests to the Digit Classification API.
Run app.py first (in a separate terminal), then run this script.
"""

import json
import requests

BASE_URL = "http://127.0.0.1:5000"

# A hand-written-style '0' pattern, 0-16 pixel scale (8x8, row-major)
sample_zero = [
    0, 0, 5, 13, 9, 1, 0, 0,
    0, 0, 13, 15, 10, 15, 5, 0,
    0, 3, 15, 2, 0, 11, 8, 0,
    0, 4, 12, 0, 0, 8, 8, 0,
    0, 5, 8, 0, 0, 9, 8, 0,
    0, 4, 11, 0, 1, 12, 7, 0,
    0, 2, 14, 5, 10, 12, 0, 0,
    0, 0, 6, 13, 10, 0, 0, 0,
]

print("== API info ==")
print(json.dumps(requests.get(f"{BASE_URL}/").json(), indent=2))

print("\n== Health check ==")
print(json.dumps(requests.get(f"{BASE_URL}/health").json(), indent=2))

print("\n== Single prediction ==")
resp = requests.post(f"{BASE_URL}/predict", json={"image": sample_zero})
print(json.dumps(resp.json(), indent=2))

print("\n== Batch prediction (same image twice) ==")
resp = requests.post(f"{BASE_URL}/predict/batch", json={"images": [sample_zero, sample_zero]})
print(json.dumps(resp.json(), indent=2))

print("\n== Error case: missing image field ==")
resp = requests.post(f"{BASE_URL}/predict", json={})
print("Status:", resp.status_code)
print(json.dumps(resp.json(), indent=2))

print("\n== Error case: wrong-size image ==")
resp = requests.post(f"{BASE_URL}/predict", json={"image": [1, 2, 3]})
print("Status:", resp.status_code)
print(json.dumps(resp.json(), indent=2))
