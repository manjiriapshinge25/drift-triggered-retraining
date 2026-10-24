"""AWS Lambda handler for API Gateway HTTP API proxy events.

Package the project root (including src/, artifacts/models/, and artifacts/registry.json)
with the dependencies. Model promotion remains workflow-controlled via registry.json.
"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from src.core import load_production_model


def lambda_handler(event, context):
    method = (event.get("requestContext", {}).get("http", {}).get("method") or "POST").upper()
    path = event.get("rawPath", "/predict")
    if method == "GET" and path == "/health":
        try:
            _, entry = load_production_model()
            return _response(200, {"status": "ok", "model_version": entry["version"]})
        except Exception as exc:
            return _response(503, {"error": str(exc)})

    try:
        body = event.get("body") or "{}"
        if event.get("isBase64Encoded"):
            import base64
            body = base64.b64decode(body).decode("utf-8")
        payload = json.loads(body) if isinstance(body, str) else body
        features = payload.get("features", payload)
        if not isinstance(features, dict) or not features:
            return _response(400, {"error": "Send a non-empty JSON object in 'features'."})
        model, entry = load_production_model()
        X = pd.DataFrame([features])
        expected = entry["features"]
        missing = sorted(set(expected) - set(X.columns))
        if missing:
            return _response(400, {"error": f"Missing features: {missing}"})
        probability = float(model.predict_proba(X[expected])[0, 1])
        return _response(200, {
            "prediction": int(probability >= 0.5),
            "churn_probability": round(probability, 6),
            "model_version": entry["version"]
        })
    except Exception as exc:
        return _response(400, {"error": f"{type(exc).__name__}: {exc}"})


def _response(status, body):
    return {
        "statusCode": status,
        "headers": {"content-type": "application/json"},
        "body": json.dumps(body)
    }
