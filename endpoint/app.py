
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src.core import load_production_model


app = FastAPI(
    title="DriftGuard | MLOps Control Center",
    description="""
## ✨ DriftGuard — Smart MLOps

Drift-triggered model retraining and customer churn prediction.

- 🟢 **Health:** Check the active production model.
- 🔮 **Prediction:** Test churn predictions using customer features.
- 📚 **Documentation:** Explore and execute API endpoints.
- 🔄 **Model lifecycle:** Production model is resolved from the registry.
""",
    version="1.0",
    docs_url=None,
    redoc_url="/redoc",
)


@app.get("/docs", include_in_schema=False)
def custom_docs():
    page = get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title="DriftGuard | MLOps Control Center",
        swagger_favicon_url="https://fastapi.tiangolo.com/img/favicon.png",
        swagger_ui_parameters={
            "defaultModelsExpandDepth": -1,
            "displayRequestDuration": True,
            "filter": True,
            "docExpansion": "list",
            "persistAuthorization": True,
        },
    )

    html = page.body.decode("utf-8")

    style = """
    <style>
    body {
        background: linear-gradient(135deg,#10152b,#17112e,#102b3b) !important;
        color: #f8fafc !important;
    }

    .swagger-ui {
        font-family: "Segoe UI", Arial, sans-serif;
    }

    .swagger-ui .topbar {
        background: linear-gradient(100deg,#6d28d9,#9333ea,#0891b2) !important;
        padding: 20px 0 !important;
        border-bottom: 3px solid #67e8f9;
    }

    .swagger-ui .topbar-wrapper .link span {
        color: white !important;
        font-size: 25px;
        font-weight: 900;
    }

    .swagger-ui .info h1,
    .swagger-ui .info h2,
    .swagger-ui .info h3,
    .swagger-ui .info p,
    .swagger-ui .info li,
    .swagger-ui .opblock-tag {
        color: #f1f5f9 !important;
    }

    .swagger-ui .info {
        margin: 35px 0;
    }

    .swagger-ui .info .title {
        color: #c4b5fd !important;
        font-size: 36px;
    }

    .swagger-ui .scheme-container,
    .swagger-ui section.models {
        background: #1e293b !important;
        border: 1px solid #475569;
        border-radius: 16px;
        box-shadow: 0 8px 28px #0004;
    }

    .swagger-ui .opblock {
        border-radius: 14px !important;
        overflow: hidden;
        margin: 18px 0 !important;
        box-shadow: 0 8px 24px #0004;
        transition: transform .2s, box-shadow .2s;
    }

    .swagger-ui .opblock:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 30px #0005;
    }

    .swagger-ui .opblock.opblock-get {
        background: linear-gradient(110deg,#102e46,#133a4b) !important;
        border: 1px solid #22d3ee !important;
    }

    .swagger-ui .opblock.opblock-post {
        background: linear-gradient(110deg,#123c35,#174b3b) !important;
        border: 1px solid #34d399 !important;
    }

    .swagger-ui .opblock .opblock-summary {
        border-color: #ffffff20 !important;
        padding: 12px;
    }

    .swagger-ui .opblock .opblock-summary-method {
        border-radius: 8px !important;
        min-width: 82px;
        font-weight: 900;
    }

    .swagger-ui .opblock-summary-description,
    .swagger-ui .opblock .opblock-summary-path,
    .swagger-ui .opblock .opblock-summary-path__deprecated {
        color: #f8fafc !important;
    }

    .swagger-ui .opblock .opblock-section-header {
        background: #263449 !important;
    }

    .swagger-ui .opblock-description-wrapper p,
    .swagger-ui .response-col_description,
    .swagger-ui .parameter__name,
    .swagger-ui .parameter__type,
    .swagger-ui label,
    .swagger-ui .tab li,
    .swagger-ui .response-col_status {
        color: #e2e8f0 !important;
    }

    .swagger-ui .btn {
        border-radius: 9px !important;
    }

    .swagger-ui .btn.execute {
        background: linear-gradient(100deg,#7c3aed,#0891b2) !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 4px 15px #7c3aed55;
    }

    .swagger-ui input,
    .swagger-ui textarea,
    .swagger-ui select {
        background: #111827 !important;
        color: white !important;
        border: 1px solid #64748b !important;
        border-radius: 7px;
    }

    .swagger-ui a {
        color: #67e8f9 !important;
    }

    .swagger-ui table thead tr td,
    .swagger-ui table thead tr th {
        color: #c4b5fd !important;
    }

    .swagger-ui .model-box,
    .swagger-ui .model {
        color: #e2e8f0 !important;
    }

    .swagger-ui .filter input {
        background: #1e293b !important;
        color: white !important;
    }

    .swagger-ui .responses-inner h4,
    .swagger-ui .responses-inner h5 {
        color: #c4b5fd !important;
    }

    @media (max-width: 600px) {
        .swagger-ui .info .title {
            font-size: 27px;
        }

        .swagger-ui .wrapper {
            padding: 0 12px;
        }
    }
    </style>
    """

    html = html.replace("</head>", style + "</head>")
    return HTMLResponse(html)


@app.get("/", include_in_schema=False)
def home():
    return {
        "project": "DriftGuard | MLOps Control Center",
        "status": "API is running",
        "model_endpoint": "/predict",
        "health_check": "/health",
        "api_documentation": "/docs",
    }


class Customer(BaseModel):
    features: dict


@app.get("/health")
def health():
    try:
        _, entry = load_production_model()
        return {
            "status": "ok",
            "model_version": entry["version"],
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/predict")
def predict(customer: Customer):
    try:
        model, entry = load_production_model()
        X = pd.DataFrame([customer.features])

        # Reject missing features and preserve the trained feature order.
        expected = entry["features"]
        missing = set(expected) - set(X.columns)

        if missing:
            raise ValueError(f"Missing features: {sorted(missing)}")

        X = X[expected]
        probability = float(model.predict_proba(X)[0, 1])
        prediction = int(probability >= 0.5)

        return {
            "prediction": prediction,
            "churn_probability": round(probability, 6),
            "model_version": entry["version"],
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
