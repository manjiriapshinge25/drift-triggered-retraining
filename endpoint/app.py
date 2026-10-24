import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from src.core import load_production_model, prepare_xy, DATA

app = FastAPI(
    title="DriftGuard",
    description="✨ Smart MLOps | Drift Detection & Churn Prediction",
    version="1.0",
    docs_url=None,
    redoc_url="/redoc"
)

from fastapi.responses import HTMLResponse
from fastapi.openapi.docs import get_swagger_ui_html

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
            "docExpansion": "list"
        }
    )

    html = page.body.decode("utf-8")
    html = html.replace(
        "</head>",
        """
        <style>
        body {
            background: #0b1020 !important;
            color: #edf2ff !important;
        }
        .swagger-ui {
            font-family: Inter, Arial, sans-serif;
        }
        .swagger-ui .topbar {
            background: linear-gradient(110deg,#111827,#312e81,#6d28d9) !important;
            padding: 14px 0;
            border-bottom: 2px solid #a78bfa;
        }
        .swagger-ui .topbar-wrapper .link span {
            color: white !important;
            font-size: 23px;
            font-weight: 800;
        }
        .swagger-ui .info,
        .swagger-ui .info h1,
        .swagger-ui .info h2,
        .swagger-ui .info p,
        .swagger-ui .opblock-tag {
            color: #edf2ff !important;
        }
        .swagger-ui .scheme-container,
        .swagger-ui section.models {
            background: #141c30 !important;
            border: 1px solid #334155;
            border-radius: 14px;
        }
        .swagger-ui .opblock {
            border-radius: 12px !important;
            overflow: hidden;
            margin: 14px 0;
            box-shadow: 0 8px 25px #0003;
        }
        .swagger-ui .opblock.opblock-get {
            background: #102b3a;
            border-color: #22d3ee;
        }
        .swagger-ui .opblock.opblock-post {
            background: #12352d;
            border-color: #34d399;
        }
        .swagger-ui .opblock .opblock-summary {
            border-color: #ffffff20;
        }
        .swagger-ui .opblock .opblock-summary-method {
            border-radius: 7px;
            min-width: 75px;
        }
        .swagger-ui .btn {
            border-radius: 8px;
        }
        .swagger-ui input,
        .swagger-ui textarea,
        .swagger-ui select {
            background: #0b1020 !important;
            color: #edf2ff !important;
            border: 1px solid #64748b !important;
        }
        .swagger-ui table thead tr td,
        .swagger-ui table thead tr th {
            color: #c4b5fd !important;
        }
        .swagger-ui .model-box,
        .swagger-ui .model {
            color: #dbeafe !important;
        }
        .swagger-ui .response-col_status {
            color: #34d399 !important;
        }
        .swagger-ui a {
            color: #67e8f9 !important;
        }
        .swagger-ui .btn.execute {
            background: linear-gradient(90deg,#7c3aed,#0891b2) !important;
            border: none !important;
            color: white !important;
        }
        </style>
        </head>
        """
    )

    return HTMLResponse(html)


@app.get("/")
def home():
    return {
        "project": "Drift-Triggered Churn Prediction",
        "status": "API is running",
        "model_endpoint": "/predict",
        "health_check": "/health",
        "api_documentation": "/docs"
    }

class Customer(BaseModel):
    features: dict

@app.get("/health")
def health():
    try:
        _,entry=load_production_model()
        return {"status":"ok","model_version":entry["version"]}
    except Exception as e:
        raise HTTPException(status_code=503,detail=str(e))

@app.post("/predict")
def predict(customer: Customer):
    try:
        model,entry=load_production_model()
        X=pd.DataFrame([customer.features])
        # Align incoming features to the trained feature list; reject missing features.
        expected=entry["features"]
        missing=set(expected)-set(X.columns)
        if missing: raise ValueError(f"Missing features: {sorted(missing)}")
        X=X[expected]
        probability=float(model.predict_proba(X)[0,1])
        pred=int(probability>=0.5)
        return {"prediction":pred,"churn_probability":round(probability,6),
                "model_version":entry["version"]}
    except Exception as e:
        raise HTTPException(status_code=400,detail=str(e))
