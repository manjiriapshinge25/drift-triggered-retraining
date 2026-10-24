import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from src.core import load_production_model, prepare_xy, DATA

app=FastAPI(title="Drift-Triggered Churn Endpoint",version="1.0")

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
