from __future__ import annotations

import json
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.monitoring import drift_status, psi

ARTIFACTS = Path("artifacts")
MODEL_PATH = ARTIFACTS / "credit_risk_model.joblib"
app = FastAPI(title="Financial MLOps Credit Risk API", version="1.0.0")


class Applicant(BaseModel):
    income: float = Field(gt=0)
    debt: float = Field(ge=0)
    utilization: float = Field(ge=0, le=1)
    delinquencies_12m: int = Field(ge=0, le=50)
    tenure_months: int = Field(ge=0, le=600)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}


@app.post("/predict")
def predict(applicant: Applicant):
    if not MODEL_PATH.exists():
        raise HTTPException(503, "Train the model first: python -m src.train")
    model = joblib.load(MODEL_PATH)
    probability = float(model.predict_proba([[applicant.model_dump()[key] for key in ["income", "debt", "utilization", "delinquencies_12m", "tenure_months"]]])[0, 1])
    return {"default_probability": round(probability, 4), "review_required": probability >= 0.35}


@app.get("/monitoring/drift")
def monitoring_drift():
    # Deterministic synthetic example; in production these arrays come from feature logs.
    import numpy as np
    baseline = np.linspace(0.1, 0.9, 500)
    current = np.linspace(0.2, 1.0, 500)
    value = psi(baseline, current)
    return {"feature": "utilization", "psi": round(value, 4), "status": drift_status(value)}

