from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.utils import load_joblib

MODEL_DIR = Path("reports/models")

# Map model keys -> joblib files (must exist)
MODEL_FILES = {
    "lasso": MODEL_DIR / "lasso.joblib",
    "cart_lasso": MODEL_DIR / "cart_lasso.joblib",
    "rf_lasso": MODEL_DIR / "rf_lasso.joblib",
    "xgb_lasso": MODEL_DIR / "xgb_lasso.joblib",
    "cart_pca": MODEL_DIR / "cart_pca.joblib",
    "rf_pca": MODEL_DIR / "rf_pca.joblib",
    "xgb_pca": MODEL_DIR / "xgb_pca.joblib",
}

# Load processed schema to ensure consistent columns
DATA_PATH = Path("data/processed/spotify_processed.csv")

app = FastAPI(title="Spotify Popularity Prediction API", version="1.0.0")

# Load once on startup
df = pd.read_csv(DATA_PATH)
X_SCHEMA = df.drop(columns=["is_popular"])
DEFAULT_ROW = {}
for c in X_SCHEMA.columns:
    if pd.api.types.is_numeric_dtype(X_SCHEMA[c]):
        DEFAULT_ROW[c] = float(X_SCHEMA[c].median())
    else:
        mode = X_SCHEMA[c].mode(dropna=True)
        DEFAULT_ROW[c] = str(mode.iloc[0]) if not mode.empty else ""

MODELS = {}
for k, p in MODEL_FILES.items():
    if p.exists():
        MODELS[k] = load_joblib(p)


class PredictRequest(BaseModel):
    model_choice: str = "xgb_lasso"
    # Send only the fields you want; others will be filled with medians/modes
    features: Dict[str, Any]
    threshold: float = 0.5


class PredictResponse(BaseModel):
    model_choice: str
    probability: float
    label: int


@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_loaded": sorted(MODELS.keys()),
        "schema_columns": len(X_SCHEMA.columns),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    model = MODELS.get(req.model_choice)
    if model is None:
        raise HTTPException(status_code=400, detail=f"Unknown model_choice: {req.model_choice}")

    # Build 1-row input with safe defaults
    row = dict(DEFAULT_ROW)
    for k, v in req.features.items():
        if k in row:
            row[k] = v  # override only known keys

    X_one = pd.DataFrame([row], columns=X_SCHEMA.columns)

    proba = float(model.predict_proba(X_one)[:, 1][0])
    label = int(proba >= req.threshold)

    return PredictResponse(model_choice=req.model_choice, probability=proba, label=label)