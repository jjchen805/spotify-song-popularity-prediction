from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.cloud import storage

GCS_BUCKET = os.getenv("GCS_BUCKET", "spotify-model-buck")
GCS_PREFIX = os.getenv("GCS_PREFIX", "models") 

MODEL_FILES = {
    "lasso": f"{GCS_PREFIX}/lasso.joblib",
    "cart_lasso": f"{GCS_PREFIX}/cart_lasso.joblib",
    "rf_lasso": f"{GCS_PREFIX}/rf_lasso.joblib",
    "xgb_lasso": f"{GCS_PREFIX}/xgb_lasso.joblib",
    "cart_pca": f"{GCS_PREFIX}/cart_pca.joblib",
    "rf_pca": f"{GCS_PREFIX}/rf_pca.joblib",
    "xgb_pca": f"{GCS_PREFIX}/xgb_pca.joblib",
}

def load_joblib_from_gcs(bucket_name: str, blob_path: str):
    client = storage.Client.create_anonymous_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)

    if not blob.exists(client):
        raise FileNotFoundError(f"GCS missing: gs://{bucket_name}/{blob_path}")

    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=True) as tmp:
        blob.download_to_filename(tmp.name)
        return joblib.load(tmp.name)
    
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
MODEL_LOAD_ERRORS = {}

for k, blob_path in MODEL_FILES.items():
    try:
        MODELS[k] = load_joblib_from_gcs(GCS_BUCKET, blob_path)
    except Exception as e:
        MODEL_LOAD_ERRORS[k] = f"{type(e).__name__}: {e}"


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
        "model_load_errors": MODEL_LOAD_ERRORS,
        "gcs_bucket": GCS_BUCKET,
        "gcs_prefix": GCS_PREFIX,
        "model_files": MODEL_FILES,
        "schema_columns": len(X_SCHEMA.columns),
        "data_path": str(DATA_PATH),
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