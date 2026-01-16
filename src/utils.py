from __future__ import annotations
from pathlib import Path
import json
import joblib
import numpy as np

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_joblib(obj, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    joblib.dump(obj, path)

def load_joblib(path: str | Path):
    return joblib.load(Path(path))

def to_json(obj, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def safe_feature_names(preprocessor) -> list[str]:
    """Best-effort extraction of feature names from a fitted ColumnTransformer."""
    names = []
    for name, trans, cols in preprocessor.transformers_:
        if name == "remainder":
            continue
        if hasattr(trans, "get_feature_names_out"):
            try:
                out = trans.get_feature_names_out(cols)
                names.extend(list(out))
            except Exception:
                names.extend([f"{name}__{c}" for c in cols])
        else:
            names.extend([f"{name}__{c}" for c in cols])
    return names
