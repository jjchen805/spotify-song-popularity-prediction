from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    random_state: int = 42
    test_size: float = 0.2

    # Label definition
    popularity_quantile: float = 0.90

    # Modeling
    cv_folds: int = 5
    scoring: str = "roc_auc"

    # If True: keep only numeric features (closest to your notebook’s simplified path)
    numeric_only: bool = True
