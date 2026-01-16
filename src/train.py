from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel

from xgboost import XGBClassifier

from .config import Config
from .features import build_preprocessor_onehot, build_preprocessor_ordinal
from .utils import save_joblib, to_json

def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--outdir", default="reports/models")
    p.add_argument("--pca-components", type=int, default=12)
    # LASSO selector aggressiveness (higher C -> less regularization -> more features kept)
    p.add_argument("--lasso-C", type=float, default=1.0)
    return p

def main():
    args = build_argparser().parse_args()
    cfg = Config()

    df = pd.read_csv(args.data)
    X = df.drop(columns=["is_popular"])
    y = df["is_popular"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, stratify=y, random_state=cfg.random_state
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ---------------- LASSO branch (OneHot) ----------------
    pre_onehot = build_preprocessor_onehot(X_train)

    # Base LASSO model (also useful as its own model)
    lasso = Pipeline([
        ("preprocessor", pre_onehot),
        ("clf", LogisticRegression(
            solver="saga",
            l1_ratio=1.0,     # L1
            C=args.lasso_C,
            max_iter=8000,
            random_state=cfg.random_state
        ))
    ])
    lasso.fit(X_train, y_train)

    # CART + LASSO-preprocess
    cart_lasso = Pipeline([
        ("preprocessor", pre_onehot),
        ("clf", DecisionTreeClassifier(random_state=cfg.random_state))
    ])
    cart_grid = GridSearchCV(
        cart_lasso,
        param_grid={
            "clf__max_depth": [3, 5, 7, None],
            "clf__min_samples_split": [2, 5, 10],
            "clf__min_samples_leaf": [1, 2, 4],
        },
        cv=cfg.cv_folds,
        scoring=cfg.scoring,
        n_jobs=-1
    )
    cart_grid.fit(X_train, y_train)
    cart_lasso_best = cart_grid.best_estimator_

    rf_lasso = Pipeline([
        ("preprocessor", pre_onehot),
        ("clf", RandomForestClassifier(
            n_estimators=400,
            random_state=cfg.random_state,
            n_jobs=-1
        ))
    ])
    rf_lasso.fit(X_train, y_train)

    # ✅ XGB with LASSO-selected features (SelectFromModel)
    lasso_selector = SelectFromModel(
        estimator=LogisticRegression(
            solver="saga",
            l1_ratio=1.0,
            C=args.lasso_C,
            max_iter=8000,
            random_state=cfg.random_state
        ),
        threshold=1e-8  # keep non-zero-ish coefficients
    )

    xgb_lasso = Pipeline([
        ("preprocessor", pre_onehot),
        ("select", lasso_selector),
        ("clf", XGBClassifier(
            n_estimators=200,
            learning_rate=0.08,
            max_depth=5,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.01,
            reg_lambda=1.0,
            random_state=cfg.random_state,
            n_jobs=-1,
            eval_metric="logloss"
        ))
    ])
    xgb_lasso.fit(X_train, y_train)

    # ---------------- PCA branch (Ordinal -> PCA) ----------------
    pre_ord = build_preprocessor_ordinal(X_train)

    cart_pca = Pipeline([
        ("preprocessor", pre_ord),
        ("pca", PCA(n_components=args.pca_components, random_state=cfg.random_state)),
        ("clf", DecisionTreeClassifier(random_state=cfg.random_state))
    ])
    cart_pca_grid = GridSearchCV(
        cart_pca,
        param_grid={
            "clf__max_depth": [3, 5, 7, None],
            "clf__min_samples_split": [2, 5, 10],
            "clf__min_samples_leaf": [1, 2, 4],
        },
        cv=cfg.cv_folds,
        scoring=cfg.scoring,
        n_jobs=-1
    )
    cart_pca_grid.fit(X_train, y_train)
    cart_pca_best = cart_pca_grid.best_estimator_

    rf_pca = Pipeline([
        ("preprocessor", pre_ord),
        ("pca", PCA(n_components=args.pca_components, random_state=cfg.random_state)),
        ("clf", RandomForestClassifier(
            n_estimators=400,
            random_state=cfg.random_state,
            n_jobs=-1
        ))
    ])
    rf_pca.fit(X_train, y_train)

    # ✅ XGB on PCA components
    xgb_pca = Pipeline([
        ("preprocessor", pre_ord),
        ("pca", PCA(n_components=args.pca_components, random_state=cfg.random_state)),
        ("clf", XGBClassifier(
            n_estimators=250,
            learning_rate=0.08,
            max_depth=6,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=cfg.random_state,
            n_jobs=-1,
            eval_metric="logloss"
        ))
    ])
    xgb_pca.fit(X_train, y_train)

    # Save artifacts
    save_joblib(lasso, outdir / "lasso.joblib")
    save_joblib(cart_lasso_best, outdir / "cart_lasso.joblib")
    save_joblib(rf_lasso, outdir / "rf_lasso.joblib")
    save_joblib(xgb_lasso, outdir / "xgb_lasso.joblib")

    save_joblib(cart_pca_best, outdir / "cart_pca.joblib")
    save_joblib(rf_pca, outdir / "rf_pca.joblib")
    save_joblib(xgb_pca, outdir / "xgb_pca.joblib")

    meta = {
        "data": args.data,
        "pca_components": args.pca_components,
        "lasso_C": args.lasso_C,
        "cart_lasso_best_params": cart_grid.best_params_,
        "cart_pca_best_params": cart_pca_grid.best_params_,
        "cart_lasso_best_cv": float(cart_grid.best_score_),
        "cart_pca_best_cv": float(cart_pca_grid.best_score_),
    }
    to_json(meta, outdir / "train_metadata.json")

    print(f"Saved models to {outdir.resolve()}")

if __name__ == "__main__":
    main()