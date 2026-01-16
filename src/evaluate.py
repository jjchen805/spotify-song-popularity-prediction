from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve
)

from .config import Config
from .utils import load_joblib, ensure_dir, to_json

def evaluate_binary(model, X_test, y_test) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    return {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    }

def plot_roc(models: dict, X_test, y_test, outpath: Path) -> None:
    plt.figure()
    for name, model in models.items():
        proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        plt.plot(fpr, tpr, label=name)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate trained models.")
    p.add_argument("--data", required=True, help="Processed CSV path")
    p.add_argument("--modeldir", default="reports/models", help="Directory containing joblib models")
    p.add_argument("--outdir", default="reports", help="Directory to write evaluation outputs")
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

    modeldir = Path(args.modeldir)
    models = {
        "LASSO_LogReg": load_joblib(modeldir / "lasso_logreg.joblib"),
        "CART_Tuned": load_joblib(modeldir / "cart_best.joblib"),
    }

    results = {name: evaluate_binary(m, X_test, y_test) for name, m in models.items()}

    outdir = Path(args.outdir)
    ensure_dir(outdir / "figures")
    to_json(results, outdir / "metrics.json")

    # Save a simple markdown summary
    lines = ["# Evaluation Summary", "", "| Model | ROC-AUC | Accuracy | Precision | Recall | F1 |", "|---|---:|---:|---:|---:|---:|"]
    for name, r in results.items():
        lines.append(f"| {name} | {r['roc_auc']:.4f} | {r['accuracy']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} |")
    (outdir / "results.md").write_text("\n".join(lines), encoding="utf-8")

    plot_roc(models, X_test, y_test, outdir / "figures" / "roc_curves.png")

    print(f"Wrote metrics to { (outdir / 'metrics.json').resolve() }")
    print(f"Wrote summary to { (outdir / 'results.md').resolve() }")
    print(f"Wrote ROC plot to { (outdir / 'figures' / 'roc_curves.png').resolve() }")

if __name__ == "__main__":
    main()
