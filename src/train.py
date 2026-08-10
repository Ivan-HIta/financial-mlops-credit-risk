from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src.data import make_dataset

FEATURES = ["income", "debt", "utilization", "delinquencies_12m", "tenure_months"]


def train(output: str | Path = "artifacts", seed: int = 11) -> dict:
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    data = make_dataset(seed=seed)
    train_df, test_df = train_test_split(data, test_size=0.25, random_state=seed, stratify=data["default"])
    model = CalibratedClassifierCV(LogisticRegression(max_iter=500, class_weight="balanced"), cv=3)
    model.fit(train_df[FEATURES], train_df["default"])
    probability = model.predict_proba(test_df[FEATURES])[:, 1]
    metrics = {
        "roc_auc": round(float(roc_auc_score(test_df["default"], probability)), 4),
        "average_precision": round(float(average_precision_score(test_df["default"], probability)), 4),
        "decision_threshold": 0.35,
        "features": FEATURES,
        "training_rows": len(train_df),
        "test_rows": len(test_df),
        "seed": seed,
    }
    joblib.dump(model, out / "credit_risk_model.joblib")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out / "model_card.md").write_text(
        "# Credit Risk Model Card\n\n"
        "Synthetic demonstration only. The model is not suitable for real credit decisions.\n\n"
        f"- ROC AUC: {metrics['roc_auc']}\n- Average precision: {metrics['average_precision']}\n"
        f"- Decision threshold: {metrics['decision_threshold']}\n"
        "- Required controls: fairness analysis, adverse-action reasons, human review, drift monitoring and approval workflow.\n",
        encoding="utf-8",
    )
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts")
    args = parser.parse_args()
    print(json.dumps(train(args.output), indent=2))

