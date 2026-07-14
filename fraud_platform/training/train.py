"""Train the fraud-detection model on PaySim (or synthetic PaySim) data.

Pipeline:
  1. Load ``data/paysim.csv`` (generate synthetic data if it is missing).
  2. Restrict to the transaction types where fraud can occur (TRANSFER,
     CASH_OUT) — scoring PAYMENT/CASH_IN/DEBIT wastes signal because they are
     never fraudulent in PaySim.
  3. Engineer features (src/features/engineering.py).
  4. Train a classifier with class weighting for the extreme imbalance.
  5. Evaluate (ROC-AUC, PR-AUC, precision/recall/F1) and persist the model
     plus a JSON metadata sidecar.

Run:
    python -m fraud_platform.training.train
    python -m fraud_platform.training.train --algorithm gradient_boosting
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from fraud_platform.config import CONFIG
from fraud_platform.data.generate_synthetic import generate
from fraud_platform.data.validate import describe, validate_paysim
from fraud_platform.features.engineering import FEATURE_COLUMNS, build_features
from fraud_platform.logging_config import get_logger

logger = get_logger(__name__)


def load_dataset(path: str) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        logger.info("%s not found; generating synthetic PaySim data", csv_path)
        df = generate(n_rows=CONFIG.data.synthetic_rows, seed=CONFIG.data.random_state)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
    else:
        logger.info("Loading dataset from %s", csv_path)
        df = pd.read_csv(csv_path)
    # Validate + normalise so a real Kaggle CSV (or a mirror with slightly
    # different column spellings) trains exactly like the synthetic one.
    df = validate_paysim(df)
    logger.info("Dataset: %s", describe(df))
    return df


def build_model(algorithm: str, scale_pos_weight: float = 1.0):
    if algorithm == "random_forest":
        return RandomForestClassifier(
            n_estimators=CONFIG.model.n_estimators,
            max_depth=CONFIG.model.max_depth,
            class_weight=CONFIG.model.class_weight,
            n_jobs=-1,
            random_state=CONFIG.data.random_state,
        )
    if algorithm == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=CONFIG.model.n_estimators,
            max_depth=3,
            random_state=CONFIG.data.random_state,
        )
    if algorithm == "xgboost":
        # Imported lazily so the rest of the platform doesn't require xgboost.
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=CONFIG.model.n_estimators,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            # Counter the extreme class imbalance (neg/pos ratio).
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr",
            n_jobs=-1,
            random_state=CONFIG.data.random_state,
        )
    if algorithm == "logistic":
        return LogisticRegression(
            class_weight=CONFIG.model.class_weight,
            max_iter=1000,
        )
    raise ValueError(f"Unknown algorithm: {algorithm}")


def train(algorithm: str | None = None) -> dict:
    algorithm = algorithm or CONFIG.model.algorithm
    df = load_dataset(CONFIG.data.raw_path)

    # Focus on fraud-eligible transaction types.
    scored_types = list(CONFIG.features.scored_types)
    subset = df[df["type"].isin(scored_types)].copy()
    logger.info(
        "Total rows: %s | scored (%s): %s | fraud in scope: %s",
        f"{len(df):,}",
        "/".join(scored_types),
        f"{len(subset):,}",
        f"{int(subset['isFraud'].sum()):,}",
    )

    X = build_features(subset)
    y = subset["isFraud"].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=CONFIG.data.test_size,
        random_state=CONFIG.data.random_state,
        stratify=y,
    )

    # For imbalance-aware models (xgboost), weight the positive class.
    n_pos = int(y_train.sum())
    scale_pos_weight = (len(y_train) - n_pos) / max(n_pos, 1)
    model = build_model(algorithm, scale_pos_weight=scale_pos_weight)
    logger.info("Training %s on %s rows", algorithm, f"{len(X_train):,}")
    start = time.perf_counter()
    model.fit(X_train, y_train)
    train_seconds = time.perf_counter() - start

    proba = model.predict_proba(X_test)[:, 1]
    threshold = float(CONFIG.model.decision_threshold)
    preds = (proba >= threshold).astype(int)

    roc_auc = float(roc_auc_score(y_test, proba))
    pr_auc = float(average_precision_score(y_test, proba))
    report = classification_report(y_test, preds, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, preds).tolist()

    logger.info("Trained in %.1fs", train_seconds)
    logger.info("ROC-AUC: %.4f | PR-AUC: %.4f", roc_auc, pr_auc)
    logger.info(
        "Fraud precision/recall/F1: %.4f / %.4f / %.4f",
        report["1"]["precision"],
        report["1"]["recall"],
        report["1"]["f1-score"],
    )
    logger.info("Confusion matrix [[TN, FP], [FN, TP]]: %s", cm)

    # Persist model + metadata sidecar.
    model_path = Path(CONFIG.model.path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    importances = None
    if hasattr(model, "feature_importances_"):
        importances = dict(
            sorted(
                zip(
                    FEATURE_COLUMNS,
                    model.feature_importances_.tolist(),
                    strict=True,
                ),
                key=lambda kv: kv[1],
                reverse=True,
            )
        )

    metadata = {
        "algorithm": algorithm,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "feature_columns": FEATURE_COLUMNS,
        "scored_types": scored_types,
        "decision_threshold": threshold,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "fraud_rate": float(np.mean(y)),
        "metrics": {
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "precision_fraud": report["1"]["precision"],
            "recall_fraud": report["1"]["recall"],
            "f1_fraud": report["1"]["f1-score"],
            "confusion_matrix": cm,
        },
        "feature_importances": importances,
        "train_seconds": train_seconds,
    }
    meta_path = Path(CONFIG.model.metadata_path)
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    logger.info("Saved model -> %s", model_path)
    logger.info("Saved metadata -> %s", meta_path)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train fraud detection model")
    parser.add_argument(
        "--algorithm",
        choices=["random_forest", "gradient_boosting", "xgboost", "logistic"],
        default=None,
    )
    args = parser.parse_args()
    train(algorithm=args.algorithm)


if __name__ == "__main__":
    main()
