"""Model loading and scoring wrapper.

A thin layer around the persisted joblib model that:
  * loads the model + metadata once and caches it,
  * runs the shared feature engineering,
  * enforces the PaySim rule that only TRANSFER / CASH_OUT can be fraud
    (everything else is scored 0 without troubling the model).

Both the FastAPI service and the Spark streaming job go through this class so
their scoring behaviour is identical.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import joblib
import pandas as pd

from fraud_platform.config import CONFIG
from fraud_platform.features.engineering import build_features


class FraudModel:
    def __init__(
        self,
        model_path: str | None = None,
        metadata_path: str | None = None,
        threshold: float | None = None,
    ) -> None:
        self.model_path = Path(model_path or CONFIG.model.path)
        self.metadata_path = Path(metadata_path or CONFIG.model.metadata_path)
        self.threshold = (
            threshold
            if threshold is not None
            else float(CONFIG.model.decision_threshold)
        )
        self.scored_types = set(CONFIG.features.scored_types)
        self._model = None
        self.metadata: dict = {}

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> FraudModel:
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. "
                "Train it first: python -m fraud_platform.training.train"
            )
        self._model = joblib.load(self.model_path)
        if self.metadata_path.exists():
            with open(self.metadata_path, encoding="utf-8") as fh:
                self.metadata = json.load(fh)
            # Prefer the threshold the model was calibrated/reported with.
            self.threshold = float(
                self.metadata.get("decision_threshold", self.threshold)
            )
        return self

    def score_frame(self, df: pd.DataFrame) -> pd.Series:
        """Return fraud probabilities for a batch of raw transactions."""
        if self._model is None:
            raise RuntimeError("Model not loaded; call load() first.")
        probs = pd.Series(0.0, index=df.index, dtype=float)
        eligible = df["type"].astype(str).isin(self.scored_types)
        if eligible.any():
            feats = build_features(df.loc[eligible])
            probs.loc[eligible] = self._model.predict_proba(feats)[:, 1]
        return probs

    def score_one(self, tx: Mapping[str, object]) -> dict:
        """Score a single transaction dict -> {probability, is_fraud, ...}."""
        tx_type = str(tx.get("type", ""))
        if tx_type not in self.scored_types:
            return {
                "fraud_probability": 0.0,
                "is_fraud": False,
                "threshold": self.threshold,
                "scored": False,
                "reason": f"type '{tx_type}' is never fraudulent in PaySim",
            }
        prob = float(self.score_frame(pd.DataFrame([dict(tx)])).iloc[0])
        return {
            "fraud_probability": prob,
            "is_fraud": prob >= self.threshold,
            "threshold": self.threshold,
            "scored": True,
        }
