from __future__ import annotations
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from ml.feature_engineering import FeatureEngineer
from app.config import PROJECT_ROOT, settings


class ReturnPredictor:
    """Loads trained artefacts and scores individual or batch orders."""

    def __init__(self, model_dir: str | None = None) -> None:
        self.model_dir = PROJECT_ROOT / (model_dir or settings.model_dir)
        self.model: Any = None
        self.fe: FeatureEngineer | None = None
        self.model_info: dict = {}
        self._loaded = False

    def load(self) -> None:
        model_path = self.model_dir / "best_model.pkl"
        fe_path = self.model_dir / "feature_engineer.pkl"
        info_path = self.model_dir / "model_info.json"

        if not model_path.exists():
            raise FileNotFoundError(
                f"No trained model found at '{model_path}'. "
                "Run  python train_model.py  first."
            )

        self.model = joblib.load(model_path)
        self.fe = FeatureEngineer.load(str(fe_path))
        with open(info_path) as f:
            self.model_info = json.load(f)
        self._loaded = True

    def predict(self, record: dict) -> tuple[float, np.ndarray]:
        """Return (probability, shap_values) for a single order record."""
        self._check_loaded()
        X = self.fe.transform_single(record)        # (1, n_features)
        prob = float(self.model.predict_proba(X)[0, 1])
        return prob, X

    def predict_batch(self, records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        """Return (probabilities, X_array) for a list of records."""
        self._check_loaded()
        import pandas as pd
        X = self.fe.transform(pd.DataFrame(records))
        probs = self.model.predict_proba(X)[:, 1]
        return probs, X

    def _check_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError("Predictor not loaded. Call .load() first.")

    @property
    def feature_names(self) -> list[str]:
        return self.model_info.get("feature_names", [])

    @property
    def meta(self) -> dict:
        return {
            "name": self.model_info.get("model_name", "Unknown"),
            "version": self.model_info.get("version", "1.0.0"),
            "trained_at": self.model_info.get("trained_at", ""),
        }


# Module-level singleton — loaded once at API startup
predictor = ReturnPredictor()
