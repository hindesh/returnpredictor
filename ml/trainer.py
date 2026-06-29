from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from tabulate import tabulate
from xgboost import XGBClassifier

from ml.data_generator import generate_synthetic_data
from ml.feature_engineering import FEATURE_NAMES, FeatureEngineer


def _business_cost(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Expected total cost: intervention cost for FP + missed-return cost for FN."""
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    return fp * 2.50 + fn * (75.0 * 0.25 + 10.0)


def train_and_benchmark(output_dir: str = "trained_models") -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    _banner("ReturnShield — ML Training Pipeline")

    # 1 · Generate data
    print("\n[1/5] Generating synthetic training data …")
    df = generate_synthetic_data(n_samples=50_000)
    print(f"      {len(df):,} orders generated | return rate: {df['is_returned'].mean():.1%}")

    # 2 · Feature engineering
    print("\n[2/5] Engineering features …")
    fe = FeatureEngineer()
    X = fe.fit_transform(df)
    y = df["is_returned"].values
    print(f"      Feature matrix: {X.shape[0]:,} × {X.shape[1]}")

    # 3 · Train / test split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 4 · Define model zoo
    models: dict[str, Any] = {
        "CatBoostClassifier": CatBoostClassifier(
            iterations=500, learning_rate=0.05, depth=6,
            eval_metric="AUC", random_seed=42, verbose=0,
        ),
        "XGBClassifier": XGBClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            eval_metric="logloss", random_state=42, n_jobs=-1,
        ),
        "LGBMClassifier": LGBMClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            min_child_samples=20, random_state=42, n_jobs=-1, verbose=-1,
        ),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=300, max_depth=8, random_state=42, n_jobs=-1,
        ),
        "GradientBoostingClassifier": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42,
        ),
    }

    # 5 · Cross-validation
    print("\n[3/5] Running 5-fold cross-validation …")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "roc_auc": "roc_auc",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
    }

    rows: list[dict] = []
    for name, model in models.items():
        print(f"      {name:<35}", end="", flush=True)
        t0 = time.time()
        scores = cross_validate(model, X_tr, y_tr, cv=cv, scoring=scoring, n_jobs=1)
        elapsed = time.time() - t0
        row = {
            "Model": name,
            "AUC-ROC": round(float(np.mean(scores["test_roc_auc"])), 4),
            "Precision": round(float(np.mean(scores["test_precision"])), 4),
            "Recall": round(float(np.mean(scores["test_recall"])), 4),
            "F1": round(float(np.mean(scores["test_f1"])), 4),
            "Time(s)": round(elapsed, 1),
        }
        rows.append(row)
        print(f"AUC={row['AUC-ROC']:.4f}  F1={row['F1']:.4f}  {elapsed:.1f}s")

    results_df = pd.DataFrame(rows).sort_values("AUC-ROC", ascending=False).reset_index(drop=True)
    print(f"\n{'─'*60}")
    print("Model Benchmark (5-fold CV on training set)")
    print(tabulate(results_df.values.tolist(), headers=list(results_df.columns),
                   tablefmt="rounded_outline", floatfmt=".4f"))

    # 6 · Select & retrain best model
    best_name = results_df.iloc[0]["Model"]
    best_model = models[best_name]
    print(f"\n[4/5] Best model: {best_name}")
    print("      Retraining on full training set …")
    best_model.fit(X_tr, y_tr)

    # 7 · Test-set evaluation
    y_prob = best_model.predict_proba(X_te)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    test_metrics = {
        "roc_auc": round(float(roc_auc_score(y_te, y_prob)), 4),
        "precision": round(float(precision_score(y_te, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_te, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_te, y_pred, zero_division=0)), 4),
        "business_cost": round(_business_cost(y_te, y_pred), 2),
    }
    print("\n      Hold-out test-set metrics:")
    for k, v in test_metrics.items():
        print(f"        {k}: {v}")

    # 8 · Save artefacts
    print(f"\n[5/5] Saving artefacts to '{output_dir}/' …")
    joblib.dump(best_model, output_path / "best_model.pkl")
    fe.save(str(output_path / "feature_engineer.pkl"))

    model_info: dict[str, Any] = {
        "model_name": best_name,
        "version": "1.0.0",
        "trained_at": pd.Timestamp.now().isoformat(),
        "n_features": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
        "training_samples": int(len(X_tr)),
        "test_metrics": test_metrics,
        "cv_results": rows,
    }
    with open(output_path / "model_info.json", "w") as f:
        json.dump(model_info, f, indent=2)

    print(f"      ✓ best_model.pkl")
    print(f"      ✓ feature_engineer.pkl")
    print(f"      ✓ model_info.json")
    _banner("Training complete! Run  python run.py  to start the API.")
    return model_info


def _banner(msg: str) -> None:
    line = "=" * 60
    print(f"\n{line}\n{msg}\n{line}")
