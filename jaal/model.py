"""Time-aware model training, calibrated scores, and feature explanations."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import average_precision_score

from jaal.features import MODEL_FEATURES


def time_split(features: pd.DataFrame, test_fraction: float = 0.25) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = np.array(sorted(pd.to_datetime(features["date"]).unique()))
    boundary = dates[max(1, int(len(dates) * (1 - test_fraction)))] if len(dates) > 1 else dates[0]
    return features[features["date"] < boundary].copy(), features[features["date"] >= boundary].copy()


def train_model(features: pd.DataFrame) -> dict:
    """Fit LightGBM when installed; retain a sklearn fallback for lightweight demos."""
    train, test = time_split(features)
    if train["label"].nunique() < 2:
        train = features
    X = train[MODEL_FEATURES].fillna(0)
    y = train["label"].astype(int)
    try:
        from lightgbm import LGBMClassifier

        model = LGBMClassifier(n_estimators=100, max_depth=4, learning_rate=0.04,
                               num_leaves=15, class_weight="balanced", verbosity=-1, random_state=17)
        model_kind = "LightGBM"
    except ImportError:
        model = HistGradientBoostingClassifier(max_iter=100, l2_regularization=1.0, random_state=17)
        model_kind = "HistGradientBoosting fallback"
    if y.nunique() < 2:
        model = RandomForestClassifier(n_estimators=25, class_weight="balanced", random_state=17)
        model_kind = "RandomForest fallback"
    model.fit(X, y)
    probability = model.predict_proba(test[MODEL_FEATURES].fillna(0))[:, 1] if not test.empty else np.array([])
    metric = float(average_precision_score(test["label"], probability)) if len(test) and test["label"].nunique() > 1 else None
    return {"model": model, "model_kind": model_kind, "train": train, "test": test, "average_precision": metric}


def add_jaal_scores(features: pd.DataFrame, trained: dict) -> pd.DataFrame:
    result = features.copy()
    probability = trained["model"].predict_proba(result[MODEL_FEATURES].fillna(0))[:, 1]
    blended = 0.45 * result["anomaly_score"] + 0.25 * result["entity_risk"] + 0.15 * result["filing_risk"] + 0.15 * probability
    result["model_probability"] = probability
    result["jaal_score"] = np.rint(100 * blended.clip(0, 1)).astype(int)
    return result


def explain_row(row: pd.Series, trained: dict, top_n: int = 4) -> list[tuple[str, float]]:
    """Return SHAP-ranked reasons, with a transparent importance fallback."""
    model = trained["model"]
    sample = row[MODEL_FEATURES].to_frame().T.fillna(0)
    try:
        import shap

        values = np.asarray(shap.TreeExplainer(model).shap_values(sample))
        if values.ndim == 3:
            values = values[0, :, -1]
        elif values.ndim == 2:
            values = values[-1] if values.shape[0] == 1 else values[:, -1]
        return sorted(zip(MODEL_FEATURES, np.asarray(values).reshape(-1).tolist()),
                      key=lambda pair: abs(pair[1]), reverse=True)[:top_n]
    except Exception:
        importance = getattr(model, "feature_importances_", np.ones(len(MODEL_FEATURES)) / len(MODEL_FEATURES))
        baseline = trained["train"][MODEL_FEATURES].mean()
        values = (sample.iloc[0] - baseline).to_numpy() * np.asarray(importance)
        return sorted(zip(MODEL_FEATURES, values.tolist()), key=lambda pair: abs(pair[1]), reverse=True)[:top_n]