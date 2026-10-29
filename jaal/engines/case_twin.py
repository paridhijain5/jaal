"""Nearest prior penalised-pattern comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

CASE_FEATURES = ["anomaly_score", "volume_zscore", "delivery_spike", "entity_risk", "filing_risk"]


def find_case_twin(row: pd.Series, features: pd.DataFrame, orders: pd.DataFrame) -> dict | None:
    """Find the most similar known SEBI-penalised stock pattern from the past."""
    if orders.empty:
        return None
    candidates = orders.copy()
    candidates["order_date"] = pd.to_datetime(candidates["order_date"])
    candidates = candidates[candidates["order_date"] <= pd.Timestamp(row["date"])]
    if candidates.empty:
        return None
    indexed = features.set_index(["date", "symbol"])
    vectors, valid = [], []
    for order in candidates.itertuples(index=False):
        key = (pd.Timestamp(order.order_date), order.symbol)
        if key in indexed.index:
            vectors.append(indexed.loc[key, CASE_FEATURES].to_numpy(dtype=float))
            valid.append(order)
    if not vectors:
        return None
    matrix = np.nan_to_num(np.asarray(vectors), nan=0.0, posinf=0.0, neginf=0.0)
    query = np.nan_to_num(row[CASE_FEATURES].to_numpy(dtype=float).reshape(1, -1))
    scaler = StandardScaler().fit(matrix)
    similarity = cosine_similarity(scaler.transform(query), scaler.transform(matrix))[0]
    best = int(similarity.argmax())
    order = valid[best]
    return {"symbol": order.symbol, "case_id": order.case_id, "order_date": pd.Timestamp(order.order_date),
            "similarity": float(similarity[best]), "summary": order.summary, "order_type": order.order_type}