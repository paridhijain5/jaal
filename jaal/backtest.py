"""Leakage-aware historical ranking metrics."""

import numpy as np
import pandas as pd


def backtest_metrics(scored: pd.DataFrame, k: int = 5) -> dict:
    """Evaluate top-k flags on the chronological holdout period."""
    test = scored[scored["is_test"]].copy() if "is_test" in scored else scored.copy()
    if test.empty:
        test = scored.copy()
    ranked = test.sort_values("jaal_score", ascending=False)
    top = ranked.head(k)
    positives = int(test["label"].sum())
    true_positives = int(top["label"].sum())
    flagged = test[test["jaal_score"] >= 50]
    false_positives = int((flagged["label"] == 0).sum())
    negatives = max(1, int((test["label"] == 0).sum()))
    lead_days = []
    if "next_order_date" in test:
        detected = test[(test["label"] == 1) & (test["jaal_score"] >= 35)]
        for row in detected.itertuples(index=False):
            if pd.notna(row.next_order_date):
                lead_days.append((pd.Timestamp(row.next_order_date) - pd.Timestamp(row.date)).days)
    return {"precision_at_k": true_positives / max(1, len(top)),
            "recall_at_k": true_positives / max(1, positives),
            "false_positive_rate": false_positives / negatives,
            "lead_time_days": float(np.mean(lead_days)) if lead_days else None, "top_k": top}