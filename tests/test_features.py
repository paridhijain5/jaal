import pandas as pd

from jaal.engines.anomaly import add_anomaly_features
from jaal.features import build_features
from jaal.ingest import load_sample_data


def test_anomaly_features_are_finite_and_trailing():
    dates = pd.bdate_range("2025-01-01", periods=30)
    frame = pd.DataFrame({"date": dates, "symbol": "TEST", "open": 10.0, "high": 10.2,
                          "low": 9.8, "close": [10 + i * 0.02 for i in range(30)],
                          "volume": [1000] * 29 + [8000], "delivery_pct": [30] * 29 + [75]})
    features = add_anomaly_features(frame)
    assert features.iloc[-1]["volume_zscore"] > 5
    assert features.iloc[-1]["anomaly_score"] > features.iloc[-2]["anomaly_score"]
    assert features[["anomaly_score", "return_volatility"]].notna().all().all()


def test_build_features_uses_asof_overlap_and_forward_labels():
    features = build_features(load_sample_data())
    assert {"entity_risk", "filing_risk", "label", "anomaly_score"}.issubset(features.columns)
    assert features["label"].isin([0, 1]).all()
    early = features[(features["symbol"] == "JAAL02") & (features["date"] < pd.Timestamp("2024-02-20"))]
    assert early["shared_entity_count"].max() == 0
    assert features["label"].sum() > 0