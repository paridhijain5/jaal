"""Feature table assembly for daily stock risk scoring."""

import numpy as np
import pandas as pd

from jaal.engines.anomaly import add_anomaly_features
from jaal.engines.filings import filing_features
from jaal.engines.graph import asof_entity_features

MODEL_FEATURES = [
    "return_1d", "return_volatility", "volume_zscore", "delivery_spike",
    "intraday_range_pct", "anomaly_score", "shared_entity_count", "max_entity_degree",
    "entity_risk", "filing_flag_count", "filing_risk",
]


def build_features(data: dict[str, pd.DataFrame], label_horizon: int = 20) -> pd.DataFrame:
    """Build daily features and forward SEBI-order labels from source tables."""
    prices = add_anomaly_features(data["prices"])
    keys = prices[["date", "symbol"]].copy()
    graph = asof_entity_features(data["entities"], keys)
    filings = filing_features(data["filings"], keys)
    features = prices.merge(graph, on=["date", "symbol"], how="left").merge(filings, on=["date", "symbol"], how="left")
    features[MODEL_FEATURES] = features[MODEL_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
    features["label"] = 0
    orders = data["orders"].copy()
    if not orders.empty:
        orders["order_date"] = pd.to_datetime(orders["order_date"])
        order_days = orders.groupby("symbol")["order_date"].apply(list).to_dict()
        features["next_order_date"] = pd.NaT
        for symbol, indices in features.groupby("symbol").groups.items():
            future_orders = np.array(order_days.get(symbol, []), dtype="datetime64[ns]")
            if len(future_orders):
                current = pd.to_datetime(features.loc[indices, "date"]).to_numpy(dtype="datetime64[ns]")
                delta = (future_orders[None, :] - current[:, None]).astype("timedelta64[D]").astype(int)
                next_index = np.where((delta >= 0).any(axis=1), np.where(delta >= 0, delta, np.iinfo(np.int32).max).argmin(axis=1), -1)
                labels = (delta >= 0) & (delta <= label_horizon)
                features.loc[indices, "label"] = labels.any(axis=1).astype(int)
                valid = next_index >= 0
                next_dates = np.full(len(indices), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
                next_dates[valid] = future_orders[next_index[valid]]
                features.loc[indices, "next_order_date"] = next_dates
    return features.sort_values(["date", "symbol"]).reset_index(drop=True)