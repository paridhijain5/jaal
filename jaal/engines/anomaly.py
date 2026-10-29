"""Robust price/volume anomaly features."""

import numpy as np
import pandas as pd


def add_anomaly_features(prices: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Add trailing-only price and volume features, avoiding look-ahead."""
    result = prices.sort_values(["symbol", "date"]).copy()
    grouped = result.groupby("symbol", group_keys=False)
    result["return_1d"] = grouped["close"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)
    result["return_volatility"] = grouped["return_1d"].transform(
        lambda values: values.rolling(window, min_periods=5).std()
    ).fillna(0)
    result["volume_zscore"] = grouped["volume"].transform(
        lambda values: (values - values.shift(1).rolling(window, min_periods=5).mean())
        / values.shift(1).rolling(window, min_periods=5).std().replace(0, np.nan).fillna(
            values.shift(1).rolling(window, min_periods=5).mean() * 0.1
        )
    ).replace([np.inf, -np.inf], np.nan).fillna(0).clip(-8, 8)
    result["delivery_spike"] = grouped["delivery_pct"].transform(
        lambda values: values - values.shift(1).rolling(window, min_periods=5).median()
    ).fillna(0)
    result["intraday_range_pct"] = ((result["high"] - result["low"]) / result["close"].replace(0, np.nan)).fillna(0)
    result["anomaly_score"] = (
        0.42 * result["volume_zscore"].clip(lower=0).div(5).clip(upper=1)
        + 0.28 * (result["return_1d"].abs() / (result["return_volatility"] + 0.015)).clip(upper=3).div(3)
        + 0.18 * (result["delivery_spike"] / 40).clip(lower=0, upper=1)
        + 0.12 * result["intraday_range_pct"].clip(upper=0.2).div(0.2)
    ).clip(0, 1)
    return result