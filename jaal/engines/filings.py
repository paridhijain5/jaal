"""Rule-based disclosure red flags."""

import pandas as pd

RULE_WEIGHTS = {
    "auditor_resignation": 0.34,
    "pledge_increase": 0.30,
    "delayed_results": 0.22,
    "preferential_allotment": 0.20,
    "insider_sale": 0.16,
    "bulk_deal": 0.10,
    "corporate_announcement": 0.05,
}


def filing_features(filings: pd.DataFrame, stock_dates: pd.DataFrame) -> pd.DataFrame:
    """Aggregate weighted filing flags in a trailing 90-calendar-day window."""
    records = filings.copy()
    records["date"] = pd.to_datetime(records["date"])
    records["rule_weight"] = records["filing_type"].map(RULE_WEIGHTS).fillna(records["severity"].fillna(0.05))
    stock_dates = stock_dates.copy()
    stock_dates["date"] = pd.to_datetime(stock_dates["date"])
    rows = []
    for item in stock_dates.itertuples(index=False):
        recent = records[(records["symbol"] == item.symbol) & (records["date"] <= item.date)
                         & (records["date"] >= item.date - pd.Timedelta(days=90))]
        rows.append({"date": item.date, "symbol": item.symbol,
                     "filing_flag_count": len(recent), "filing_risk": min(1.0, recent["rule_weight"].sum())})
    return pd.DataFrame(rows)