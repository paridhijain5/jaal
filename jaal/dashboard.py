"""Shared Streamlit state, theme helpers, and chart conventions."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from jaal.features import build_features
from jaal.ingest import load_sample_data
from jaal.model import add_jaal_scores, explain_row, time_split, train_model


@st.cache_resource(show_spinner="Building offline risk signals...")
def get_dashboard() -> dict:
    data = load_sample_data()
    features = build_features(data)
    trained = train_model(features)
    scored = add_jaal_scores(features, trained)
    _, test = time_split(features)
    test_start = test["date"].min() if not test.empty else scored["date"].max()
    scored["is_test"] = scored["date"] >= test_start
    return {"data": data, "features": features, "trained": trained, "scored": scored,
            "test_start": pd.Timestamp(test_start)}


def inject_styles() -> None:
    st.markdown("""<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    .block-container { max-width: 1440px; padding-top: 2rem; padding-bottom: 3rem; }
    [data-testid="stSidebar"] { border-right: 1px solid #18333b; }
    [data-testid="stMetric"] { background: linear-gradient(135deg,#10242a,#0b171c); border: 1px solid #1b3c44; padding: 16px 18px; border-radius: 6px; }
    [data-testid="stMetricValue"] { font-family: 'IBM Plex Mono', monospace; }
    div[data-testid="stDataFrame"] { border: 1px solid #1b3c44; }
    .jaal-kicker { color: #48e2e8; font: 600 11px 'IBM Plex Mono', monospace; text-transform: uppercase; }
    .jaal-warning { border-left: 3px solid #e84c5b; background: #26171c; padding: 12px 16px; color: #ffd9dc; margin: 12px 0 20px; }
    .jaal-divider { border: 0; border-top: 1px solid #1b3c44; margin: 18px 0; }
    </style>""", unsafe_allow_html=True)
    st.sidebar.markdown("<div class='jaal-kicker'>JAAL / SIGNAL DESK</div>", unsafe_allow_html=True)
    st.sidebar.caption("Indian small-cap surveillance research")
    st.sidebar.markdown("<hr class='jaal-divider'>", unsafe_allow_html=True)
    st.sidebar.markdown("<div class='jaal-warning'>Risk signals are not accusations. Verify independently before acting.</div>", unsafe_allow_html=True)
    st.sidebar.caption("Demo uses synthetic, planted sample cases. No live trading or enforcement inference.")


def page_header(title: str, subtitle: str) -> None:
    st.markdown("<div class='jaal-kicker'>MARKET INTEGRITY / EARLY WARNING</div>", unsafe_allow_html=True)
    st.title(title)
    st.caption(subtitle)


def latest_snapshot(scored: pd.DataFrame) -> pd.DataFrame:
    latest = scored[scored["date"] == scored["date"].max()].copy()
    return latest.sort_values("jaal_score", ascending=False)


def reasons_for(row: pd.Series, trained: dict) -> list[tuple[str, float]]:
    names = {"volume_zscore": "Abnormal traded volume", "delivery_spike": "Delivery concentration shift",
             "entity_risk": "Repeat-entity overlap", "filing_risk": "Filing red flags",
             "anomaly_score": "Price-volume anomaly", "return_1d": "Daily price move",
             "return_volatility": "Elevated volatility", "shared_entity_count": "Shared entities",
             "max_entity_degree": "Maximum network reach", "filing_flag_count": "Recent filing count",
             "intraday_range_pct": "Intraday range"}
    return [(names.get(name, name), value) for name, value in explain_row(row, trained)]