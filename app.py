"""JAAL Watchlist: an offline-first early-warning research dashboard."""

import streamlit as st

from jaal.dashboard import get_dashboard, inject_styles, latest_snapshot, page_header

st.set_page_config(page_title="JAAL | Watchlist", page_icon="◉", layout="wide")
inject_styles()
bundle = get_dashboard()
scored = bundle["scored"]
page_header("Watchlist", "Ranked daily risk signals across the synthetic small-cap universe.")

latest = latest_snapshot(scored)
first, second, third = st.columns(3)
first.metric("Universe", f"{latest['symbol'].nunique()}", "offline sample")
second.metric("High-risk signals", f"{int((latest['jaal_score'] >= 60).sum())}", "score 60+")
third.metric("As-of date", latest["date"].max().strftime("%d %b %Y"), bundle["trained"]["model_kind"])

left, right = st.columns([1, 2])
threshold = left.slider("Minimum Jaal Score", min_value=0, max_value=100, value=0, step=5)
only_network = right.checkbox("Only stocks with repeat-entity overlap", value=False)
view = latest[latest["jaal_score"] >= threshold].copy()
if only_network:
    view = view[view["shared_entity_count"] > 0]
view["signal"] = view["jaal_score"].map(lambda score: "ELEVATED" if score >= 60 else "WATCH" if score >= 35 else "BASELINE")
st.dataframe(view[["symbol", "company", "close", "return_1d", "volume_zscore", "shared_entity_count", "filing_flag_count", "jaal_score", "signal"]],
             hide_index=True, width="stretch", column_config={
                 "close": st.column_config.NumberColumn("Close (INR)", format="₹%.2f"),
                 "return_1d": st.column_config.NumberColumn("1D return", format="%.2%%"),
                 "volume_zscore": st.column_config.NumberColumn("Volume z-score", format="%.1f"),
                 "jaal_score": st.column_config.ProgressColumn("Jaal Score", min_value=0, max_value=100),
             })
st.caption("Scores combine trailing price-volume behavior, disclosed entity overlap, filing rules, and a time-split LightGBM model.")
st.markdown("<div class='jaal-warning'>A high score indicates a pattern worth reviewing. It does not establish intent, misconduct, or legal liability.</div>", unsafe_allow_html=True)