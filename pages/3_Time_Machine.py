import pandas as pd
import streamlit as st

from jaal.dashboard import get_dashboard, inject_styles, page_header

st.set_page_config(page_title="JAAL | Time Machine", page_icon="◉", layout="wide")
inject_styles()
bundle = get_dashboard()
scored = bundle["scored"]
page_header("Time Machine", "Replay the information set that was available on a historical trading date.")
minimum, maximum = scored["date"].min().date(), scored["date"].max().date()
selected = st.slider("Replay date", min_value=minimum, max_value=maximum, value=maximum, step=pd.Timedelta(days=1))
day = scored[scored["date"] <= pd.Timestamp(selected)]["date"].max()
snapshot = scored[scored["date"] == day].sort_values("jaal_score", ascending=False)
st.caption(f"Showing the latest available trading observation on or before {selected:%d %b %Y}: {day:%d %b %Y}")
left, middle, right = st.columns(3)
left.metric("Stocks flagged 60+", int((snapshot["jaal_score"] >= 60).sum()))
middle.metric("Highest score", int(snapshot["jaal_score"].max()))
right.metric("Repeat-entity links", int((snapshot["shared_entity_count"] > 0).sum()))
st.dataframe(snapshot[["symbol", "company", "close", "return_1d", "volume_zscore", "shared_entity_count", "filing_flag_count", "jaal_score"]],
    hide_index=True, width="stretch", column_config={
        "close": st.column_config.NumberColumn("Close (INR)", format="₹%.2f"),
        "return_1d": st.column_config.NumberColumn("1D return", format="%.2%%"),
        "volume_zscore": st.column_config.NumberColumn("Volume z-score", format="%.1f"),
        "jaal_score": st.column_config.ProgressColumn("Jaal Score", min_value=0, max_value=100)})
selected_symbol = st.selectbox("Show score history", sorted(scored["symbol"].unique()))
history = scored[(scored["symbol"] == selected_symbol) & (scored["date"] <= day)]
st.line_chart(history.set_index("date")["jaal_score"], color="#48e2e8")
st.caption("Replay scores are calculated from trailing price history and disclosures known by each replay date.")