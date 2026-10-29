import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from jaal.backtest import backtest_metrics
from jaal.dashboard import get_dashboard, inject_styles, page_header

st.set_page_config(page_title="JAAL | Backtest", page_icon="◉", layout="wide")
inject_styles()
bundle = get_dashboard()
scored = bundle["scored"]
page_header("Backtest", "Chronological holdout comparison; planted sample outcomes are synthetic labels.")
k = st.slider("Top-ranked cases (k)", min_value=1, max_value=20, value=5)
holdout = scored[scored["is_test"]].copy()
jaal = backtest_metrics(holdout, k)
price = holdout.copy()
price["jaal_score"] = (price["anomaly_score"] * 100).round().astype(int)
baseline = backtest_metrics(price, k)
st.caption(f"Training observations precede {bundle['test_start']:%d %b %Y}; all reported metrics use later dates only.")
columns = st.columns(4)
columns[0].metric(f"JAAL precision@{k}", f"{jaal['precision_at_k']:.1%}", f"price-only {baseline['precision_at_k']:.1%}")
columns[1].metric(f"JAAL recall@{k}", f"{jaal['recall_at_k']:.1%}", f"price-only {baseline['recall_at_k']:.1%}")
columns[2].metric("JAAL false-positive rate", f"{jaal['false_positive_rate']:.1%}", f"price-only {baseline['false_positive_rate']:.1%}")
columns[3].metric("Mean lead time", f"{jaal['lead_time_days']:.1f} days" if jaal["lead_time_days"] is not None else "n/a",
                  f"price-only {baseline['lead_time_days']:.1f} days" if baseline["lead_time_days"] is not None else "")
results = pd.DataFrame({"Metric": [f"Precision@{k}", f"Recall@{k}", "False-positive rate", "Mean lead time (days)"],
    "JAAL": [jaal["precision_at_k"], jaal["recall_at_k"], jaal["false_positive_rate"], jaal["lead_time_days"] or 0],
    "Price-only": [baseline["precision_at_k"], baseline["recall_at_k"], baseline["false_positive_rate"], baseline["lead_time_days"] or 0]})
figure = go.Figure()
figure.add_trace(go.Bar(x=results["Metric"], y=results["JAAL"], name="JAAL", marker_color="#48e2e8"))
figure.add_trace(go.Bar(x=results["Metric"], y=results["Price-only"], name="Price-only baseline", marker_color="#e84c5b"))
figure.update_layout(barmode="group", height=330, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#09171c", yaxis={"rangemode": "tozero"}, margin={"t": 25, "b": 5})
st.plotly_chart(figure, width="stretch")
st.subheader("Highest-ranked holdout signals")
top = jaal["top_k"].copy()
st.dataframe(top[["date", "symbol", "jaal_score", "anomaly_score", "entity_risk", "filing_risk", "label", "next_order_date"]], hide_index=True, width="stretch",
    column_config={"jaal_score": st.column_config.ProgressColumn("Jaal Score", min_value=0, max_value=100),
                   "label": st.column_config.CheckboxColumn("Penalised label")})
st.markdown("<div class='jaal-warning'>These metrics measure recovery of synthetic sample labels, not real-world predictive performance. The prototype is not investment advice or an allegation system.</div>", unsafe_allow_html=True)