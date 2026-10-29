import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from jaal.dashboard import get_dashboard, inject_styles, page_header, reasons_for
from jaal.engines.case_twin import find_case_twin

st.set_page_config(page_title="JAAL | Stock Detail", page_icon="◉", layout="wide")
inject_styles()
bundle = get_dashboard()
scored, data = bundle["scored"], bundle["data"]
page_header("Stock Detail", "Inspect the score, evidence trail, and nearest historical enforcement pattern.")

symbols = sorted(scored["symbol"].unique())
symbol = st.selectbox("Company", symbols, format_func=lambda value: f"{value}  ·  {scored.loc[scored.symbol == value, 'company'].iloc[0]}")
history = scored[scored["symbol"] == symbol].sort_values("date")
row = history.iloc[-1]
gauge, reasons = st.columns([1, 2])
with gauge:
    figure = go.Figure(go.Indicator(mode="gauge+number", value=int(row["jaal_score"]),
        title={"text": "JAAL SCORE / 100", "font": {"color": "#b8cbd0", "size": 14}},
        gauge={"axis": {"range": [0, 100], "tickcolor": "#7b969c"},
               "bar": {"color": "#48e2e8" if row["jaal_score"] < 60 else "#e84c5b"},
               "bgcolor": "#0d1d24", "bordercolor": "#29434a",
               "steps": [{"range": [0, 35], "color": "#10242a"}, {"range": [35, 60], "color": "#382c1b"}, {"range": [60, 100], "color": "#361c24"}],
               "threshold": {"line": {"color": "#e84c5b", "width": 3}, "thickness": 0.8, "value": 60}}))
    figure.update_layout(height=250, margin={"t": 35, "b": 10, "l": 25, "r": 25}, paper_bgcolor="rgba(0,0,0,0)", font_color="#e7f2f4")
    st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})
with reasons:
    st.subheader("Top contributing signals")
    for label, impact in reasons_for(row, bundle["trained"]):
        st.markdown(f"**{label}**  ·  {'+' if impact >= 0 else ''}{impact:.3f}")
    st.caption(f"Model: {bundle['trained']['model_kind']} · latest observation: {row['date']:%d %b %Y}")

price_chart = make_subplots(specs=[[{"secondary_y": True}]])
price_chart.add_trace(go.Scatter(x=history["date"], y=history["close"], name="Close", line={"color": "#48e2e8", "width": 2}), secondary_y=False)
flags = history[history["anomaly_score"] >= history["anomaly_score"].quantile(0.97)]
price_chart.add_trace(go.Scatter(x=flags["date"], y=flags["close"], mode="markers", name="Flagged day", marker={"color": "#ed5260", "size": 9, "symbol": "diamond"}), secondary_y=False)
price_chart.add_trace(go.Bar(x=history["date"], y=history["volume"], name="Volume", marker_color="#426b73", opacity=0.35), secondary_y=True)
price_chart.update_layout(height=360, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#09171c", hovermode="x unified", legend={"orientation": "h", "y": 1.12}, margin={"t": 25, "b": 5})
price_chart.update_yaxes(title_text="Close (INR)", secondary_y=False, gridcolor="#18333b")
price_chart.update_yaxes(title_text="Volume", secondary_y=True, showgrid=False)
st.plotly_chart(price_chart, width="stretch")

evidence_col, case_col = st.columns([1, 1])
with evidence_col:
    st.subheader("Disclosed evidence")
    filings = data["filings"].query("symbol == @symbol").sort_values("date", ascending=False)
    if filings.empty:
        st.info("No sample filings for this stock.")
    else:
        st.dataframe(filings[["date", "filing_type", "description", "entity"]], hide_index=True, width="stretch")
    entity_rows = data["entities"].query("symbol == @symbol")
    st.caption("Disclosed relationships (not proof of coordination)")
    st.dataframe(entity_rows[["date", "entity", "entity_type", "relationship", "source"]], hide_index=True, width="stretch")
with case_col:
    st.subheader("Case Twin")
    twin = find_case_twin(row, scored, data["orders"])
    if twin:
        other = scored[(scored["symbol"] == twin["symbol"]) & (scored["date"] == twin["order_date"])].iloc[0]
        left, right = st.columns(2)
        left.metric(f"Current · {symbol}", f"{int(row['jaal_score'])}/100", row["date"].strftime("%d %b %Y"))
        right.metric(f"Past case · {twin['symbol']}", f"{int(other['jaal_score'])}/100", twin["order_date"].strftime("%d %b %Y"))
        st.progress(max(0.0, min(1.0, twin["similarity"])), text=f"Cosine similarity · {twin['similarity']:.0%}")
        st.write(f"**{twin['case_id']} · {twin['order_type']}**")
        st.write(twin["summary"])
        st.caption("Synthetic teaching case. Similarity is not a prediction of enforcement.")
    else:
        st.info("No earlier penalised sample pattern is available for comparison.")