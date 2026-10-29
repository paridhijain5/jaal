import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import streamlit as st

from jaal.dashboard import get_dashboard, inject_styles, page_header
from jaal.engines.graph import build_entity_graph, graph_communities

st.set_page_config(page_title="JAAL | Ring Radar", page_icon="◉", layout="wide")
inject_styles()
bundle = get_dashboard()
page_header("Ring Radar", "Explore shared disclosed entities and the stocks connected to them.")
as_of = st.date_input("Relationships known as of", value=bundle["scored"]["date"].max().date())
graph = build_entity_graph(bundle["data"]["entities"], as_of=as_of)
communities = graph_communities(graph)
entities = sorted(node[7:] for node, attrs in graph.nodes(data=True) if attrs["node_type"] == "entity")
if not entities:
    st.info("No relationships have been disclosed by this date.")
    st.stop()
selected = st.selectbox("Inspect entity", entities)
neighbors = sorted(graph.neighbors(f"entity:{selected}"))
touched = sorted(graph.nodes[node]["label"] for node in neighbors if graph.nodes[node]["node_type"] == "stock")
left, right = st.columns([2, 1])
with left:
    positions = nx.spring_layout(graph, seed=17, k=0.9)
    edge_x, edge_y = [], []
    for start, end in graph.edges:
        edge_x.extend([positions[start][0], positions[end][0], None])
        edge_y.extend([positions[start][1], positions[end][1], None])
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line={"width": 1, "color": "#25434b"}, hoverinfo="skip", showlegend=False))
    stock_nodes = [node for node, attrs in graph.nodes(data=True) if attrs["node_type"] == "stock"]
    entity_nodes = [node for node, attrs in graph.nodes(data=True) if attrs["node_type"] == "entity"]
    for nodes, kind, color, size in [(stock_nodes, "Stock", "#48e2e8", 16), (entity_nodes, "Entity", "#e84c5b", 12)]:
        figure.add_trace(go.Scatter(x=[positions[node][0] for node in nodes], y=[positions[node][1] for node in nodes],
            mode="markers+text", name=kind, text=[graph.nodes[node]["label"] for node in nodes], textposition="top center",
            textfont={"size": 10, "color": "#d2e3e6"}, marker={"size": [size + 7 if node == f"entity:{selected}" else size for node in nodes],
            "color": ["#ffffff" if node == f"entity:{selected}" else color for node in nodes],
            "line": {"width": 1, "color": "#071116"}}, hovertemplate="%{text}<extra>" + kind + "</extra>"))
    figure.update_layout(height=590, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#09171c",
        showlegend=True, hovermode="closest", margin={"t": 10, "b": 10, "l": 5, "r": 5},
        xaxis={"visible": False}, yaxis={"visible": False})
    st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})
with right:
    st.metric("Connected stocks", len(touched))
    st.write(f"**{selected}**")
    for symbol in touched:
        node = f"stock:{symbol}"
        community = communities.get(node)
        st.write(f"{symbol} · community {community + 1 if community is not None else 'n/a'}")
    st.caption("Edges represent disclosed relationships. Network proximity does not imply collusion.")
    st.write("**Observed links**")
    details = bundle["data"]["entities"]
    details = details[(details["entity"] == selected) & (details["date"] <= pd.Timestamp(as_of))]
    st.dataframe(details[["date", "symbol", "relationship", "source"]].sort_values("date"), hide_index=True, width="stretch")