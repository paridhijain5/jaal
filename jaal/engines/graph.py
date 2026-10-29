"""As-of entity overlap and community signals."""

from __future__ import annotations

import networkx as nx
import pandas as pd


def build_entity_graph(entities: pd.DataFrame, as_of: pd.Timestamp | None = None) -> nx.Graph:
    """Build a stock/entity bipartite graph using only disclosures known by date."""
    records = entities.copy()
    records["date"] = pd.to_datetime(records["date"])
    if as_of is not None:
        records = records[records["date"] <= pd.Timestamp(as_of)]
    graph = nx.Graph()
    for row in records.itertuples(index=False):
        stock = f"stock:{row.symbol}"
        entity = f"entity:{row.entity}"
        graph.add_node(stock, node_type="stock", label=row.symbol)
        graph.add_node(entity, node_type="entity", label=row.entity, entity_type=getattr(row, "entity_type", "Other"))
        graph.add_edge(stock, entity, relationship=getattr(row, "relationship", "associated"), source=getattr(row, "source", "disclosure"))
    return graph


def asof_entity_features(entities: pd.DataFrame, stock_dates: pd.DataFrame) -> pd.DataFrame:
    """Return repeat-entity risk for each (date, symbol), with no future edges."""
    records = entities.copy()
    records["date"] = pd.to_datetime(records["date"])
    stock_dates = stock_dates.copy()
    stock_dates["date"] = pd.to_datetime(stock_dates["date"])
    rows = []
    for day in sorted(stock_dates["date"].unique()):
        known = records[records["date"] <= day]
        stock_entities = known.groupby("symbol")["entity"].agg(lambda values: set(values)).to_dict()
        entity_degree = known.groupby("entity")["symbol"].nunique().to_dict()
        day_stocks = stock_dates.loc[stock_dates["date"] == day, "symbol"].unique()
        for symbol in day_stocks:
            linked = stock_entities.get(symbol, set())
            repeat_entities = [name for name in linked if entity_degree.get(name, 0) > 1]
            rows.append({"date": pd.Timestamp(day), "symbol": symbol,
                         "shared_entity_count": len(repeat_entities),
                         "max_entity_degree": max((entity_degree[name] for name in linked), default=0),
                         "entity_risk": min(1.0, sum(entity_degree[name] - 1 for name in repeat_entities) / 5)})
    return pd.DataFrame(rows)


def graph_communities(graph: nx.Graph) -> dict[str, int]:
    """Assign community ids using Louvain, with greedy modularity fallback."""
    if graph.number_of_edges() == 0:
        return {}
    try:
        communities = nx.community.louvain_communities(graph, seed=17)
    except AttributeError:
        communities = nx.community.greedy_modularity_communities(graph)
    return {node: index for index, group in enumerate(communities) for node in group}