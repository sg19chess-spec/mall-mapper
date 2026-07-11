"""Routable indoor topology graph: nodes are physical features (Store,
Escalator, Entrance, Intersection), edges are walkable connections. Built
in-memory from published IndoorFeatures at request time by the Indoor
Mapping Agent -- not a separate store of record.
"""
from __future__ import annotations

import networkx as nx


def build_indoor_graph(features: list[dict]) -> nx.Graph:
    g = nx.Graph()
    for f in features:
        g.add_node(f["feature_id"], feature_type=f["feature_type"], floor=f.get("floor"),
                   name=f.get("properties", {}).get("name"))

    # naive walkability: connect every feature to the nearest corridor node
    # on the same floor (a real system would use actual corridor geometry).
    corridors = [f for f in features if f["feature_type"] == "corridor"]
    for f in features:
        if f["feature_type"] == "corridor":
            continue
        same_floor_corridors = [c for c in corridors if c.get("floor") == f.get("floor")]
        if same_floor_corridors:
            g.add_edge(f["feature_id"], same_floor_corridors[0]["feature_id"], relation="walkable")

    # escalators/elevators connect corridors on adjacent floors
    for f in features:
        if f["feature_type"] in ("escalator", "elevator"):
            floor = f.get("floor")
            for c in corridors:
                if c.get("floor") in (floor - 1, floor + 1):
                    g.add_edge(f["feature_id"], c["feature_id"], relation="connects_floor")

    return g


def shortest_route(g: nx.Graph, feature_a: str, feature_b: str) -> list[str] | None:
    try:
        return nx.shortest_path(g, feature_a, feature_b)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None
