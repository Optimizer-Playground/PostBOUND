from __future__ import annotations

import typing

import graphviz as gv
import matplotlib as mpl
import networkx as nx


@typing.overload
def plot_graph(graph: nx.Graph) -> gv.Graph:
    pass


@typing.overload
def plot_graph(graph: nx.DiGraph) -> gv.Digraph:
    pass


def plot_graph(graph: nx.Graph | nx.DiGraph, *, directed: bool | None = None, color: str = "") -> gv.Graph | gv.Digraph:
    if directed is None:
        gv_graph = gv.Digraph() if isinstance(graph, nx.DiGraph) else gv.Graph()
    else:
        gv_graph = gv.Digraph() if directed else gv.Graph()

    unique_color_labels = set()
    color_mapping: dict = {}
    if color:
        for _node, d in graph.nodes.data():
            unique_color_labels.add(d[color])
        viridis = mpl.cm.viridis
        normalized_colors = mpl.colors.Normalize(vmin=0, vmax=len(unique_color_labels) - 1)
        for i, color_label in enumerate(unique_color_labels):
            color_mapping[color_label] = mpl.colors.rgb2hex(viridis(normalized_colors(i)))
    for n, d in graph.nodes.data():
        atts = {"color": color_mapping[d[color]]} if color_mapping else {}
        gv_graph.node(str(n), label=gv.escape(str(n)), style="bold", **atts)
    for s, t in graph.edges:
        gv_graph.edge(str(s), str(t))

    return gv_graph
