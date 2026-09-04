from __future__ import annotations

import random
from typing import Optional

import graphviz as gv

from ..opt import tonic


def _unique_node_identifier(identifier: tonic.QepsIdentifier) -> str:
    return str(hash((identifier, random.random())))


def _render_subquery_path(qeps: tonic.QEPsNode, current_node: str, current_graph: gv.Digraph) -> None:
    for identifier, qeps_child in qeps.child_nodes.items():
        child_node = _make_node_label(identifier, qeps_child)
        node_identifier = _unique_node_identifier(identifier)
        current_graph.node(node_identifier, label=child_node, style="dashed")
        current_graph.edge(current_node, node_identifier, style="dashed")
        _render_subquery_path(qeps_child, node_identifier, current_graph)


def _make_node_label(identifier: tonic.QepsIdentifier, node: tonic.QEPsNode) -> str:
    cost_str = (
        "[" + ", ".join(f"{operator.value}={cost}" for operator, cost in node.operator_costs.items()) + "]"
        if node.operator_costs
        else ""
    )
    label = str(identifier)
    return label + "\n" + cost_str


def plot_tonic_qeps(
    qeps: tonic.QEPsNode | tonic.QEPSynopsis,
    *,
    _current_node: Optional[str] = None,
    _current_graph: Optional[gv.Digraph] = None,
) -> gv.Digraph:
    if not _current_graph:
        _current_graph = gv.Digraph()

    if isinstance(qeps, tonic.QEPSynopsis):
        _current_node = "∅"
        _current_graph.node(_current_node, style="dotted")
        qeps = qeps.root

    if qeps.subquery_root:
        _render_subquery_path(qeps.subquery_root, _current_node, _current_graph)

    for identifier, qeps_child in qeps.child_nodes.items():
        child_node = _make_node_label(identifier, qeps_child)
        node_identifier = _unique_node_identifier(identifier)
        if qeps_child.subquery_root:
            _current_graph.node(node_identifier, label=child_node, style="dashed")
        else:
            _current_graph.node(node_identifier, label=child_node)
        if _current_node:
            _current_graph.edge(_current_node, node_identifier)
        plot_tonic_qeps(qeps_child, _current_node=node_identifier, _current_graph=_current_graph)

    return _current_graph
