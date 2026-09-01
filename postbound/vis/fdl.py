"""Force-directed layout algorithms"""

from __future__ import annotations

import collections
import random
import typing
from collections.abc import Callable, Hashable, Iterable

import networkx as nx
import numpy as np

T = typing.TypeVar("T", bound=Hashable)
Debug = True

if Debug:
    random.seed(321)
    np.random.seed(321)


def force_directed_layout(elements: Iterable[T], difference_score: Callable[[T, T], float]) -> dict[T, np.ndarray]:
    """Lays out the supplied elements in a 2D-space according to the difference score.

    Pairs of points with a large difference score are positioned further apart than points with a low difference score.

    The returned dictionary maps each of the input element to the pair of (x, y) coordinates.
    """
    return DefaultLayoutEngine(elements, difference_score)


def kamada_kawai_layout(elements: Iterable[T], difference_score: Callable[[T, T], float]) -> dict[T, np.ndarray]:
    elements = list(elements)
    layout_graph = nx.complete_graph(elements)

    distance_map = collections.defaultdict(dict)
    for a_idx, a in enumerate(elements):
        for b in elements[a_idx:]:
            current_score = difference_score(a, b)
            distance_map[a][b] = current_score
            distance_map[b][a] = current_score

    elem_pos_spread = len(elements)
    initial_pos = {elem: (random.random() * elem_pos_spread, random.random() * elem_pos_spread) for elem in elements}
    return nx.kamada_kawai_layout(layout_graph, dist=distance_map, pos=initial_pos)


def fruchterman_reingold_layout(
    elements: Iterable[T],
    similarity_score: Callable[[T, T], float],
    *,
    n_iter: int = 100,
) -> dict[T, np.ndarray]:
    elements = list(elements)
    layout_graph = nx.Graph()
    layout_graph.add_nodes_from(elements)
    for a_idx, a in enumerate(elements):
        for b in elements[a_idx:]:
            layout_graph.add_edge(a, b, attraction=similarity_score(a, b))
    return nx.spring_layout(layout_graph, weight="attraction", iterations=n_iter)


DefaultLayoutEngine = kamada_kawai_layout
