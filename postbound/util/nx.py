"""Provides graph-centric algorithms based on NetworkX [nx]_.

References
----------

.. [nx] Aric A. Hagberg, Daniel A. Schult and Pieter J. Swart, "Exploring network structure, dynamics, and function using
        NetworkX", in Proceedings of the 7th Python in Science Conference (SciPy2008), Gäel Varoquaux, Travis Vaught, and
        Jarrod Millman (Eds), (Pasadena, CA USA), pp. 11-15, Aug 2008
"""

from __future__ import annotations

import dataclasses
import random
import typing
from collections.abc import (
    Callable,
    Collection,
    Generator,
    Iterator,
    Sequence,
)

import networkx as nx

from .collections import Queue

NodeType = typing.TypeVar("NodeType")
"""Generic type to model the specific nodes contained in a NetworkX graph."""


def nx_sinks(graph: nx.DiGraph) -> Collection[NodeType]:
    """Determines all sink nodes in a directed graph.

    A sink is a node with no outgoing edges.

    Parameters
    ----------
    graph : nx.DiGraph
        The graph to check

    Returns
    -------
    Collection[NodeType]
        All sink nodes. Can be an empty collection.
    """
    return [n for n in graph.nodes if graph.out_degree(n) == 0]


def nx_sources(graph: nx.DiGraph) -> Collection[NodeType]:
    """Determines all source nodes in a directed graph.

    A source is a node with no incoming edges.

    Parameters
    ----------
    graph : nx.DiGraph
        The graph to check

    Returns
    -------
    Collection[NodeType]
        All source nodes. Can be an empty collection.
    """
    return [n for n in graph.nodes if graph.in_degree(n) == 0]


def nx_filter_nodes(graph: nx.Graph, predicate: Callable[[NodeType, dict], bool]) -> Collection[tuple[NodeType, dict]]:
    return [(n, d) for n, d in graph.nodes.data() if predicate(n, d)]


def nx_random_walk(graph: nx.Graph, *, starting_node: NodeType | None = None) -> Generator[NodeType, None, None]:
    """A modified random walk implementation for NetworkX graphs.

    A random walk starts at any of the nodes of the graph. At each iteration, a neighboring node is selected and moved to.
    Afterwards, the iteration continues with that node.

    Our implementation uses the following modifications: after each stop, the walk may jump to a node that is connected to
    one of the visited nodes. This node does not necessarily have to be connected to the current node. Secondly, if the
    graph contains multiple connected components, the walk will first explore one component before jumping to the next
    one.

    The walk finishes when all nodes have been explored.

    Parameters
    ----------
    graph : nx.Graph
        The graph to walk over
    starting_node : Optional[NodeType], optional
        The node where the walk starts. If unspecified, a random node is selected.

    Yields
    ------
    Generator[NodeType, None, None]
        The nodes in the order in which they have been moved to.
    """
    # TODO: could be refactored to use the GraphWalk class instead
    shell_nodes = set()
    visited_nodes = set()

    total_n_nodes = len(graph.nodes)

    current_node = random.choice(list(graph.nodes)) if starting_node is None else starting_node
    visited_nodes.add(current_node)
    yield current_node

    while len(visited_nodes) < total_n_nodes:
        shell_nodes |= set(n for n in graph.adj[current_node] if n not in visited_nodes)
        if not shell_nodes:
            # we have multiple connected components and need to jump into the other component
            current_node = random.choice([n for n in graph.nodes if n not in visited_nodes])
            visited_nodes.add(current_node)
            yield current_node
            continue

        current_node = random.choice(list(shell_nodes))
        shell_nodes.remove(current_node)
        visited_nodes.add(current_node)
        yield current_node


def nx_bfs_tree(
    graph: nx.Graph,
    start_node: NodeType,
    condition: Callable[[NodeType, dict], bool],
    *,
    node_order: Callable[[NodeType, dict], int] | None = None,
) -> Generator[tuple[NodeType, dict], None, None]:
    """Traverses a specific graph in breadth-first manner, yielding its nodes along the way.

    The traversal starts at a specific start node. During the traversal all nodes that match a condition are provided. If no
    more nodes are found or the condition cannot be satisfied for any more nodes, traversal terminates. Notice that there is
    no "early stopping": if a parent node fails the condition check, its children are still explored.

    Parameters
    ----------
    graph : nx.Graph
        The graph to explore
    start_node : NodeType
        The node where the exploration starts. This node will never be yielded.
    condition : Callable[[NodeType, dict], bool]
        A condition that is satisfied by all nodes that should be yielded
    node_order : Callable[[NodeType, dict], int] | None, optional
        The sequence in which child nodes should be explored. This function receives the child node as well as the edge from
        its parent as arguments and produces a numerical position value as output (lower values indicate earlier yielding).
        If unspecified, this produces the nodes in an arbitrary order.

    Yields
    ------
    Generator[tuple[NodeType, dict], None, None]
        The node along with their edge data from the parent.

    See Also
    --------

    .. NetworkX documentation on usage and definition of edge data:
       https://networkx.org/documentation/stable/reference/introduction.html#nodes-and-edges
    """
    shell_nodes = Queue([(node, edge) for node, edge in graph.adj[start_node].items()])
    visited_nodes = {start_node}
    while shell_nodes:
        current_node, current_edge = shell_nodes.pop()  # ty: ignore[not-iterable] -- guarded by the `while shell_nodes` above
        visited_nodes.add(current_node)
        if condition(current_node, current_edge):
            neighbor_nodes = [
                (node, edge) for node, edge in graph.adj[current_node].items() if node not in visited_nodes
            ]
            if node_order:
                sorted(
                    neighbor_nodes,
                    key=lambda neighbor: node_order(neighbor[0], neighbor[1]),
                )
            shell_nodes.extend(neighbor_nodes)
            yield current_node, current_edge


@dataclasses.dataclass
class GraphWalk(typing.Generic[NodeType]):
    """A graph walk models a traversal of some graph.

    Each walk begins at a specific *start node* and then follows a *path* along other nodes and edges.

    Notice that depending on the specific use-case the path might deviate from a normal walk. More specifically, two
    special cases might occur:

    1. the edge data can be ``None``. This indicates that the walk jumped to a different node without using an edge. For
       example, this might happen if the walk moved to a different connected component of the graph.
    2. the next node in the walk might not be connected to the current node in the path, but to some node that has
       already been visited instead. This is especially the case for so-called *frontier walks* which can be computed
       using the `nx_frontier_walks` method.

    The walk can be iterated over by its nodes and nodes can be checked for containment in the walk. Length calculation is
    also supported.

    Attributes
    ----------
    start_node : NodeType
        The origin of the traversal
    path : Sequence[tuple[NodeType, Optional[dict]]]
        The nodes that have been visited during the traversal, in the order in which they were explored. The dictionary stores
        the NetworkX edge data of the edge that has been used to move to the node. This may be ``None`` if the node was
        "jumped to".
    """

    start_node: NodeType
    path: Sequence[tuple[NodeType, dict | None]] = dataclasses.field(default_factory=list)

    def nodes(self) -> Sequence[NodeType]:
        """Provides all nodes that are visited by this walk, in the sequence in which they are visited.

        Returns
        -------
        Sequence[NodeType]
            The nodes
        """
        return [self.start_node] + [node[0] for node in self.path]

    def final_node(self) -> NodeType:
        """Provides the very last node that was visited by this walk.

        Returns
        -------
        NodeType
            The last node. This can be the `start_node` if the path is empty.
        """
        return self.start_node if not self.path else self.path[-1][0]

    def expand(self, next_node: NodeType, edge_data: dict | None = None) -> GraphWalk:
        """Creates a new walk by prolonging the current one with one more edge at the end.

        Parameters
        ----------
        next_node : NodeType
            The node to move to from the final node of the current graph.
        edge_data : Optional[dict], optional
            The NetworkX edge data for the traversal. Can be ``None`` if the new node is being jumped to.

        Returns
        -------
        GraphWalk
            The resulting larger walk. The original walk is not modified in any way.
        """
        return GraphWalk(self.start_node, [*list(self.path), (next_node, edge_data)])

    def nodes_hash(self) -> int:
        """Provides a hash value only based on the nodes sequence, not the selected predicates.

        Returns
        -------
        int
            The computed hash value
        """
        return hash(tuple(self.nodes()))

    def __len__(self) -> int:
        return 1 + len(self.path)

    def __iter__(self) -> Iterator[NodeType]:
        return self.nodes().__iter__()

    def __contains__(self, other: object) -> bool:
        return other in self.nodes()

    def __hash__(self) -> int:
        return hash((self.start_node, tuple(self.path)))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self)) and self.start_node == other.start_node and self.path == other.path

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return " -> ".join(str(node) for node in self.nodes())


def _walk_frontier(
    graph: nx.Graph, current_walk: GraphWalk, current_frontier: set[NodeType]
) -> Generator[GraphWalk, None, None]:
    """Worker method to recursively expand graph traversals to candidate nodes.

    This method expands a specific walk by considering all possible traversals to candidate/frontier nodes. Jumps are included
    if the graph contains multiple connected components. The frontier is composed of all nodes that are adjacent to one of the
    nodes that has already been visited.

    Only paths of unique nodes are considered - if a node has already been visited, it will not be visited again.

    Parameters
    ----------
    graph : nx.Graph
        The graph to traverse
    current_walk : GraphWalk
        The path that was already selected
    current_frontier : set[NodeType]
        All nodes that can be moved to next. This datastructure is mutated after each traversal to include the new (yet
        unexplored) nodes that are adjacent to the selected next node.

    Yields
    ------
    Generator[GraphWalk, None, None]
        All unique walks over the complete graph
    """
    available_edges = []
    for frontier_node in current_frontier:
        current_edges = graph.adj[frontier_node]
        current_edges = [
            (target_node, edge_data)
            for target_node, edge_data in current_edges.items()
            if target_node not in current_walk and target_node not in current_frontier
        ]
        available_edges.extend(current_edges)

    if not available_edges and len(current_walk) < len(graph):
        jump_nodes = [node for node in graph.nodes if node not in current_frontier]
        for jump_node in jump_nodes:
            yield from _walk_frontier(graph, current_walk.expand(jump_node), current_frontier | {jump_node})
    elif not available_edges:
        yield current_walk
    else:
        for target_node, edge_data in available_edges:
            yield from _walk_frontier(
                graph,
                current_walk.expand(target_node, edge_data),
                current_frontier | {target_node},
            )


def nx_frontier_walks(graph: nx.Graph) -> Generator[GraphWalk, None, None]:
    """Provides all possible frontier walks over a specific graph.

    A *frontier walk* is a generalized version of a normal walk over a graph: Whereas a normal walk traverses the edges in the
    graph to move from node to node in a local fashion (i.e. only based on the edges of the current node), a frontier walk
    remembers all the nodes that have already been visited. This is called the *frontier* of the current walk. To find the next
    node, any edge from any of the nodes in the frontier can be selected.

    Notice that the frontier walk also remembers nodes that have already been visited and prevents them from being visited
    again.

    Our implementation augments this procedure by also allowing jumps to other partitions in the graph. This will happen if all
    nodes in the current connected component have been visited, but more unexplored nodes remain.

    Parameters
    ----------
    graph : nx.Graph
        The graph to traverse

    Yields
    ------
    Generator[GraphWalk, None, None]
        All frontier walks over the graph

    Notes
    -----

    Notice that this method already distinguishes between paths if they differ in the traversed edges, even if the sequence of
    nodes is the same.

    For example, consider this fully-connected graph:

    ::

         a
        /  \\
        b - c


    The frontier walks produced by this function will include the sequence *a* -> *b* -> *c* twice (among many other
    sequences):
    Once by traversing the edge *a* -> *c* to reach *c*, and once by traversing the edge *b* -> *c* to reach *c* again.
    """
    for node in graph.nodes:
        yield from _walk_frontier(graph, GraphWalk(node), current_frontier={node})
