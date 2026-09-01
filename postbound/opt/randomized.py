"""Provides "optimization" strategies that generate random query plans."""

from __future__ import annotations

import random
from collections.abc import Generator, Iterable
from typing import Literal, Optional

import networkx as nx

from .._core import JoinOperator, PhysicalOperator, ScanOperator, TableReference
from .._hints import (
    JoinOperatorAssignment,
    JoinTree,
    PhysicalOperatorAssignment,
    ScanOperatorAssignment,
)
from .._qep import QueryPlan
from .._stages import (
    CompleteOptimizationAlgorithm,
    JoinOrderOptimization,
    OptimizationPreCheck,
    PhysicalOperatorSelection,
)
from ..db import Database, DatabasePool
from ..qal import SelectStatement
from ..util import nx as nx_utils
from ..validation import CompoundCheck, CrossProductPreCheck, SupportedHintCheck
from ._helpers import to_query_plan


def _merge_nodes(
    query: SelectStatement,
    start: JoinTree | TableReference,
    end: JoinTree | TableReference,
) -> JoinTree:
    """Provides a join tree that combines two specific trees or tables.

    This is a shortcut method to merge arbitrary tables or trees without having to check whether a table-based or tree-based
    merge has to be performed.

    Parameters
    ----------
    query : SqlQuery
        The query to which the (partial) join trees belong. This parameter is necessary to generate the correct metadata for
        the join tree
    start : JoinTree | TableReference
        The first tree to merge. If this is a base table, it will be treated as a join tree of just a scan of that table.
    end : JoinTree | TableReference
        The second tree to merge. If this is a base table, it will be treated as a join tree of just a scan of that table.

    Returns
    -------
    JoinTree
        A join tree combining the input trees. The `start` node will be the left node of the tree and the `end` node will be
        the right node.
    """
    start = JoinTree.create_scan(start) if isinstance(start, TableReference) else start
    end = JoinTree.create_scan(end) if isinstance(end, TableReference) else end
    return start.join_with(end)


def _sample_join_graph(
    query: SelectStatement,
    join_graph: nx.Graph,
    *,
    base_table: Optional[TableReference] = None,
) -> JoinTree:
    """Generates a random join order for the given join graph.

    Parameters
    ----------
    query : SqlQuery
        The query to which the join graph belongs. This parameter is necessary to generate the correct metadata for the join
        tree.
    join_graph : nx.Graph
        The join graph that should be "optimized". This should be a pure join graph as provided by the *qal* module.
    base_table : Optional[TableReference], optional
            An optional table that should always be joined first. If unspecified, base tables are selected at random.

    Returns
    -------
    JoinTree
        A random join order for the given join graph.

    Warnings
    --------
    This algorithm does not work for join graphs that contain cross products (i.e. multiple connected components).

    Notes
    -----
    This algorithm works in an iterative manner: At each step, two connected nodes are selected. For these nodes, a join is
    simulated. This is done by generating a join tree for the nodes and merging them into a single node for the join tree. The
    iteration stops as soon as the graph only consists of a single node. This node represents the join tree for the entire
    graph. Depending on the order in which the edges are selected, a different join tree is produced.
    """
    if base_table is not None:
        candidate_edges: list[TableReference] = list(join_graph.adj[base_table])
        initial_join_partner = random.choice(candidate_edges)
        right, left = (
            (base_table, initial_join_partner) if random.random() < 0.5 else (initial_join_partner, base_table)
        )
        join_tree = _merge_nodes(query, right, left)
        join_graph = nx.contracted_nodes(join_graph, base_table, initial_join_partner, self_loops=False)
        join_graph = nx.relabel_nodes(join_graph, {base_table: join_tree})

    while len(join_graph.nodes) > 1:
        join_predicates = list(join_graph.edges)
        next_edge = random.choice(join_predicates)
        start_node, target_node = next_edge
        right, left = (start_node, target_node) if random.random() < 0.5 else (target_node, start_node)
        join_tree = _merge_nodes(query, right, left)

        join_graph = nx.contracted_nodes(join_graph, start_node, target_node, self_loops=False)
        join_graph = nx.relabel_nodes(join_graph, {start_node: join_tree})

    final_node: JoinTree = list(join_graph.nodes)[0]
    return final_node


class RandomJoinOrderGenerator:
    """Utility service to produce randomized join orders for an input query.

    The service produces a generator that in turn provides the join orders. This is done in the `random_join_orders_for`
    method. The provided join orders can include linear, as well as bushy, join orders. The structure can be customized during
    service creation.

    Parameters
    ----------
    eliminate_duplicates : bool, optional
        Whether repeated calls to the generator should be guaranteed to provide different join orders. Defaults to ``False``,
        which permits duplicates.
    tree_structure : Literal[bushy, left-deep, right-deep], optional
        The kind of join orders that are generated by the service. "bushy" allows join orders with arbitrary branches to be
        generated (including linear join orders). "right-deep" and "left-deep" restrict the join orders to the respective
        linear trees. Defaults to "bushy".

    Warnings
    --------
    For now, the underlying algorithm is limited to queries without cross-products.
    """

    def __init__(
        self,
        eliminate_duplicates: bool = False,
        *,
        tree_structure: Literal["bushy", "right-deep", "left-deep"] = "bushy",
    ) -> None:
        self._eliminate_duplicates = eliminate_duplicates
        self._tree_structure = tree_structure

    def random_join_orders_for(
        self, query: SelectStatement, *, base_table: Optional[TableReference] = None
    ) -> Generator[JoinTree, None, None]:
        """Provides a generator that successively provides join orders at random.

        Parameters
        ----------
        query : SqlQuery
            The query for which the join orders should be generated
        base_table : Optional[TableReference], optional
            An optional table that should always be joined first. If unspecified, base tables are selected at random.

        Yields
        ------
        Generator[JoinTree, None, None]
            A generator that produces random join orders for the input query. The structure of these join orders depends on the
            service configuration. Consult the class-level documentation for more details. Depeding on the
            `eliminate_duplicates` attribute, the join orders are guaranteed to be unique.

        Raises
        ------
        ValueError
            If the query contains cross products.

        Warnings
        --------
        For now, the underlying algorithm is limited to queries without cross-products.
        """
        join_graph = query.predicates().join_graph()
        if len(join_graph.nodes) == 0:
            return
        elif len(join_graph.nodes) == 1:
            base_table = list(join_graph.nodes)[0]
            join_tree = JoinTree.create_scan(base_table)
            while True:
                yield join_tree
        elif not nx.is_connected(join_graph):
            raise ValueError("Cross products are not yet supported for random join order generation!")

        join_order_generator = (
            self._bushy_join_orders(query, join_graph, base_table=base_table)
            if self._tree_structure == "bushy"
            else self._linear_join_orders(query, join_graph, base_table=base_table)
        )

        join_order_hashes = set()
        for current_join_order in join_order_generator:
            if self._eliminate_duplicates:
                current_hash = hash(current_join_order)
                if current_hash in join_order_hashes:
                    continue
                else:
                    join_order_hashes.add(current_hash)

            yield current_join_order

    def _linear_join_orders(
        self,
        query: SelectStatement,
        join_graph: nx.Graph,
        *,
        base_table: Optional[TableReference] = None,
    ) -> Generator[JoinTree, None, None]:
        """Handler method to generate left-deep or right-deep join orders.

        The specific kind of join orders is inferred based on the `_tree_structure` attribute.

        Parameters
        ----------
        query : SqlQuery
            The query to "optimize"
        join_graph : nx.Graph
            The join graph of the query to optimize
        base_table : Optional[TableReference], optional
            An optional table that should always be joined first. If unspecified, the base join is selected at random.

        Yields
        ------
        Generator[JoinTree, None, None]
            A generator that produces all possible join orders for the input query.
        """
        direction = "inner" if self._tree_structure == "left-deep" else "outer"
        while True:
            join_path = [node for node in nx_utils.nx_random_walk(join_graph, starting_node=base_table)]
            join_tree = JoinTree()
            for table in join_path:
                join_tree = join_tree.join_with(table, partner_direction=direction)
            yield join_tree

    def _bushy_join_orders(
        self,
        query: SelectStatement,
        join_graph: nx.Graph,
        *,
        base_table: Optional[TableReference] = None,
    ) -> Generator[JoinTree, None, None]:
        """Handler method to generate bushy join orders.

        Notice that linear join orders are considered a subclass of bushy join trees. Hence, bushy join orders may occasionally
        be linear.

        Parameters
        ----------
        query : SqlQuery
            The query to "optimize"
        join_graph : nx.Graph
            The join graph of the query to optimize
        base_table : Optional[TableReference], optional
            An optional table that should always be joined first. If unspecified, base tables are selected at random.

        Yields
        ------
        Generator[JoinTree, None, None]
            A generator that produces all possible join orders for the input query.
        """
        while True:
            yield _sample_join_graph(query, join_graph, base_table=base_table)


class RandomJoinOrderOptimizer(JoinOrderOptimization):
    """Optimization stage that produces a randomized join order.

    This class acts as a wrapper around a `RandomJoinOrderGenerator` for the join optimization interface. The setup of the
    generator can be customized during creation of the optimizer. Consult the documentation of the generator for details.

    Parameters
    ----------
    generator_args : Optional[dict], optional
        Arguments to customize the generator operation. All parameters are forwarded to its ``__init__`` method.

    See Also
    --------
    RandomJoinOrderGenerator
    """

    def __init__(self, *, generator_args: Optional[dict] = None) -> None:
        super().__init__()
        generator_args = generator_args if generator_args is not None else {}
        self._generator = RandomJoinOrderGenerator(**generator_args)

    def optimize_join_order(self, query: SelectStatement) -> Optional[JoinTree]:
        return next(self._generator.random_join_orders_for(query))

    def describe(self) -> dict:
        return {
            "name": "random",
            "structure": self._generator._tree_structure,
            "eliminates_duplicates": self._generator._eliminate_duplicates,
        }

    def pre_check(self) -> OptimizationPreCheck:
        return CrossProductPreCheck()


class RandomOperatorGenerator:
    """Utility service to generate random assignments of physical operators for a join order.

    The service produces a generator that in turn provides the operator assignments. This is done in the
    `random_operator_assignments_for` method. The precise properties of the generated assignments depends on the configuration
    of this service. It can be set up to only use a subset of the available operators or to exclude operators for scans or
    joins completely. By default, the service uses all operators that are supported by the target database system.

    Parameters
    ----------
    scan_operators : Optional[Iterable[ScanOperators]], optional
        The scan operators that can be used in the query plans. If this is ``None`` or empty, all scans supported by the
        `database` are used. Likewise, if the iterable contains an operator that is not supported by the database, it is
        exlcuded from generation.
    join_operators : Optional[Iterable[JoinOperators]], optional
        The join operators that can be used in the query plans. If this is ``None`` or empty, all joins supported by the
        `database` are used. Likewise, if the iterable contains an operator that is not supported by the database, it is
        exlcuded from generation.
    include_scans : bool, optional
        Whether the assignment should contain scan operators at all. By default, this is enabled. However, if scans are
        disabled, this overwrites any supplied operators in the `scan_operators` parameter.
    include_joins : bool, optional
        Whether the assignment should contain join operators at all. By default, this is enabled. However, if joins are
        disabled, this overwrites any supplied operators in the `join_operators` parameter.
    eliminate_duplicates : bool, optional
        Whether repeated calls to the generator should be guaranteed to provide different operator assignments. Defaults to
        ``False``, which permits duplicates.
    database : Optional[db.Database], optional
        The database that should execute the queries in the end. The database connection is necessary to determine the
        operators that are actually supported by the system. If this parameter is omitted, it is inferred from the
        `DatabasePool`.

    Raises
    ------
    ValueError
        If both scans and joins are disabled
    """

    def __init__(
        self,
        scan_operators: Optional[Iterable[ScanOperator]] = None,
        join_operators: Optional[Iterable[JoinOperator]] = None,
        *,
        include_scans: bool = True,
        include_joins: bool = True,
        eliminate_duplicates: bool = False,
        database: Optional[Database] = None,
    ) -> None:
        if not include_joins and not include_scans:
            raise ValueError("Cannot exclude both join hints and scan hints")
        self._db = database if database is not None else DatabasePool.get_instance().current_database()
        self._eliminate_duplicates = eliminate_duplicates
        self._include_scans = include_scans
        self._include_joins = include_joins
        allowed_scan_ops = scan_operators if scan_operators else ScanOperator
        allowed_join_ops = join_operators if join_operators else JoinOperator
        self.allowed_scan_ops = frozenset(
            scan_op for scan_op in allowed_scan_ops if self._db.hinting().supports_hint(scan_op)
        )
        self.allowed_join_ops = frozenset(
            join_op for join_op in allowed_join_ops if self._db.hinting().supports_hint(join_op)
        )

    def random_operator_assignments_for(
        self, query: SelectStatement, join_order: JoinTree
    ) -> Generator[PhysicalOperatorAssignment, None, None]:
        """Produces a generator for random operator assignments of the allowed operators.

        The precise structure of the operator assignments depends on the service configuration. Take a look at the class
        documentation for details.

        Parameters
        ----------
        query : SqlQuery
            The query to "optimize"
        join_order : jointree.JoinTree
            The join sequence to use. This contains all required tables to be scanned and joins to be performed.

        Yields
        ------
        Generator[PhysicalOperatorAssignment, None, None]
            A generator producing random operator assignments. The assignments will not contain any cost estimates, nor
            will they specify join directions or parallization data.
        """
        allowed_scans = list(self.allowed_scan_ops) if self._include_scans else []
        allowed_joins = list(self.allowed_join_ops) if self._include_joins else []
        assignment_hashes = set()

        while True:
            current_assignment = PhysicalOperatorAssignment()

            if self._include_joins:
                for join in join_order.iterjoins():
                    selected_operator = random.choice(allowed_joins)
                    current_assignment.set_join_operator(JoinOperatorAssignment(selected_operator, join.tables()))

            if self._include_scans:
                for table in join_order.tables():
                    selected_operator = random.choice(allowed_scans)
                    current_assignment.set_scan_operator(ScanOperatorAssignment(selected_operator, table))

            if self._eliminate_duplicates:
                current_hash = hash(current_assignment)
                if current_hash in assignment_hashes:
                    continue
                else:
                    assignment_hashes.add(current_hash)

            yield current_assignment

    def necessary_hints(self) -> frozenset[PhysicalOperator]:
        """Provides all hints that a database system must support in order for the generator to work properly.

        Returns
        -------
        frozenset[PhysicalOperator]
            The required operator hints
        """
        return self.allowed_join_ops | self.allowed_scan_ops


class RandomOperatorOptimizer(PhysicalOperatorSelection):
    """Optimization stage that produces a randomized operator assignment.

    This class acts as a wrapper around a `RandomOperatorGenerator` for the operator optimization interface. The setup of the
    generator can be customized during creation of the optimizer. Consult the documentation of the generator for details.

    Parameters
    ----------
    generator_args : Optional[dict], optional
        Arguments to customize the generator operation. All parameters are forwarded to its ``__init__`` method.

    See Also
    --------
    RandomOperatorGenerator
    """

    def __init__(self, *, generator_args: Optional[dict] = None) -> None:
        super().__init__()
        generator_args = generator_args if generator_args is not None else {}
        self._generator = RandomOperatorGenerator(**generator_args)

    def select_physical_operators(self, query: SelectStatement, join_order: Optional[JoinTree]) -> PhysicalOperatorAssignment:
        if join_order is None:
            raise ValueError("Join order is required for operator selection.")
        return next(self._generator.random_operator_assignments_for(query, join_order))

    def describe(self) -> dict:
        allowed_scans = self._generator.allowed_scan_ops if self._generator._include_scans else []
        allowed_joins = self._generator.allowed_join_ops if self._generator._include_joins else []
        return {
            "name": "random",
            "allowed_operators": {"scans": allowed_scans, "joins": allowed_joins},
            "eliminates_duplicates": self._generator._eliminate_duplicates,
        }

    def pre_check(self) -> OptimizationPreCheck:
        return SupportedHintCheck(self._generator.necessary_hints())


class RandomPlanGenerator:
    """Utility service to provide random exection plans for a query.

    This service combines the `RandomJoinOrderGenerator` and `RandomOperatorGenerator` into a single high-level
    service. Therefore, it underlies the same restrictions as these two services. The produced generator can be accessed via
    the `random_plans_for` method.


    Parameters
    ----------
    eliminate_duplicates : bool, optional
        Whether repeated calls to the generator should be guaranteed to provide different plans. Defaults to ``False``, which
        permits duplicates. This setting can be overwritten on a per-generator basis by specifying it in the dedicated
        generator arguments.
    join_order_args : Optional[dict], optional
        Configuration for the `RandomJoinOrderGenerator`. This is forwarded to the service's ``__init__`` method.
    operator_args : Optional[dict], optional
        Configuration for the `RandomOperatorGenerator`. This is forwarded to the service's ``__init__`` method.
    database : Optional[db.Database], optional
        The database for the operator selection. This parameter can also be specified in the operator generator arguments, or
        even left completely unspecified.

    See Also
    --------
    RandomJoinOrderGenerator
    RandomOperatorGenerator
    """

    def __init__(
        self,
        *,
        eliminate_duplicates: bool = False,
        join_order_args: Optional[dict] = None,
        operator_args: Optional[dict] = None,
        database: Optional[Database] = None,
    ) -> None:
        join_order_args = dict(join_order_args) if join_order_args is not None else {}
        operator_args = dict(operator_args) if operator_args is not None else {}
        if "database" not in operator_args:
            operator_args["database"] = database

        self._eliminate_duplicates = eliminate_duplicates
        self._join_order_generator = RandomJoinOrderGenerator(**join_order_args)
        self._operator_generator = RandomOperatorGenerator(**operator_args)

    def random_plans_for(self, query: SelectStatement) -> Generator[QueryPlan, None, None]:
        """Produces a generator for random query plans of an input query.

        The structure of the provided plans can be restricted by configuring the underlying services. Consult the class-level
        documentation for details.

        Parameters
        ----------
        query : SqlQuery
            The query to "optimize"

        Yields
        ------
        Generator[QueryPlan, None, None]
            A generator producing random query plans
        """
        join_order_generator = self._join_order_generator.random_join_orders_for(query)
        plan_hashes = set()
        while True:
            join_order = next(join_order_generator)
            operator_generator = self._operator_generator.random_operator_assignments_for(query, join_order)
            physical_operators = next(operator_generator)

            query_plan = to_query_plan(join_order, physical_ops=physical_operators)
            if self._eliminate_duplicates:
                current_plan_hash = hash(query_plan)
                if current_plan_hash in plan_hashes:
                    continue
                else:
                    plan_hashes.add(current_plan_hash)

            yield query_plan


class RandomPlanOptimizer(CompleteOptimizationAlgorithm):
    """Optimization stage that produces a random query plan.

    This class acts as a wrapper around a `RandomPlanGenerator` and passes all its arguments to that service.

    Parameters
    ----------
    join_order_args : Optional[dict], optional
        Configuration for the `RandomJoinOrderGenerator`. This is forwarded to the service's ``__init__`` method.
    operator_args : Optional[dict], optional
        Configuration for the `RandomOperatorGenerator`. This is forwarded to the service's ``__init__`` method.
    database : Optional[db.Database], optional
        The database for the operator selection. This parameter can also be specified in the operator generator arguments, or
        even left completely unspecified.

    See Also
    --------
    RandomPlanGenerator

    Notes
    -----
    It is not necessary to request duplicate elimination for any of the generators, since the underlying Python generator
    objects cannot be re-used between multiple optimization passes for the same input query. Therefore, it is not possible to
    enforce duplicate elimination for a join order or operator assignment.

    Because multiple calls to optimizer with the same input query should not influence each other, the optimizer also does not
    provide its own duplicate elimination.
    """

    def __init__(
        self,
        *,
        join_order_args: Optional[dict] = None,
        operator_args: Optional[dict] = None,
        database: Optional[Database] = None,
    ) -> None:
        super().__init__()
        self._generator = RandomPlanGenerator(
            join_order_args=join_order_args,
            operator_args=operator_args,
            database=database,
        )

    def optimize_query(self, query: SelectStatement) -> QueryPlan:
        return next(self._generator.random_plans_for(query))

    def describe(self) -> dict:
        scan_ops = (
            self._generator._operator_generator.allowed_scan_ops
            if self._generator._operator_generator._include_scans
            else []
        )
        join_ops = (
            self._generator._operator_generator.allowed_join_ops
            if self._generator._operator_generator._include_joins
            else []
        )
        return {
            "name": "random",
            "join_order": {"tree_structure": self._generator._join_order_generator._tree_structure},
            "physical_operators": {"scans": scan_ops, "joins": join_ops},
        }

    def pre_check(self) -> OptimizationPreCheck:
        return CompoundCheck(
            [CrossProductPreCheck(), SupportedHintCheck(self._generator._operator_generator.necessary_hints())]
        )
