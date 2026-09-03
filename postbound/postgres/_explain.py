from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from typing import Any, Literal, Optional, get_args

from .._core import Cardinality, IntermediateOperator, JoinOperator, ScanOperator, TableReference
from .._qep import QueryPlan, SortKey

PostgresExplainJoinNodes = {
    "Nested Loop": JoinOperator.NestedLoopJoin,
    "Hash Join": JoinOperator.HashJoin,
    "Merge Join": JoinOperator.SortMergeJoin,
}
"""A mapping from Postgres EXPLAIN node names to the corresponding join operators."""

PostgresExplainScanNodes = {
    "Seq Scan": ScanOperator.SequentialScan,
    "Index Scan": ScanOperator.IndexScan,
    "Index Only Scan": ScanOperator.IndexOnlyScan,
    "Bitmap Heap Scan": ScanOperator.BitmapScan,
}
"""A mapping from Postgres EXPLAIN node names to the corresponding scan operators."""

PostgresExplainIntermediateNodes = {
    "Materialize": IntermediateOperator.Materialize,
    "Memoize": IntermediateOperator.Memoize,
    "Sort": IntermediateOperator.Sort,
}
"""A mapping from Postgres EXPLAIN node names to the corresponding intermediate operators."""


NodeType = Literal[
    "Result",
    "ProjectSet",
    "ModifyTable",
    "Append",
    "Merge Append",
    "Recursive Union",
    "BitmapAnd",
    "BitmapOr",
    "Nested Loop",
    "Merge Join",
    "Hash Join",
    "Seq Scan",
    "Sample Scan",
    "Gather",
    "Gather Merge",
    "Index Scan",
    "Index Only Scan",
    "Bitmap Index Scan",
    "Bitmap Heap Scan",
    "Tid Scan",
    "Tid Range Scan",
    "Subquery Scan",
    "Function Scan",
    "Table Function Scan",
    "Values Scan",
    "CTE Scan",
    "Named Tuplestore Scan",
    "WorkTable Scan",
    "Foreign Scan",
    "Custom Scan",
    "Materialize",
    "Memoize",
    "Sort",
    "Incremental Sort",
    "Group",
    "Aggregate",
    "WindowAgg",
    "Unique",
    "SetOp",
    "LockRows",
    "Limit",
    "Hash",
]
"""All different nodes that can be created by Postgres.

This has been extracted directly from ExplainNode() in explain.c from the Postgres source code.
"""


class PostgresExplainNode:
    """Simplified model of a plan node as provided by Postgres' *EXPLAIN* output in JSON format.

    Generally speaking, a node stores all the information about the plan node that we currently care about. This is mostly
    focused on optimizer statistics, along with some additional data. Explain nodes form a hierarchichal structure with each
    node containing an arbitrary number of child nodes. Notice that this model is very loose in the sense that no constraints
    are enforced and no sanity checking is performed. For example, this means that nodes can contain more than two children
    even though this can never happen in a real *EXPLAIN* plan. Similarly, the correspondence between filter predicates and
    the node typse (e.g. join filter for a join node) is not checked.

    All relevant data from the explain node is exposed as attributes on the objects. Even though these are mutable, they should
    be thought of as read-only data objects.

    Parameters
    ----------
    explain_data : dict
        The JSON data of the current explain node. This is parsed and prepared as part of the *__init__* method.

    Attributes
    ----------
    node_type : NodeType | None, default None
        The node type. This should never be empty or *None*, even though it is technically allowed.
    cost : float, default NaN
        The optimizer's cost estimation for this node. This includes the cost of all child nodes as well. This should normally
        not be *NaN*, even though it is technically allowed.
    cardinality_estimate : float, default NaN
        The optimizer's estimation of the number of tuples that will be *produced* by this operator. This should normally not
        be *NaN*, even though it is technically allowed.
    execution_time : float, default NaN
        For *EXPLAIN ANALYZE* plans, this is the actual total execution time of the node in seconds. For pure *EXPLAIN*
        plans, this is *NaN*
    true_cardinality : float, default NaN
        For *EXPLAIN ANALYZE* plans, this is the average of the number of tuples that were actually produced for each loop of
        the node. For pure *EXPLAIN* plans, this is *NaN*
    loops : int, default 1
        For *EXPLAIN ANALYZE* plans, this is the number of times the operator was invoked. The number of invocations can mean
        a number of different things: for parallel operators, this normally matches the number of parallel workers. For scans,
        this matches the number of times a new tuple was requested (e.g. for an index nested-loop join the number of loops of
        the index scan part indicates how many times the index was probed).
    relation_name : str | None, default None
        The name of the relation/table that is processed by this node. This should be defined on scan nodes, but could also
        be present on other nodes.
    relation_alias : str | None, default None
        The alias of the relation/table under which the relation was accessed in th equery plan. See `relation_name`.
    index_name : str | None, default None
        The name of the index that was probed. This should be defined on index scans and index-only scans, but could also be
        present on other nodes.
    filter_condition : str | None, default None
        A post-processing filter that is applied to all rows emitted by this operator. This is most important for scan
        operations with an attached filter predicate, but can also be present on some joins.
    index_condition : str | None, default None
        The condition that is used to locate the matching tuples in an index scan or index-only scan
    join_filter : str | None, default None
        The condition that is used to determine matching tuples in a join
    hash_condition : str | None, default None
        The condition that is used to determine matching tuples in a hash join
    recheck_condition : str | None, default None
        For lossy bitmap scans or bitmap scans based on lossy indexes, this is post-processing check for whether the produced
        tuples actually match the filter condition
    parent_relationship : str | None, default None
        Describes the role that this node plays in relation to its parent. Common values are *inner* which denotes that
        this is the inner child of a join and *outer* which denotes the opposite.
    parallel_workers : int | float, default NaN
        For parallel operators in *EXPLAIN ANALYZE* plans, this is the actual number of worker processes that were started.
        Notice that in total there is one additional worker. This process takes care of spawning the other workers and
        managing them, but can also take part in the input processing.
    sort_keys : list[str]
        The columns that are used to sort the tuples that are produced by this node. This is most important for sort nodes,
        but can also be present on other nodes.
    shared_blocks_read : float, default NaN
        For *EXPLAIN ANALYZE* plans with *BUFFERS* enabled, this is the number of blocks/pages that where retrieved from
        disk while executing this node, including the reads of all its child nodes.
    shared_blocks_buffered : float, default NaN
        For *EXPLAIN ANALYZE* plans with *BUFFERS* enabled, this is the number of blocks/pages that where retrieved from
        the shared buffer while executing this node, including the hits of all its child nodes.
    temp_blocks_read : float, default NaN
        For *EXPLAIN ANALYZE* blocks with *BUFFERS* enabled, this is the number of short-term data structures (e.g. hash
        tables, sorts) that where read by this node, including reads of all its child nodes.
    temp_blocks_written : float, default NaN
        For *EXPLAIN ANALYZE* blocks with *BUFFERS* enabled, this is the number of short-term data structures (e.g. hash
        tables, sorts) that where written by this node, including writes of all its child nodes.
    plan_width : float, default NaN
        The average width of the tuples that are produced by this node.
    children : list[PostgresExplainNode]
        All child / input nodes for the current node
    """

    @staticmethod
    def all_node_types() -> frozenset[NodeType]:
        """All node types that are currently recognized by PostBOUND."""
        return frozenset(get_args(NodeType))

    def __init__(self, explain_data: dict) -> None:
        self.node_type: NodeType = explain_data["Node Type"]

        self.cost: float = explain_data.get("Total Cost", math.nan)
        self.cardinality_estimate: float = explain_data.get("Plan Rows", math.nan)
        self.execution_time: float = explain_data.get("Actual Total Time", math.nan) / 1000

        # true_cardinality is accessed as a property to add a warning for BitmapAnd/Or nodes
        self._true_card: float = explain_data.get("Actual Rows", math.nan)

        self.loops: float = explain_data.get("Actual Loops", 1)

        self.relation_name: str | None = explain_data.get("Relation Name", None)
        self.relation_alias: str | None = explain_data.get("Alias", None)
        self.index_name: str | None = explain_data.get("Index Name", None)
        self.subplan_name: str | None = explain_data.get("Subplan Name", None)
        self.cte_name: str | None = explain_data.get("CTE Name", None)

        self.filter_condition: str | None = explain_data.get("Filter", None)
        self.index_condition: str | None = explain_data.get("Index Cond", None)
        self.join_filter: str | None = explain_data.get("Join Filter", None)
        self.hash_condition: str | None = explain_data.get("Hash Cond", None)
        self.recheck_condition: str | None = explain_data.get("Recheck Cond", None)
        self.parent_relationship: str | None = explain_data.get("Parent Relationship", None)
        self.launched_workers: int = explain_data.get("Workers Launched", 0)
        self.planned_workers: int = explain_data.get("Workers Planned", 0)
        self.sort_keys: str = explain_data.get("Sort Key", "")

        self.shared_blocks_read: int = explain_data.get("Shared Read Blocks", math.nan)
        self.shared_blocks_cached: int = explain_data.get("Shared Hit Blocks", math.nan)
        self.temp_blocks_read: int = explain_data.get("Temp Read Blocks", math.nan)
        self.temp_blocks_written: int = explain_data.get("Temp Written Blocks", math.nan)
        self.plan_width: int = explain_data.get("Plan Width", math.nan)
        self.children = [PostgresExplainNode(child) for child in explain_data.get("Plans", [])]

        self.explain_data: dict = explain_data
        self._hash_val = hash(
            (
                self.node_type,
                self.relation_name,
                self.relation_alias,
                self.index_name,
                self.subplan_name,
                self.cte_name,
                self.filter_condition,
                self.index_condition,
                self.join_filter,
                self.hash_condition,
                self.recheck_condition,
                self.parent_relationship,
                self.launched_workers,
                tuple(self.children),
            )
        )

    @property
    def true_cardinality(self) -> float:
        if self.node_type in {"BitmapAnd", "BitmapOr"}:
            # For BitmapAnd/BitmapOr nodes, the actual number of rows is always 0.
            # This is due to limitations in the Postgres implementation.
            warnings.warn(
                "Postgres does not report the actual number of rows for bitmap nodes correctly. Returning NaN.",
                stacklevel=2,
            )
            return math.nan
        return self._true_card

    def is_scan(self) -> bool:
        """Checks, whether the current node corresponds to a scan node.

        For Bitmap index scans, which are multi-level scan operators, this is true for the heap scan part that takes care of
        actually reading the tuples according to the bitmap provided by the bitmap index scan operators.

        Returns
        -------
        bool
            Whether the node is a scan node
        """
        return self.node_type in PostgresExplainScanNodes

    def is_join(self) -> bool:
        """Checks, whether the current node corresponds to a join node.

        Returns
        -------
        bool
            Whether the node is a join node
        """
        return self.node_type in PostgresExplainJoinNodes

    def is_analyze(self) -> bool:
        """Checks, whether this *EXPLAIN* plan is an *EXPLAIN ANALYZE* plan or a pure *EXPLAIN* plan.

        The analyze variant does not only obtain the plan, but actually executes it. This enables the comparison of the
        optimizer's estimates to the actual values. If a plan is an *EXPLAIN ANALYZE* plan, some attributes of this node
        receive actual values. These include `execution_time`, `true_cardinality`, `loops` and `parallel_workers`.


        Returns
        -------
        bool
            Whether the node represents part of an *EXPLAIN ANALYZE* plan
        """
        return not math.isnan(self.execution_time)

    def filter_conditions(self) -> dict[str, str]:
        """Collects all filter conditions that are defined on this node

        Returns
        -------
        dict[str, str]
            A dictionary mapping the type of filter condition (e.g. index condition or join filter) to the actual filter value.
        """
        conditions: dict[str, str] = {}
        if self.filter_condition is not None:
            conditions["Filter"] = self.filter_condition
        if self.index_condition is not None:
            conditions["Index Cond"] = self.index_condition
        if self.join_filter is not None:
            conditions["Join Filter"] = self.join_filter
        if self.hash_condition is not None:
            conditions["Hash Cond"] = self.hash_condition
        if self.recheck_condition is not None:
            conditions["Recheck Cond"] = self.recheck_condition
        return conditions

    def inner_outer_children(self) -> Sequence[PostgresExplainNode]:
        """Provides the children of this node in a sequence of inner, outer if applicable.

        For all nodes where this structure is not meaningful (e.g. intermediate nodes that operate on a single relation or
        scan nodes), the child nodes are returned as-is (e.g. as a list of a single child or an empty list).

        Returns
        -------
        Sequence[PostgresExplainNode]
            The children of the current node in a unified format
        """
        if len(self.children) < 2:
            return self.children
        assert len(self.children) == 2

        first_child, second_child = self.children
        inner_child = first_child if first_child.parent_relationship == "Inner" else second_child
        outer_child = first_child if second_child == inner_child else second_child
        return (inner_child, outer_child)

    def parse_table(self) -> Optional[TableReference]:
        """Provides the table that is processed by this node.

        Returns
        -------
        Optional[TableReference]
            The table being scanned. For non-scan nodes, or nodes where no table can be inferred, *None* will be returned.
        """
        if not self.relation_name:
            return None
        alias = (
            self.relation_alias if self.relation_alias is not None and self.relation_alias != self.relation_name else ""
        )
        return TableReference(self.relation_name, alias)

    def as_qep(self) -> QueryPlan:
        """Transforms the postgres-specific plan to a standardized `QueryPlan` instance.

        Notice that this transformation is lossy since not all information from the Postgres plan can be represented in query
        execution plan instances. Furthermore, this transformation can be problematic for complicated queries that use
        special Postgres features. Most importantly, for queries involving subqueries, special node types and parent
        relationships can be contained in the plan, that cannot be represented by other parts of PostBOUND. If this method
        and the resulting query execution plans should be used on complex workloads, it is advisable to check the plans twice
        before continuing.

        Returns
        -------
        QueryPlan
            The equivalent query execution plan for this node

        Raises
        ------
        ValueError
            If the node contains more than two children.
        """
        return self._generate_qep()

    def _generate_qep(self, *, card_adjustment: float = 1.0) -> QueryPlan:
        child_nodes = []
        inner_child, outer_child, subplan_child = None, None, None

        # planned workers is 0 for sequential execution, otherwise it contains the number of additional workers
        # if we already have a card adjustment, we are alrady in a parallel subplan so we re-use the existing adjustment
        # otherwise, the total adjustment is planned_workers + 1 to account for the main process
        child_adjustment = card_adjustment if card_adjustment > 1 else (self.planned_workers + 1)

        for child in self.children:
            parent_rel = child.parent_relationship
            qep_child = child._generate_qep(card_adjustment=child_adjustment)

            match parent_rel:
                case "Inner":
                    inner_child = qep_child
                case "Outer":
                    outer_child = qep_child
                case "SubPlan" | "InitPlan" | "Subquery":
                    subplan_child = qep_child
                case "Member":
                    child_nodes.append(qep_child)
                case _:
                    raise ValueError(f"Unknown parent relationship '{parent_rel}' for child {child}")

        if inner_child and outer_child:
            child_nodes = [outer_child, inner_child] + child_nodes
        elif outer_child:
            child_nodes.insert(0, outer_child)
        elif inner_child:
            child_nodes.insert(0, inner_child)

        table = self.parse_table()
        subplan_name = self.subplan_name or self.cte_name

        true_card = self.true_cardinality * self.loops
        estimated_card = self.cardinality_estimate * card_adjustment

        if self.is_scan():
            operator = PostgresExplainScanNodes.get(self.node_type, None)
        elif self.is_join():
            operator = PostgresExplainJoinNodes.get(self.node_type, None)
        else:
            operator = PostgresExplainIntermediateNodes.get(self.node_type, None)

        sort_keys = self._parse_sort_keys() if self.sort_keys else self._infer_sorting_from_children()
        shared_hits = None if math.isnan(self.shared_blocks_cached) else self.shared_blocks_cached
        shared_misses = None if math.isnan(self.shared_blocks_read) else self.shared_blocks_read

        if self.launched_workers > 0:
            par_workers = self.launched_workers
        elif self.planned_workers > 0:
            par_workers = self.planned_workers
        else:
            par_workers = None

        plan = QueryPlan(
            self.node_type,
            base_table=table,
            operator=operator,
            children=child_nodes,
            parallel_workers=par_workers,
            index=self.index_name,
            sort_keys=sort_keys,
            estimated_cost=self.cost,
            estimated_cardinality=Cardinality(estimated_card),
            actual_cardinality=Cardinality(true_card),
            execution_time=self.execution_time,
            cache_hits=shared_hits,
            cache_misses=shared_misses,
            subplan_root=subplan_child,
            subplan_name=subplan_name,
        )

        return plan

    def inspect(self, *, _indentation: int = 0) -> str:
        """Provides a pretty string representation of the *EXPLAIN* sub-plan that can be printed.

        Parameters
        ----------
        _indentation : int, optional
            This parameter is internal to the method and ensures that the correct indentation is used for the child nodes
            of the plan. When inspecting the root node, this value is set to its default value of `0`.

        Returns
        -------
        str
            A string representation of the *EXPLAIN* sub-plan.
        """
        if self.parent_relationship in ("InitPlan", "SubPlan"):
            padding = " " * (max(_indentation - 2, 0))
            cte_name = self.subplan_name if self.subplan_name else ""
            own_inspection = [f"{padding}{self.parent_relationship}: {cte_name}"]
        else:
            own_inspection = []
        padding = " " * _indentation
        prefix = f"{padding}<- " if padding else ""
        own_inspection += [prefix + str(self)]
        child_inspections = [child.inspect(_indentation=_indentation + 2) for child in self.children]
        return "\n".join(own_inspection + child_inspections)

    def _infer_sorting_from_children(self) -> list[SortKey]:
        # TODO: Postgres is a cruel mistress. Even if output is sorted, it might not be marked as such.
        # For example, in index scans, this is implictly encoded in the index condition, somethimes even nested in other
        # expressions. We first need a reliable way to parse the expressions into a PostBOUND-compatible format.
        # See _parse_sort_keys for a start.
        return []

    def _parse_sort_keys(self) -> list[SortKey]:
        # TODO implementation
        return []

    def __hash__(self) -> int:
        return self._hash_val

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, type(self))
            and self.node_type == other.node_type
            and self.relation_name == other.relation_name
            and self.relation_alias == other.relation_alias
            and self.children == other.children
        )

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        analyze_content = (
            f" (actual time={self.execution_time}s rows={self.true_cardinality} loops={self.loops})"
            if self.is_analyze()
            else ""
        )
        explain_content = f"(cost={self.cost} rows={self.cardinality_estimate})"
        conditions = " ".join(f"{condition}: {value}" for condition, value in self.filter_conditions().items())
        conditions = " " + conditions if conditions else ""
        if self.is_scan():
            tab = self.parse_table()
            assert tab is not None
            scan_info = f" on {tab.identifier()}"
        elif self.cte_name:
            scan_info = f" on {self.cte_name}"
        else:
            scan_info = ""
        return self.node_type + scan_info + explain_content + analyze_content + conditions


class PostgresExplainPlan:
    """Models an entire *EXPLAIN* plan produced by Postgres

    In contrast to `PostgresExplainNode`, this includes additional parameters (planning time and execution time) for the entire
    plan, rather than just portions of it.

    This class supports all methods that are specified on the general `QueryPlan` and returns the correct data for its actual
    plan.

    Parameters
    ----------
    explain_data : dict | list[dict]
        The JSON data of the entire explain plan. This is parsed and prepared as part of the *__init__* method.


    Attributes
    ----------
    planning_time : float
        The time in seconds that the optimizer spent to build the plan
    execution_time : float
        The time in seconds the query execution engine needed to calculate the result set of the query. This does not account
        for network time to transmit the result set.
    query_plan : PostgresExplainNode
        The actual plan
    """

    def __init__(self, explain_data: dict | list[dict]) -> None:
        self.explain_data = explain_data[0] if isinstance(explain_data, list) else explain_data
        if not (isinstance(self.explain_data, dict) and "Plan" in self.explain_data):
            raise ValueError(f"Invalid explain data: missing 'Plan' key: {explain_data}")

        self.planning_time: float = self.explain_data.get("Planning Time", math.nan) / 1000
        self.execution_time: float = self.explain_data.get("Execution Time", math.nan) / 1000
        self.query_plan = PostgresExplainNode(self.explain_data["Plan"])
        self._normalized_plan = self.query_plan.as_qep()

    @property
    def root(self) -> PostgresExplainNode:
        """Gets the root node of the actual query plan."""
        return self.query_plan

    def is_analyze(self) -> bool:
        """Checks, whether this *EXPLAIN* plan is an *EXPLAIN ANALYZE* plan or a pure *EXPLAIN* plan.

        The analyze variant does not only obtain the plan, but actually executes it. This enables the comparison of the
        optimizer's estimates to the actual values. If a plan is an *EXPLAIN ANALYZE* plan, some attributes of this node
        receive actual values. These include `execution_time`, `true_cardinality`, `loops` and `parallel_workers`.


        Returns
        -------
        bool
            Whether the plan represents an *EXPLAIN ANALYZE* plan
        """
        return self.query_plan.is_analyze()

    def as_qep(self) -> QueryPlan:
        """Provides the actual explain plan as a normalized query execution plan instance

        For notes on pecularities of this method, take a look at the *See Also* section

        Returns
        -------
        QueryPlan
            The query execution plan

        See Also
        --------
        PostgresExplainNode.as_qep
        """
        return self._normalized_plan

    def inspect(self) -> str:
        """Provides a pretty string representation of the actual plan.

        Returns
        -------
        str
            A string representation of the plan

        See Also
        --------
        PostgresExplainNode.inspect
        """
        return self.query_plan.inspect()

    def __json__(self) -> Any:
        return self.explain_data

    def __getattribute__(self, name: str) -> Any:
        # All methods that are not defined on the Postgres plan delegate to the default DB plan
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            root_plan_node = object.__getattribute__(self, "query_plan")
            try:
                return root_plan_node.__getattribute__(name)
            except AttributeError:
                normalized_plan = object.__getattribute__(self, "_normalized_plan")
                return normalized_plan.__getattribute__(name)

    def __hash__(self) -> int:
        return hash(self.query_plan)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self)) and self.query_plan == other.query_plan

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        if self.is_analyze():
            prefix = f"EXPLAIN ANALYZE (plan time={self.planning_time}, exec time={self.execution_time})"
        else:
            prefix = "EXPLAIN"

        return f"{prefix} root: {self.query_plan}"
