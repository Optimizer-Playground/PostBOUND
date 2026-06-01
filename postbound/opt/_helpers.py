from __future__ import annotations

import json
import math
import warnings
from typing import Literal, Optional, Union

from .. import Cardinality, parser
from .._core import (
    IntermediateOperator,
    JoinOperator,
    PhysicalOperator,
    ScanOperator,
    TableReference,
)
from .._hints import (
    DirectionalJoinOperatorAssignment,
    JoinOperatorAssignment,
    JoinTree,
    LogicalJoinTree,
    PhysicalOperatorAssignment,
    PlanParameterization,
    ScanOperatorAssignment,
    jointree_from_plan,
    operators_from_plan,
    parameters_from_plan,
)
from .._qep import PlanEstimates, PlanMeasures, PlanParams, QueryPlan, SortKey, Subplan
from ..qal import SqlQuery


def read_operator_json(
    json_data: dict | str,
) -> Optional[PhysicalOperator | ScanOperatorAssignment | JoinOperatorAssignment]:
    """Reads a physical operator assignment from a JSON dictionary.

    Parameters
    ----------
    json_data : dict | str
        Either the JSON dictionary, or a string encoding of the dictionary (which will be parsed by *json.loads*).

    Returns
    -------
    Optional[ScanOperators | JoinOperators | ScanOperatorAssignment | JoinOperatorAssignment]
        The parsed assignment. Whether it is a scan or join assignment is inferred from the JSON dictionary. If the input is
        empty or *None*, *None* is returned.
    """
    if not json_data:
        return None

    if isinstance(json_data, str):
        if json_data in {op.value for op in ScanOperator}:
            return ScanOperator(json_data)
        elif json_data in {op.value for op in JoinOperator}:
            return JoinOperator(json_data)
        elif json_data in {op.value for op in IntermediateOperator}:
            return IntermediateOperator(json_data)
        else:
            json_data = json.loads(json_data)

    parallel_workers = json_data.get("parallel_workers", math.nan)

    if "table" in json_data:
        parsed_table = parser.load_table_json(json_data["table"])
        scan_operator = ScanOperator(json_data["operator"])
        return ScanOperatorAssignment(scan_operator, parsed_table, parallel_workers)
    elif "join" not in json_data and not (
        "inner" in json_data and "outer" in json_data
    ):
        raise ValueError(
            f"Malformed operator JSON: either 'table' or 'join' must be given: '{json_data}'"
        )

    directional = json_data["directional"]
    join_operator = JoinOperator(json_data["operator"])
    if directional:
        inner = [parser.load_table_json(tab) for tab in json_data["inner"]]
        outer = [parser.load_table_json(tab) for tab in json_data["outer"]]
        return DirectionalJoinOperatorAssignment(
            join_operator, inner, outer, parallel_workers=parallel_workers
        )

    joined_tables = [parser.load_table_json(tab) for tab in json_data["join"]]
    return JoinOperatorAssignment(
        join_operator, joined_tables, parallel_workers=parallel_workers
    )


def read_operator_assignment_json(json_data: dict | str) -> PhysicalOperatorAssignment:
    """Loads an operator assignment from its JSON representation.

    Parameters
    ----------
    json_data : dict | str
        Either the JSON dictionary, or a string encoding of the dictionary (which will be parsed by *json.loads*).

    Returns
    -------
    PhysicalOperatorAssignment
        The assignment
    """
    json_data = json.loads(json_data) if isinstance(json_data, str) else json_data
    assignment = PhysicalOperatorAssignment()

    for hint in json_data.get("global_settings", []):
        enabled = hint["enabled"]
        match hint["kind"]:
            case "scan":
                assignment.global_settings[ScanOperator(hint["operator"])] = enabled
            case "join":
                assignment.global_settings[JoinOperator(hint["operator"])] = enabled
            case "intermediate":
                assignment.global_settings[IntermediateOperator(hint["operator"])] = (
                    enabled
                )
            case _:
                raise ValueError(f"Unknown operator kind: {hint['kind']}")

    for hint in json_data.get("scan_operators", []):
        parsed_table = parser.load_table_json(hint["table"])
        assignment.scan_operators[parsed_table] = ScanOperatorAssignment(
            ScanOperator(hint["operator"]), parsed_table
        )

    for hint in json_data.get("join_operators", []):
        parsed_tables = frozenset(
            parser.load_table_json(tab) for tab in hint["intermediate"]
        )
        assignment.join_operators[parsed_tables] = JoinOperatorAssignment(
            JoinOperator(hint["operator"]), parsed_tables
        )

    for hint in json_data.get("intermediate_operators", []):
        parsed_tables = frozenset(
            parser.load_table_json(tab) for tab in hint["intermediate"]
        )
        assignment.intermediate_operators[parsed_tables] = IntermediateOperator(
            hint["operator"]
        )

    return assignment


def read_plan_params_json(json_data: dict | str) -> PlanParameterization:
    """Loads a plan parameterization from its JSON representation.

    Parameters
    ----------
    json_data : dict | str
        Either the JSON dictionary, or a string encoding of the dictionary (which will be parsed by *json.loads*).

    Returns
    -------
    PlanParameterization
        The plan parameterization
    """
    json_data = json.loads(json_data) if isinstance(json_data, str) else json_data
    params = PlanParameterization()

    for hint in json_data.get("cardinalities", []):
        parsed_intermediate = parser.load_table_json(hint["intermediate"])
        parsed_card = Cardinality(hint["cardinality"])
        params.add_cardinality(parsed_intermediate, parsed_card)

    for hint in json_data.get("parallel_workers", []):
        parsed_intermediate = parser.load_table_json(hint["intermediate"])
        parsed_workers = hint["parallel_workers"]
        params.set_workers(parsed_intermediate, parsed_workers)

    return params


def update_plan(
    query_plan: QueryPlan,
    *,
    operators: Optional[PhysicalOperatorAssignment] = None,
    params: Optional[PlanParameterization] = None,
    simplify: bool = True,
) -> QueryPlan:
    """Assigns new operators and/or new estimates to a query plan, leaving the join order intact.

    Notice that this update method is not particularly smart and only operates on a per-node basis. This means that high-level
    functions that are composed of multiple operators might not be updated properly. For example, Postgres represents a hash
    join as a combination of a hash operator (which builds the actual hash table) and a follow-up hash join operator (which
    performs the probing). If the update changes the hash join to a different join, the hash operator will still exist, likely
    leading to an invalid query plan. To circumvent such problems, the query plan is by default simplified before processing.
    Simplification removes all auxiliary non-join and non-scan operators, thereby effectively only leaving those nodes with a
    corresponding operator. But, there is no free lunch and the simplification might also remove some other important
    operators, such as using hash-based or sort-based aggregation operators. Therefore, simplification can be disabled by
    setting the `simplify` parameter to *False*.

    Parameters
    ----------
    query_plan : QueryPlan
        The plan to update.
    operators : Optional[PhysicalOperatorAssignment], optional
        The new operators to use. This can be a partial assignment, in which case only the operators that are present in the
        new assignment are used and all others are left unchanged. If this parameter is not given, no operators are updated.
    params : Optional[PlanParameterization], optional
        The new parameters to use. This can be a partial assignment, in which case only the cardinalities/parallel workers in
        the new assignment are used and all others are left unchanged. If this parameter is not given, no parameters are
        updated.
    simplify : bool, optional
        Whether to simplify the query plan before updating it. For a detailed discussion, see the high-level documentatio of
        this method. Simplifications is enabled by default.

    Returns
    -------
    QueryPlan
        The updated query plan

    See Also
    --------
    QueryPlan.simplify
    """
    query_plan = query_plan.canonical() if simplify else query_plan

    updated_operator = (
        operators.get(query_plan.tables(), query_plan.operator)
        if operators
        else query_plan.operator
    )
    updated_card_est = (
        params.cardinalities.get(query_plan.tables(), query_plan.estimated_cardinality)
        if params
        else query_plan.estimated_cardinality
    )
    updated_workers = (
        params.parallel_workers.get(
            query_plan.tables(), query_plan.params.parallel_workers
        )
        if params
        else query_plan.params.parallel_workers
    )

    updated_params = PlanParams(
        **(query_plan.params.items() | {"parallel_workers": updated_workers})
    )
    updated_estimates = PlanEstimates(
        **(query_plan.estimates.items() | {"estimated_cardinality": updated_card_est})
    )
    updated_children = [
        update_plan(child, operators=operators, params=params)
        for child in query_plan.children
    ]

    return QueryPlan(
        query_plan.node_type,
        operator=updated_operator,
        children=updated_children,
        plan_params=updated_params,
        estimates=updated_estimates,
        measures=query_plan.measures,
        subplan=query_plan.subplan,
    )


NestedTableSequence = Union[
    tuple["NestedTableSequence", "NestedTableSequence"], TableReference
]
"""Type alias for a convenient format to notate join trees.

The notation is composed of nested lists. These lists can either contain more lists, or references to base tables.
Each list correponds to a branch in the join tree and the each table reference to a leaf.

Examples
--------

The nested sequence ``[[S, T], R]`` corresponds to the following tree:

::

    ⨝
    ├── ⨝
    │   ├── S
    │   └── T
    └── R

In this example, tables are simply denoted by their full name.
"""


def parse_nested_table_sequence(sequence: list[dict | list]) -> NestedTableSequence:
    """Loads the table sequence that is encoded by JSON-representation of the base tables.

    This is the inverse operation to writing a proper nested table sequence to a JSON object.

    Parameters
    ----------
    sequence : list[dict  |  list]
        The (parsed) JSON data. Each table is represented as a dictionary/nested JSON object.

    Returns
    -------
    NestedTableSequence
        The corresponding table sequence

    Raises
    ------
    TypeError
        If the list contains something other than more lists and dictionaries.
    """
    if isinstance(sequence, list):
        return [parse_nested_table_sequence(item) for item in sequence]
    elif isinstance(sequence, dict):
        table_name, alias = sequence["full_name"], sequence.get("alias", "")
        return TableReference(table_name, alias)
    else:
        raise TypeError(f"Unknown list element: {sequence}")


def _make_simple_plan(
    join_tree: JoinTree,
    *,
    scan_op: ScanOperator,
    join_op: JoinOperator,
    query: Optional[SqlQuery] = None,
    plan_params: Optional[PlanParameterization] = None,
) -> QueryPlan:
    """Handler function to create a query plan with default operators.

    (Estimated) cardinalities can still be customized accroding to the plan parameters. However, parallel workers are ignored.
    """
    tables = frozenset(join_tree.tables())
    if plan_params and plan_params.cardinalities.get(tables, None):
        cardinality = plan_params.cardinalities[tables]
    elif isinstance(join_tree, LogicalJoinTree):
        cardinality = join_tree.annotation
    else:
        cardinality = math.nan

    if join_tree.is_join():
        operator = join_op
        outer_plan = _make_simple_plan(
            join_tree.outer_child,
            scan_op=scan_op,
            join_op=join_op,
            query=query,
            plan_params=plan_params,
        )
        inner_plan = _make_simple_plan(
            join_tree.inner_child,
            scan_op=scan_op,
            join_op=join_op,
            query=query,
            plan_params=plan_params,
        )
        children = (outer_plan, inner_plan)
    else:
        operator = scan_op
        children = []

    if query is None:
        return QueryPlan(operator, children=children, estimated_cardinality=cardinality)

    predicates = query.predicates()
    filter_condition = (
        predicates.joins_between(
            join_tree.outer_child.tables(), join_tree.inner_child.tables()
        )
        if join_tree.is_join()
        else predicates.filters_for(join_tree.base_table)
    )
    return QueryPlan(
        operator,
        children=children,
        estimated_cardinality=cardinality,
        filter_condition=filter_condition,
    )


def _make_custom_plan(
    join_tree: JoinTree,
    *,
    physical_ops: PhysicalOperatorAssignment,
    query: Optional[SqlQuery] = None,
    plan_params: Optional[PlanParameterization] = None,
    fallback_scan_op: Optional[ScanOperator] = None,
    fallback_join_op: Optional[JoinOperator] = None,
) -> QueryPlan:
    """Handler function to create a query plan with a dynamic assignment of physical operators.

    If an operator is not contained in the assignment, the fallback operators are used. If these are also not available,
    this is an error.

    In addition to the operators, the estimated cardinalities as well as the parallel workers can be customized using the plan
    parameters. As a fallback, cardinalities from the join tree annotations are used.
    """
    tables = frozenset(join_tree.tables())
    if plan_params and plan_params.cardinalities.get(tables, None):
        cardinality = plan_params.cardinalities[tables]
    elif isinstance(join_tree, LogicalJoinTree):
        cardinality = join_tree.annotation
    else:
        cardinality = math.nan

    par_workers = (
        plan_params.parallel_workers.get(tables, None) if plan_params else None
    )

    operator = physical_ops.get(tables)
    if not operator and len(tables) == 1:
        operator = fallback_scan_op
    elif not operator and len(tables) > 1:
        operator = fallback_join_op
    if not operator:
        raise ValueError("No operator assignment found for join: " + str(tables))

    if join_tree.is_join():
        outer_plan = _make_simple_plan(
            join_tree.outer_child, physical_ops=physical_ops, plan_params=plan_params
        )
        inner_plan = _make_simple_plan(
            join_tree.inner_child, physical_ops=physical_ops, plan_params=plan_params
        )
        children = (outer_plan, inner_plan)
    else:
        children = []

    if query is None:
        plan = QueryPlan(
            operator,
            children=children,
            estimated_cardinality=cardinality,
            parallel_workers=par_workers,
        )
    else:
        predicates = query.predicates()
        filter_condition = (
            predicates.joins_between(
                join_tree.outer_child.tables(), join_tree.inner_child.tables()
            )
            if join_tree.is_join()
            else predicates.filters_for(join_tree.base_table)
        )
        plan = QueryPlan(
            operator,
            children=children,
            estimated_cardinality=cardinality,
            filter_condition=filter_condition,
            parallel_workers=par_workers,
        )

    intermediate_op = physical_ops.intermediate_operators.get(frozenset(plan.tables()))
    if not intermediate_op:
        return plan
    if intermediate_op in {IntermediateOperator.Sort, IntermediateOperator.Memoize}:
        warnings.warn(
            "Ignoring intermediate operator for sort/memoize. These require additional information to be inserted."
        )
        return plan

    plan = QueryPlan(intermediate_op, children=plan, estimated_cardinality=cardinality)
    return plan


def to_query_plan(
    join_tree: JoinTree,
    *,
    query: Optional[SqlQuery] = None,
    physical_ops: Optional[PhysicalOperatorAssignment] = None,
    plan_params: Optional[PlanParameterization] = None,
    scan_op: Optional[ScanOperator] = None,
    join_op: Optional[JoinOperator] = None,
) -> QueryPlan:
    """Creates a query plan from a join tree.

    This function operates in two different modes: physical operators can either be assigned to each node of the join tree
    individually using the `physical_ops`, or the same operator can be assigned to all scans and joins using the `scan_op` and
    `join_op` parameters. If the former approach is used, fallback/default operators can be provided to compensate missing
    operators in the assignment.
    Furthermore, `plan_params` can be used to inject custom cardinality estimates and parallel workers to the nodes.

    If the supplied `join_tree` is a `LogicalJoinTree`, its cardinality estimates are used as a fallback if no estimate from
    the plan parameters is available.

    Notice that the resulting query plan does not contain any DB-specific features. For example, assigning a hash join to
    an intermediate does not also insert a hash operator, as is done by some database systems.

    Parameters
    ----------
    join_tree : JoinTree
        The join order to use for the query plan. If this is a logical join tree, the cardinality estimates can be added to the
        query plan if no more specific estimates are available through the `plan_params`.
    query : Optional[SqlQuery], optional
        The query that is computed by the query plan. If this is supplied, it is used to compute join predicates and filters
        that can be computed at the various nodes of the query plan.
    physical_ops : Optional[PhysicalOperatorAssignment], optional
        The physical operators that should be used for individual nodes of the join tree. If this is supplied, the `scan_op`
        and `join_op` parameters are used as a fallback if no assignment exists for a specific intermediate. Notice that
        parallel workers contained in the operator assignments are never used since this information should be made available
        through the `plan_params`.
    plan_params : Optional[PlanParameterization], optional
        Optional cardinality estimates and parallelization info for the nodes of the join tree. If this is not supplied,
        cardinality estimates are inferred from a logical join tree or left as NaN otherwise.
    scan_op : Optional[ScanOperator], optional
        The operator to assign to all scans in the query plan. If no `physical_ops` are given, this parameter has to be
        specified. If `physical_ops` are indeed given, this parameter is used as a fallback if no assignment exists for a
        specific scan.
    join_op : Optional[JoinOperator], optional
        The operator to assign to all joins in the query plan. If no `physical_ops` are given, this parameter has to be
        specified. If `physical_ops` are indeed given, this parameter is used as a fallback if no assignment exists for a
        specific join.

    Returns
    -------
    QueryPlan
        The resulting query plan
    """
    if physical_ops:
        return _make_custom_plan(
            join_tree,
            physical_ops=physical_ops,
            query=query,
            plan_params=plan_params,
            fallback_scan_op=scan_op,
            fallback_join_op=join_op,
        )
    elif scan_op is not None and join_op is not None:
        return _make_simple_plan(
            join_tree,
            scan_op=scan_op,
            join_op=join_op,
            query=query,
            plan_params=plan_params,
        )
    else:
        raise ValueError(
            "Either operator assignment or default operators must be provided"
        )


def read_query_plan_json(json_data: dict | str) -> QueryPlan:
    """Reads a query plan from its JSON representation.

    Parameters
    ----------
    json_data : dict | str
        Either the JSON dictionary, or a string encoding of the dictionary (which will be parsed by *json.loads*).

    Returns
    -------
    QueryPlan
        The corresponding query plan
    """
    json_data = json.loads(json_data) if isinstance(json_data, str) else json_data
    node_type: str = json_data["node_type"]
    operator: PhysicalOperator = read_operator_json(json_data.get("operator"))
    children = [read_query_plan_json(child) for child in json_data.get("children", [])]

    params_json: dict = json_data.get("plan_params", {})
    base_table_json: dict | None = params_json.get("base_table")
    base_table = parser.load_table_json(base_table_json) if base_table_json else None

    predicate_json: dict | None = params_json.get("filter_predicate")
    filter_predicate = (
        parser.load_predicate_json(predicate_json) if predicate_json else None
    )

    sort_keys: list[SortKey] = []
    for sort_key_json in params_json.get("sort_keys", []):
        sort_column = [
            parser.load_expression_json(col)
            for col in sort_key_json.get("equivalence_class", [])
        ]
        ascending = sort_key_json["ascending"]
        sort_keys.append(SortKey.of(sort_column, ascending))

    index = params_json.get("index", "")
    additional_params = {
        key: value
        for key, value in params_json.items()
        if key not in {"base_table", "filter_predicate", "sort_keys", "index"}
    }

    plan_params = PlanParams(
        base_table=base_table,
        filter_predicate=filter_predicate,
        sort_keys=sort_keys,
        index=index,
        **additional_params,
    )

    estimates_json: dict = json_data.get("estimates", {})
    cardinality = estimates_json.get("cardinality", math.nan)
    cost = estimates_json.get("cost", math.nan)
    additional_estimates = {
        key: value
        for key, value in estimates_json.items()
        if key not in {"cardinality", "cost"}
    }
    estimates = PlanEstimates(
        cardinality=cardinality, cost=cost, **additional_estimates
    )

    measures_json: dict = json_data.get("measures", {})
    cardinality = measures_json.get("cardinality", math.nan)
    exec_time = measures_json.get("execution_time", math.nan)
    cache_hits = measures_json.get("cache_hits")
    cache_misses = measures_json.get("cache_misses")
    additional_measures = {
        key: value
        for key, value in measures_json.items()
        if key not in {"cardinality", "execution_time", "cache_hits", "cache_misses"}
    }
    measures = PlanMeasures(
        cardinality=cardinality,
        execution_time=exec_time,
        cache_hits=cache_hits,
        cache_misses=cache_misses,
        **additional_measures,
    )

    subplan_json: dict = json_data.get("subplan", {})
    if subplan_json:
        subplan_root = parser.parse_query(subplan_json["root"])
        subplan_target = subplan_json.get("target_name", "")
        subplan = Subplan(root=subplan_root, target_name=subplan_target)
    else:
        subplan = None

    return QueryPlan(
        node_type,
        operator=operator,
        children=children,
        plan_params=plan_params,
        estimates=estimates,
        measures=measures,
        subplan=subplan,
    )


def jointree_from_sequence(sequence: NestedTableSequence) -> JoinTree[None]:
    """Creates a raw join tree from a table sequence.

    The table sequence encodes the join structure using nested lists, see `NestedTableSequence` for details.
    """
    if isinstance(sequence, TableReference):
        return JoinTree(base_table=sequence)

    outer, inner = sequence
    return JoinTree.join(jointree_from_sequence(outer), jointree_from_sequence(inner))


def read_jointree_json(json_data: dict | str) -> JoinTree:
    """Loads a jointree from its JSON representations.

    Parameters
    ----------
    json_data : dict | str
        Either the JSON dictionary, or a string encoding of the dictionary (which will be parsed by *json.loads*).

    Returns
    -------
    JoinTree
        The corresponding join tree
    """
    json_data = json.loads(json_data) if isinstance(json_data, str) else json_data

    annotation = json_data.get("annotation", None)

    table_json = json_data.get("table", None)
    if table_json:
        base_table = parser.load_table_json(table_json)
        return JoinTree.scan(base_table, annotation=annotation)

    outer_child = read_jointree_json(json_data["outer"])
    inner_child = read_jointree_json(json_data["inner"])
    return JoinTree.join(outer_child, inner_child, annotation=annotation)


def explode_query_plan(
    query_plan: QueryPlan, *, card_source: Literal["estimated", "actual"] = "estimated"
) -> tuple[LogicalJoinTree, PhysicalOperatorAssignment, PlanParameterization]:
    """Extracts the join tree, physical operators, and plan parameters from a query plan.

    Parameters
    ----------
    query_plan : QueryPlan
        The query plan to extract the information from
    card_source : Literal["estimated", "actual"], optional
        Which cardinalities to use in the join tree and the plan parameters. Defaults to the estimated cardinalities.

    Returns
    -------
    tuple[LogicalJoinTree, PhysicalOperatorAssignment, PlanParameterization]
        The different components of the query plan
    """
    return (
        jointree_from_plan(query_plan, card_source=card_source),
        operators_from_plan(query_plan),
        parameters_from_plan(query_plan, target_cardinality=card_source),
    )
