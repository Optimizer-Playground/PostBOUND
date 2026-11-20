"""Tests for `postbound._qep` -- `QueryPlan`, the central plan representation.

Plans are built directly through the `QueryPlan` constructor. The shape predicates under test only look at the
operators and the tree structure, so neither estimates nor captured EXPLAIN output are needed.

Parsing EXPLAIN output into plans is covered in `test_postgres_explain.py`, reading plans from their JSON
representation in `test_opt_helpers.py`.
"""

from __future__ import annotations

import pytest

from postbound._core import IntermediateOperator, JoinOperator, ScanOperator, TableReference
from postbound._qep import QueryPlan

R = TableReference("r")
S = TableReference("s")
T = TableReference("t")


def scan(table: TableReference) -> QueryPlan:
    return QueryPlan("Seq Scan", operator=ScanOperator.SequentialScan, base_table=table)


def index_scan(table: TableReference, index: str) -> QueryPlan:
    return QueryPlan("Index Scan", operator=ScanOperator.IndexScan, base_table=table, index=index)


def join(outer: QueryPlan, inner: QueryPlan) -> QueryPlan:
    return QueryPlan("Nested Loop", operator=JoinOperator.NestedLoopJoin, children=[outer, inner])


# -- equality and hashing ---------------------------------------------------------------------------------


def test_structurally_identical_plans_compare_and_hash_equal() -> None:
    plan = join(index_scan(R, "r_pkey"), scan(S))
    same_plan = join(index_scan(R, "r_pkey"), scan(S))

    assert plan == same_plan
    assert hash(plan) == hash(same_plan)


def test_plans_with_different_join_orders_are_not_equal() -> None:
    assert join(scan(R), scan(S)) != join(scan(S), scan(R))


def test_plans_with_different_node_types_are_not_equal() -> None:
    parallel_scan = QueryPlan("Parallel Seq Scan", operator=ScanOperator.SequentialScan, base_table=R)

    assert scan(R) != parallel_scan


# -- regression tests --------------------------------------------------------------------------------------


def test_is_right_deep_looks_through_an_intermediate_node() -> None:
    """Regression guard for cacf542: for an auxiliary node with a single input (Sort, Materialize, ...),
    `is_right_deep()` delegated to ``input_node.is_bushy()`` instead of ``input_node.is_right_deep()``. A sorted
    right-deep plan was therefore reported as not right-deep.
    """
    right_deep = join(scan(R), join(scan(S), scan(T)))
    sorted_plan = QueryPlan("Sort", operator=IntermediateOperator.Sort, children=[right_deep])

    assert sorted_plan.is_right_deep() is True


@pytest.mark.parametrize(
    ("plan", "other_plan"),
    [
        (index_scan(R, "r_pkey"), index_scan(R, "r_a_idx")),
        (join(index_scan(R, "r_pkey"), scan(S)), join(index_scan(R, "r_a_idx"), scan(S))),
    ],
    ids=["scan", "nested-in-a-join"],
)
def test_plans_scanning_different_indexes_are_not_equal(plan: QueryPlan, other_plan: QueryPlan) -> None:
    """Regression guard for 9715342: `QueryPlan.__eq__` and `__hash__` only looked at the node type, the base table and
    the children, but not at the index in the plan parameters. Two index scans of the same table over different indexes
    therefore compared equal, e.g. collapsing distinct plans into one when deduplicating them in a set.
    """
    assert plan != other_plan
    assert len({plan, other_plan}) == 2
