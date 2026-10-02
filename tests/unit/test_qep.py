"""Tests for `postbound._qep` -- `QueryPlan`, the central plan representation.

Plans are built directly through the `QueryPlan` constructor. The shape predicates under test only look at the
operators and the tree structure, so neither estimates nor captured EXPLAIN output are needed.

Currently this module only holds regression tests. Parsing EXPLAIN output into plans is covered in
`test_postgres_explain.py`.
"""

from __future__ import annotations

from postbound._core import IntermediateOperator, JoinOperator, ScanOperator, TableReference
from postbound._qep import QueryPlan

R = TableReference("r")
S = TableReference("s")
T = TableReference("t")


def scan(table: TableReference) -> QueryPlan:
    return QueryPlan("Seq Scan", operator=ScanOperator.SequentialScan, base_table=table)


def join(outer: QueryPlan, inner: QueryPlan) -> QueryPlan:
    return QueryPlan("Nested Loop", operator=JoinOperator.NestedLoopJoin, children=[outer, inner])


# -- regression tests --------------------------------------------------------------------------------------


def test_is_right_deep_looks_through_an_intermediate_node() -> None:
    """Regression guard for cacf542: for an auxiliary node with a single input (Sort, Materialize, ...),
    `is_right_deep()` delegated to ``input_node.is_bushy()`` instead of ``input_node.is_right_deep()``. A sorted
    right-deep plan was therefore reported as not right-deep.
    """
    right_deep = join(scan(R), join(scan(S), scan(T)))
    sorted_plan = QueryPlan("Sort", operator=IntermediateOperator.Sort, children=[right_deep])

    assert sorted_plan.is_right_deep() is True
