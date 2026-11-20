"""Tests for `postbound.opt._helpers` -- utilities to build plans and to read them back from JSON.

All tests are tier 0. Plans are built through the `QueryPlan` constructor and serialized with `util.jsonize.to_json`,
i.e. exactly the format that `read_query_plan_json` is meant to read back.

Currently this module only covers `read_query_plan_json`. The plan construction helpers (`make_query_plan`, ...) are
not covered yet.
"""

from __future__ import annotations

import json

import pytest

from postbound import util
from postbound._core import IntermediateOperator, JoinOperator, ScanOperator, TableReference
from postbound._qep import QueryPlan
from postbound.opt import read_query_plan_json

R = TableReference("r")
S = TableReference("s", "s1")


def scan(table: TableReference) -> QueryPlan:
    return QueryPlan("Seq Scan", operator=ScanOperator.SequentialScan, base_table=table)


def index_scan(table: TableReference, index: str) -> QueryPlan:
    return QueryPlan("Index Scan", operator=ScanOperator.IndexScan, base_table=table, index=index)


def round_trip(plan: QueryPlan) -> QueryPlan:
    encoded = util.to_json(plan)
    assert encoded is not None
    return read_query_plan_json(encoded)


# -- read_query_plan_json ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "plan",
    [
        scan(R),
        index_scan(R, "r_pkey"),
        QueryPlan("Hash Join", operator=JoinOperator.HashJoin, children=[scan(R), index_scan(S, "s_pkey")]),
        QueryPlan("Sort", operator=IntermediateOperator.Sort, children=[scan(R)]),
    ],
    ids=["seq-scan", "index-scan", "join", "intermediate"],
)
def test_read_query_plan_json_round_trips_a_plan(plan: QueryPlan) -> None:
    restored = round_trip(plan)

    assert restored == plan
    assert restored.operator == plan.operator


def test_read_query_plan_json_keeps_the_index_name() -> None:
    restored = round_trip(index_scan(R, "r_pkey"))

    assert restored.params.index == "r_pkey"


def test_read_query_plan_json_accepts_an_already_decoded_dict() -> None:
    plan = scan(R)
    encoded = util.to_json(plan)
    assert encoded is not None

    restored = read_query_plan_json(json.loads(encoded))

    assert restored == plan


def test_read_query_plan_json_rejects_an_operator_assignment_instead_of_an_operator() -> None:
    """Synthetic input: a plan node never serializes an operator *assignment*, but `read_operator_json` can produce
    one from a dict carrying a table, which is not a valid plan operator.
    """
    encoded_table = util.to_json(R)
    assert encoded_table is not None
    corrupted = {"node_type": "Seq Scan", "operator": {"operator": "Seq. Scan", "table": json.loads(encoded_table)}}

    with pytest.raises(ValueError, match="Expected physical operator"):
        read_query_plan_json(corrupted)


# -- regression tests --------------------------------------------------------------------------------------


def test_read_query_plan_json_accepts_nodes_without_a_physical_operator() -> None:
    """Regression guard for 743ba26: nodes such as Postgres' *Aggregate* have no PostBOUND-modeled operator, so their
    JSON contains ``"operator": null``. `read_operator_json` correctly returns ``None`` for it, but
    `read_query_plan_json` then failed its ``isinstance(operator, PhysicalOperator)`` check and raised a
    ``ValueError("Malformed query plan ...")`` for every plan that contained such a node.
    """
    plan = QueryPlan("Aggregate", children=[index_scan(R, "r_pkey")])

    restored = round_trip(plan)

    assert restored.operator is None
    assert restored == plan
