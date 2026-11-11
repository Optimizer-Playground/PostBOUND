"""Tests for `postbound._hints` -- the partial optimization decisions handed to the hint layer.

Query plans are mostly built directly through the `QueryPlan` constructor: they are PostBOUND's own model and the
functions under test only read their estimates and measurements. The exception are behaviours that depend on the
*shape* of plans a real system produces (e.g. Postgres bitmap scans, whose index lookups carry no base table): these
replay verbatim captured ``EXPLAIN (FORMAT JSON)`` output through `PostgresPlan`, see `test_postgres_explain.py` for
the reasoning behind captured fixtures. Nothing is executed at test time, so everything stays in tier 0.

Currently this module only holds regression tests. The hint generation of the individual backends is covered in
`test_postgres_hints.py`.
"""

from __future__ import annotations

import warnings
from typing import Literal

import pytest

from postbound._core import Cardinality, ScanOperator, TableReference
from postbound._hints import HintType, parameters_from_plan
from postbound._qep import QueryPlan
from postbound.postgres._explain import PostgresPlan

R = TableReference("r")

ESTIMATED = Cardinality(10)
ACTUAL = Cardinality(99)
SCAN_WITH_ESTIMATE_AND_MEASUREMENT = QueryPlan(
    "Seq Scan",
    operator=ScanOperator.SequentialScan,
    base_table=R,
    estimated_cardinality=ESTIMATED,
    actual_cardinality=ACTUAL,
)


# Captured against the `stats` workload (`.psycopg_connection_stats`) on PostgreSQL 18.6 with a default configuration:
#
# EXPLAIN (FORMAT JSON, ANALYZE)
# SELECT p.id, u.displayname FROM posts p JOIN users u ON p.owneruserid = u.id
# WHERE p.owneruserid = 5 OR p.lasteditoruserid = 5
#
# The Bitmap Heap Scan is annotated with the posts table, but its BitmapOr child and the two Bitmap Index Scans below it
# only reference indexes -- they have no base table at all. Their estimates (and the measured rows of the index scans)
# are nevertheless valid cardinalities.
BITMAP_HEAP_SCAN_JOIN_ANALYZE = [
    {
        "Plan": {
            "Node Type": "Nested Loop",
            "Parallel Aware": False,
            "Async Capable": False,
            "Join Type": "Inner",
            "Startup Cost": 9.75,
            "Total Cost": 446.75,
            "Plan Rows": 109,
            "Plan Width": 13,
            "Actual Startup Time": 5.794,
            "Actual Total Time": 11.802,
            "Actual Rows": 123.0,
            "Actual Loops": 1,
            "Disabled": False,
            "Inner Unique": True,
            "Shared Hit Blocks": 361,
            "Shared Read Blocks": 108,
            "Shared Dirtied Blocks": 0,
            "Shared Written Blocks": 0,
            "Local Hit Blocks": 0,
            "Local Read Blocks": 0,
            "Local Dirtied Blocks": 0,
            "Local Written Blocks": 0,
            "Temp Read Blocks": 0,
            "Temp Written Blocks": 0,
            "Plans": [
                {
                    "Node Type": "Bitmap Heap Scan",
                    "Parent Relationship": "Outer",
                    "Parallel Aware": False,
                    "Async Capable": False,
                    "Relation Name": "posts",
                    "Alias": "p",
                    "Startup Cost": 9.46,
                    "Total Cost": 415.12,
                    "Plan Rows": 109,
                    "Plan Width": 8,
                    "Actual Startup Time": 5.45,
                    "Actual Total Time": 8.796,
                    "Actual Rows": 126.0,
                    "Actual Loops": 1,
                    "Disabled": False,
                    "Recheck Cond": "((owneruserid = 5) OR (lasteditoruserid = 5))",
                    "Rows Removed by Index Recheck": 0,
                    "Exact Heap Blocks": 96,
                    "Lossy Heap Blocks": 0,
                    "Shared Hit Blocks": 2,
                    "Shared Read Blocks": 98,
                    "Shared Dirtied Blocks": 0,
                    "Shared Written Blocks": 0,
                    "Local Hit Blocks": 0,
                    "Local Read Blocks": 0,
                    "Local Dirtied Blocks": 0,
                    "Local Written Blocks": 0,
                    "Temp Read Blocks": 0,
                    "Temp Written Blocks": 0,
                    "Plans": [
                        {
                            "Node Type": "BitmapOr",
                            "Parent Relationship": "Outer",
                            "Parallel Aware": False,
                            "Async Capable": False,
                            "Startup Cost": 9.46,
                            "Total Cost": 9.46,
                            "Plan Rows": 109,
                            "Plan Width": 0,
                            "Actual Startup Time": 1.969,
                            "Actual Total Time": 1.971,
                            "Actual Rows": 0.0,
                            "Actual Loops": 1,
                            "Disabled": False,
                            "Shared Hit Blocks": 2,
                            "Shared Read Blocks": 2,
                            "Shared Dirtied Blocks": 0,
                            "Shared Written Blocks": 0,
                            "Local Hit Blocks": 0,
                            "Local Read Blocks": 0,
                            "Local Dirtied Blocks": 0,
                            "Local Written Blocks": 0,
                            "Temp Read Blocks": 0,
                            "Temp Written Blocks": 0,
                            "Plans": [
                                {
                                    "Node Type": "Bitmap Index Scan",
                                    "Parent Relationship": "Member",
                                    "Parallel Aware": False,
                                    "Async Capable": False,
                                    "Index Name": "posts_owneruserid_fkey",
                                    "Startup Cost": 0.0,
                                    "Total Cost": 5.07,
                                    "Plan Rows": 104,
                                    "Plan Width": 0,
                                    "Actual Startup Time": 0.019,
                                    "Actual Total Time": 0.019,
                                    "Actual Rows": 117.0,
                                    "Actual Loops": 1,
                                    "Disabled": False,
                                    "Index Cond": "(owneruserid = 5)",
                                    "Index Searches": 1,
                                    "Shared Hit Blocks": 2,
                                    "Shared Read Blocks": 0,
                                    "Shared Dirtied Blocks": 0,
                                    "Shared Written Blocks": 0,
                                    "Local Hit Blocks": 0,
                                    "Local Read Blocks": 0,
                                    "Local Dirtied Blocks": 0,
                                    "Local Written Blocks": 0,
                                    "Temp Read Blocks": 0,
                                    "Temp Written Blocks": 0,
                                },
                                {
                                    "Node Type": "Bitmap Index Scan",
                                    "Parent Relationship": "Member",
                                    "Parallel Aware": False,
                                    "Async Capable": False,
                                    "Index Name": "posts_lasteditoruserid_fkey",
                                    "Startup Cost": 0.0,
                                    "Total Cost": 4.33,
                                    "Plan Rows": 5,
                                    "Plan Width": 0,
                                    "Actual Startup Time": 1.949,
                                    "Actual Total Time": 1.95,
                                    "Actual Rows": 47.0,
                                    "Actual Loops": 1,
                                    "Disabled": False,
                                    "Index Cond": "(lasteditoruserid = 5)",
                                    "Index Searches": 1,
                                    "Shared Hit Blocks": 0,
                                    "Shared Read Blocks": 2,
                                    "Shared Dirtied Blocks": 0,
                                    "Shared Written Blocks": 0,
                                    "Local Hit Blocks": 0,
                                    "Local Read Blocks": 0,
                                    "Local Dirtied Blocks": 0,
                                    "Local Written Blocks": 0,
                                    "Temp Read Blocks": 0,
                                    "Temp Written Blocks": 0,
                                },
                            ],
                        }
                    ],
                },
                {
                    "Node Type": "Index Scan",
                    "Parent Relationship": "Inner",
                    "Parallel Aware": False,
                    "Async Capable": False,
                    "Scan Direction": "Forward",
                    "Index Name": "users_pkey",
                    "Relation Name": "users",
                    "Alias": "u",
                    "Startup Cost": 0.29,
                    "Total Cost": 6.8,
                    "Plan Rows": 40325,
                    "Plan Width": 13,
                    "Actual Startup Time": 0.023,
                    "Actual Total Time": 0.023,
                    "Actual Rows": 0.98,
                    "Actual Loops": 126,
                    "Disabled": False,
                    "Index Cond": "(id = p.owneruserid)",
                    "Rows Removed by Index Recheck": 0,
                    "Index Searches": 123,
                    "Shared Hit Blocks": 359,
                    "Shared Read Blocks": 10,
                    "Shared Dirtied Blocks": 0,
                    "Shared Written Blocks": 0,
                    "Local Hit Blocks": 0,
                    "Local Read Blocks": 0,
                    "Local Dirtied Blocks": 0,
                    "Local Written Blocks": 0,
                    "Temp Read Blocks": 0,
                    "Temp Written Blocks": 0,
                },
            ],
        },
        "Planning": {
            "Shared Hit Blocks": 260,
            "Shared Read Blocks": 0,
            "Shared Dirtied Blocks": 0,
            "Shared Written Blocks": 0,
            "Local Hit Blocks": 0,
            "Local Read Blocks": 0,
            "Local Dirtied Blocks": 0,
            "Local Written Blocks": 0,
            "Temp Read Blocks": 0,
            "Temp Written Blocks": 0,
        },
        "Planning Time": 1.978,
        "Triggers": [],
        "Execution Time": 11.94,
    },
    {"Optimizer": {"Planner": "Custom Hook", "Join Ordering": "Dynamic Programming"}},
]

POSTS = TableReference("posts", "p")
USERS = TableReference("users", "u")


def bitmap_heap_scan_join() -> QueryPlan:
    """The captured bitmap heap scan join, converted into PostBOUND's plan model."""
    with warnings.catch_warnings():
        # Converting the BitmapOr node warns that Postgres does not report its actual rows. This is the documented
        # behaviour of the EXPLAIN parser (tested in test_postgres_explain.py) and irrelevant here.
        warnings.filterwarnings("ignore", message=".*actual number of rows for bitmap nodes.*", category=UserWarning)
        return PostgresPlan(BITMAP_HEAP_SCAN_JOIN_ANALYZE).as_qep()


# -- regression tests --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"target_cardinality": "estimates"}],
    ids=["default", "explicit"],
)
def test_parameters_from_plan_uses_the_estimates_when_asked_for_estimates(kwargs: dict) -> None:
    """Regression guard for cacf542: the `target_cardinality` literal was renamed from ``"estimated"`` to
    ``"estimates"`` (also the new default), but the body still compared against ``"estimated"``. Every request for
    estimates therefore fell through to the fallback branch, which prefers the actual cardinality whenever one exists.
    """
    params = parameters_from_plan(SCAN_WITH_ESTIMATE_AND_MEASUREMENT, **kwargs)

    assert params.cardinalities == {frozenset([R]): ESTIMATED}


def test_join_order_hint_type_has_a_clean_label() -> None:
    """Regression guard for 8ac3b52: merging the linear and bushy join order hint types introduced the value
    ``"Join orderˆˆ"`` with two stray characters, which leaked into `str()` output and serialized pre-check results.
    """
    assert HintType.JoinOrder.value == "Join order"


@pytest.mark.parametrize("target_cardinality", ["estimates", "actual"])
def test_parameters_from_plan_ignores_index_lookups_below_a_bitmap_heap_scan(
    target_cardinality: Literal["estimates", "actual"],
) -> None:
    """Regression guard for f422da3: `parameters_from_plan` recursed into every child of a plan node, including the
    children of a Postgres Bitmap Heap Scan. Only the heap scan carries the base table, while the BitmapOr / Bitmap
    Index Scan nodes below it only reference indexes, so their `tables()` is empty. Since these nodes do have valid
    cardinalities, the function generated cardinality hints for the empty set of tables (and, once
    `PlanParameterization.add_cardinality` started rejecting empty table sets, crashed with a ValueError instead).
    """
    plan = bitmap_heap_scan_join()
    index_lookups = list(plan.find_all_nodes(lambda node: not node.tables()))
    assert len(index_lookups) == 3, "the fixture must contain the BitmapOr and both Bitmap Index Scans"

    params = parameters_from_plan(plan, target_cardinality=target_cardinality)

    assert params.cardinalities.keys() == {frozenset([POSTS]), frozenset([USERS]), frozenset([POSTS, USERS])}
