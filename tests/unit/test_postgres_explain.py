"""Tests for `postbound.postgres._explain`, driven by real EXPLAIN (FORMAT JSON) output.

`PostgresExplain` and `PostgresPlan` take a raw EXPLAIN dict and never touch a connection -- this is the best
isolated module in the whole backend. But hand-assembling that JSON ourselves would risk encoding our own
assumptions about Postgres's output shape rather than testing against the real thing, which is exactly the
kind of bug this module exists to catch.

So every fixture below is the **unmodified, verbatim** output of ``EXPLAIN (FORMAT JSON [, ANALYZE, BUFFERS])``
against the ``stats`` workload database (Stack Overflow schema: ``posts``, ``users``, ``badges``), captured
once and hard-coded here. Nothing is executed at test time -- these are static Python literals, parsed exactly
like any other test fixture. Each fixture's docstring names the exact query that produced it, so it can be
re-captured if the schema or Postgres version ever changes enough to matter.

The one exception is `MINIMAL_NODE`, used only to test default values when fields are *absent* -- that is a
property of `PostgresExplain.__init__`'s `.get(..., default)` calls, not a claim about what a real plan looks
like, so a deliberately minimal input is the right tool there.
"""

from __future__ import annotations

import math

import pytest

from postbound._core import Cardinality, JoinOperator, ScanOperator, TableReference
from postbound.postgres._explain import PostgresExplain, PostgresPlan

# -- real, captured fixtures --------------------------------------------------------------------------
#
# Captured against the `stats` workload (`.psycopg_connection_stats`) on an otherwise-default Postgres
# configuration. Table sizes at capture time: badges ~79.9k rows, posts ~92.0k rows, users ~40.3k rows.

# EXPLAIN (FORMAT JSON) SELECT * FROM badges WHERE name = 'Editor'
SEQ_SCAN_WITH_FILTER = [
    {
        "Plan": {
            "Node Type": "Seq Scan",
            "Parallel Aware": False,
            "Async Capable": False,
            "Relation Name": "badges",
            "Alias": "badges",
            "Startup Cost": 0.0,
            "Total Cost": 1558.14,
            "Plan Rows": 8254,
            "Plan Width": 26,
            "Disabled": False,
            "Filter": "((name)::text = 'Editor'::text)",
        }
    }
]

# EXPLAIN (FORMAT JSON) SELECT * FROM users WHERE id = 42
INDEX_SCAN = [
    {
        "Plan": {
            "Node Type": "Index Scan",
            "Parallel Aware": False,
            "Async Capable": False,
            "Scan Direction": "Forward",
            "Index Name": "users_pkey",
            "Relation Name": "users",
            "Alias": "users",
            "Startup Cost": 0.29,
            "Total Cost": 8.31,
            "Plan Rows": 1,
            "Plan Width": 374,
            "Disabled": False,
            "Index Cond": "(id = 42)",
        }
    }
]

# SET enable_hashjoin = off; SET enable_mergejoin = off;
# EXPLAIN (FORMAT JSON)
# SELECT p.id, u.displayname FROM posts p JOIN users u ON p.owneruserid = u.id WHERE u.id = 42
#
# Real Postgres lists the Outer child (users, the smaller/driving side) before the Inner child (posts) in
# "Plans" -- the opposite order from what a naive hand-built fixture would likely guess.
NESTED_LOOP_JOIN = [
    {
        "Plan": {
            "Node Type": "Nested Loop",
            "Parallel Aware": False,
            "Async Capable": False,
            "Join Type": "Inner",
            "Startup Cost": 0.58,
            "Total Cost": 25.48,
            "Plan Rows": 5,
            "Plan Width": 13,
            "Disabled": False,
            "Inner Unique": False,
            "Plans": [
                {
                    "Node Type": "Index Scan",
                    "Parent Relationship": "Outer",
                    "Parallel Aware": False,
                    "Async Capable": False,
                    "Scan Direction": "Forward",
                    "Index Name": "users_pkey",
                    "Relation Name": "users",
                    "Alias": "u",
                    "Startup Cost": 0.29,
                    "Total Cost": 8.31,
                    "Plan Rows": 1,
                    "Plan Width": 13,
                    "Disabled": False,
                    "Index Cond": "(id = 42)",
                },
                {
                    "Node Type": "Index Scan",
                    "Parent Relationship": "Inner",
                    "Parallel Aware": False,
                    "Async Capable": False,
                    "Scan Direction": "Forward",
                    "Index Name": "posts_owneruserid_fkey",
                    "Relation Name": "posts",
                    "Alias": "p",
                    "Startup Cost": 0.29,
                    "Total Cost": 17.13,
                    "Plan Rows": 5,
                    "Plan Width": 8,
                    "Disabled": False,
                    "Index Cond": "(owneruserid = 42)",
                },
            ],
        }
    }
]

# EXPLAIN (FORMAT JSON, ANALYZE, BUFFERS) SELECT * FROM users WHERE id = -1
# (id = -1 is a real row in the Stack Overflow dump, conventionally used for a deleted/system user.)
ANALYZE_INDEX_SCAN = [
    {
        "Plan": {
            "Node Type": "Index Scan",
            "Parallel Aware": False,
            "Async Capable": False,
            "Scan Direction": "Forward",
            "Index Name": "users_pkey",
            "Relation Name": "users",
            "Alias": "users",
            "Startup Cost": 0.29,
            "Total Cost": 8.31,
            "Plan Rows": 1,
            "Plan Width": 374,
            "Actual Startup Time": 0.018,
            "Actual Total Time": 0.019,
            "Actual Rows": 1.0,
            "Actual Loops": 1,
            "Disabled": False,
            "Index Cond": "(id = '-1'::integer)",
            "Rows Removed by Index Recheck": 0,
            "Index Searches": 1,
            "Shared Hit Blocks": 3,
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
        "Planning": {
            "Shared Hit Blocks": 74,
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
        "Planning Time": 0.046,
        "Triggers": [],
        "Execution Time": 0.03,
    }
]

# SET max_parallel_workers_per_gather = 4;
# SET parallel_setup_cost = 0; SET parallel_tuple_cost = 0; SET min_parallel_table_scan_size = 0;
# EXPLAIN (FORMAT JSON) SELECT * FROM posts WHERE score > 0
#
# The child Seq Scan's "Plan Rows" (17697) is Postgres's *per-worker* estimate; the Gather's own "Plan Rows"
# (70789) is its already-aggregated total. PostBOUND's own cardinality-scaling logic re-derives the child's
# *total* estimate independently, by multiplying the per-worker figure by (workers planned + 1) -- see the
# `as_qep` tests below for what that actually produces.
PARALLEL_GATHER = [
    {
        "Plan": {
            "Node Type": "Gather",
            "Parallel Aware": False,
            "Async Capable": False,
            "Startup Cost": 0.0,
            "Total Cost": 11679.42,
            "Plan Rows": 70789,
            "Plan Width": 1017,
            "Disabled": False,
            "Workers Planned": 4,
            "Single Copy": False,
            "Plans": [
                {
                    "Node Type": "Seq Scan",
                    "Parent Relationship": "Outer",
                    "Parallel Aware": True,
                    "Async Capable": False,
                    "Relation Name": "posts",
                    "Alias": "posts",
                    "Startup Cost": 0.0,
                    "Total Cost": 11679.42,
                    "Plan Rows": 17697,
                    "Plan Width": 1017,
                    "Disabled": False,
                    "Filter": "(score > 0)",
                }
            ],
        }
    }
]

# EXPLAIN (FORMAT JSON)
# SELECT u.id, (SELECT count(*) FROM posts p WHERE p.owneruserid = u.id) AS post_count
# FROM users u WHERE u.id = -1
#
# A correlated scalar subquery in the SELECT list. Postgres attaches it as a sibling "Plans" entry on the
# outer Index Only Scan with "Parent Relationship": "SubPlan", rather than nesting it as a join child.
SUBPLAN_SCALAR = [
    {
        "Plan": {
            "Node Type": "Index Only Scan",
            "Parallel Aware": False,
            "Async Capable": False,
            "Scan Direction": "Forward",
            "Index Name": "users_pkey",
            "Relation Name": "users",
            "Alias": "u",
            "Startup Cost": 0.29,
            "Total Cost": 16.75,
            "Plan Rows": 1,
            "Plan Width": 12,
            "Disabled": False,
            "Index Cond": "(id = '-1'::integer)",
            "Plans": [
                {
                    "Node Type": "Aggregate",
                    "Strategy": "Plain",
                    "Partial Mode": "Simple",
                    "Parent Relationship": "SubPlan",
                    "Subplan Name": "SubPlan 1",
                    "Parallel Aware": False,
                    "Async Capable": False,
                    "Startup Cost": 8.43,
                    "Total Cost": 8.44,
                    "Plan Rows": 1,
                    "Plan Width": 8,
                    "Disabled": False,
                    "Plans": [
                        {
                            "Node Type": "Index Only Scan",
                            "Parent Relationship": "Outer",
                            "Parallel Aware": False,
                            "Async Capable": False,
                            "Scan Direction": "Forward",
                            "Index Name": "posts_owneruserid_fkey",
                            "Relation Name": "posts",
                            "Alias": "p",
                            "Startup Cost": 0.29,
                            "Total Cost": 8.42,
                            "Plan Rows": 7,
                            "Plan Width": 0,
                            "Disabled": False,
                            "Index Cond": "(owneruserid = u.id)",
                        }
                    ],
                }
            ],
        }
    }
]

# SET enable_seqscan = off;
# EXPLAIN (FORMAT JSON, ANALYZE) SELECT * FROM posts WHERE owneruserid = 5 OR lasteditoruserid = 5
#
# A real BitmapOr over two index scans. The BitmapOr node's own "Actual Rows" is 0.0 even though 126 rows
# were genuinely produced by its parent Bitmap Heap Scan -- this is the documented Postgres limitation that
# `PostgresExplain.true_cardinality` warns about for BitmapAnd/BitmapOr nodes.
BITMAP_OR_ANALYZE = [
    {
        "Plan": {
            "Node Type": "Bitmap Heap Scan",
            "Parallel Aware": False,
            "Async Capable": False,
            "Relation Name": "posts",
            "Alias": "posts",
            "Startup Cost": 10.2,
            "Total Cost": 737.13,
            "Plan Rows": 202,
            "Plan Width": 1017,
            "Actual Startup Time": 2.523,
            "Actual Total Time": 12.983,
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
                    "Startup Cost": 10.2,
                    "Total Cost": 10.2,
                    "Plan Rows": 202,
                    "Plan Width": 0,
                    "Actual Startup Time": 0.463,
                    "Actual Total Time": 0.464,
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
                            "Total Cost": 5.37,
                            "Plan Rows": 144,
                            "Plan Width": 0,
                            "Actual Startup Time": 0.447,
                            "Actual Total Time": 0.448,
                            "Actual Rows": 117.0,
                            "Actual Loops": 1,
                            "Disabled": False,
                            "Index Cond": "(owneruserid = 5)",
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
                        {
                            "Node Type": "Bitmap Index Scan",
                            "Parent Relationship": "Member",
                            "Parallel Aware": False,
                            "Async Capable": False,
                            "Index Name": "posts_lasteditoruserid_fkey",
                            "Startup Cost": 0.0,
                            "Total Cost": 4.73,
                            "Plan Rows": 58,
                            "Plan Width": 0,
                            "Actual Startup Time": 0.014,
                            "Actual Total Time": 0.014,
                            "Actual Rows": 47.0,
                            "Actual Loops": 1,
                            "Disabled": False,
                            "Index Cond": "(lasteditoruserid = 5)",
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
                    ],
                }
            ],
        },
        "Planning": {
            "Shared Hit Blocks": 192,
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
        "Planning Time": 1.225,
        "Triggers": [],
        "Execution Time": 13.074,
    }
]

#: Deliberately minimal -- see the module docstring for why this one is not a captured real plan.
MINIMAL_NODE = {"Node Type": "Result"}


# -- PostgresExplain: field parsing ------------------------------------------------------------------


def test_basic_fields_are_parsed_from_the_node() -> None:
    node = PostgresExplain(SEQ_SCAN_WITH_FILTER[0]["Plan"])

    assert node.node_type == "Seq Scan"
    assert node.relation_name == "badges"
    assert node.relation_alias == "badges"
    assert node.cost == 1558.14
    assert node.cardinality_estimate == 8254
    assert node.children == []


def test_missing_fields_use_documented_defaults() -> None:
    node = PostgresExplain(MINIMAL_NODE)

    assert math.isnan(node.cost)
    assert math.isnan(node.cardinality_estimate)
    assert math.isnan(node.execution_time)
    assert node.loops == 1
    assert node.relation_name is None
    assert node.launched_workers == 0
    assert node.planned_workers == 0
    assert node.sort_keys == ""


def test_execution_time_is_converted_from_milliseconds_to_seconds() -> None:
    node = PostgresExplain(ANALYZE_INDEX_SCAN[0]["Plan"])

    assert node.execution_time == pytest.approx(0.019 / 1000)


def test_true_cardinality_reads_actual_rows() -> None:
    node = PostgresExplain(ANALYZE_INDEX_SCAN[0]["Plan"])

    assert node.true_cardinality == 1.0


def test_true_cardinality_warns_and_returns_nan_for_bitmap_or() -> None:
    """Postgres cannot report a meaningful actual-row count for BitmapAnd/Or nodes.

    `BITMAP_OR_ANALYZE`'s BitmapOr node reports "Actual Rows": 0.0, even though its parent Bitmap Heap Scan
    genuinely produced 126 rows from it -- the exact Postgres limitation this property warns about.
    """
    bitmap_or_node = PostgresExplain(BITMAP_OR_ANALYZE[0]["Plan"]).children[0]
    assert bitmap_or_node.node_type == "BitmapOr"

    with pytest.warns(UserWarning, match="bitmap nodes"):
        result = bitmap_or_node.true_cardinality

    assert math.isnan(result)


def test_children_are_parsed_recursively() -> None:
    node = PostgresExplain(NESTED_LOOP_JOIN[0]["Plan"])

    assert len(node.children) == 2
    assert all(isinstance(child, PostgresExplain) for child in node.children)
    assert {child.relation_name for child in node.children} == {"users", "posts"}


# -- classification -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("fixture", "is_scan", "is_join"),
    [
        (SEQ_SCAN_WITH_FILTER, True, False),
        (INDEX_SCAN, True, False),
        (NESTED_LOOP_JOIN, False, True),
        (SUBPLAN_SCALAR, True, False),
    ],
)
def test_is_scan_and_is_join_classify_by_node_type(fixture: list[dict], is_scan: bool, is_join: bool) -> None:
    node = PostgresExplain(fixture[0]["Plan"])

    assert node.is_scan() is is_scan
    assert node.is_join() is is_join


def test_is_analyze_true_for_an_analyze_plan() -> None:
    node = PostgresExplain(ANALYZE_INDEX_SCAN[0]["Plan"])

    assert node.is_analyze() is True


def test_is_analyze_false_for_a_plain_explain_plan() -> None:
    node = PostgresExplain(SEQ_SCAN_WITH_FILTER[0]["Plan"])

    assert node.is_analyze() is False


def test_all_node_types_contains_seq_scan() -> None:
    assert "Seq Scan" in PostgresExplain.all_node_types()
    assert "Hash Join" in PostgresExplain.all_node_types()


# -- filter_conditions ---------------------------------------------------------------------------------


def test_filter_conditions_reports_a_seq_scan_filter() -> None:
    node = PostgresExplain(SEQ_SCAN_WITH_FILTER[0]["Plan"])

    assert node.filter_conditions() == {"Filter": "((name)::text = 'Editor'::text)"}


def test_filter_conditions_reports_an_index_condition() -> None:
    node = PostgresExplain(INDEX_SCAN[0]["Plan"])

    assert node.filter_conditions() == {"Index Cond": "(id = 42)"}


def test_filter_conditions_is_empty_when_none_present() -> None:
    node = PostgresExplain(PARALLEL_GATHER[0]["Plan"])  # the Gather node itself has no filter of its own

    assert node.filter_conditions() == {}


# -- inner_outer_children -------------------------------------------------------------------------------


def test_inner_outer_children_returns_empty_for_leaf_node() -> None:
    node = PostgresExplain(SEQ_SCAN_WITH_FILTER[0]["Plan"])

    assert node.inner_outer_children() == []


def test_inner_outer_children_returns_single_child_unchanged() -> None:
    node = PostgresExplain(PARALLEL_GATHER[0]["Plan"])

    assert len(node.inner_outer_children()) == 1


def test_inner_outer_children_orders_inner_before_outer() -> None:
    """`NESTED_LOOP_JOIN` lists the Outer child (users) first in "Plans", then the Inner child (posts) --
    `inner_outer_children` must still return (inner, outer).
    """
    node = PostgresExplain(NESTED_LOOP_JOIN[0]["Plan"])

    inner, outer = node.inner_outer_children()
    assert inner.relation_name == "posts"
    assert outer.relation_name == "users"


# -- parse_table -----------------------------------------------------------------------------------------


def test_parse_table_returns_none_for_non_scan_nodes() -> None:
    node = PostgresExplain(NESTED_LOOP_JOIN[0]["Plan"])

    assert node.parse_table() is None


def test_parse_table_uses_relation_and_alias() -> None:
    join_node = PostgresExplain(NESTED_LOOP_JOIN[0]["Plan"])
    posts_scan = next(child for child in join_node.children if child.relation_name == "posts")

    table = posts_scan.parse_table()
    assert table == TableReference("posts", "p")


def test_parse_table_alias_equal_to_relation_name_is_dropped() -> None:
    """`SEQ_SCAN_WITH_FILTER` was captured from an un-aliased query, so Postgres reports the same string for
    both "Relation Name" and "Alias" -- there is no real alias in that case.
    """
    node = PostgresExplain(SEQ_SCAN_WITH_FILTER[0]["Plan"])

    table = node.parse_table()
    assert table is not None
    assert table.alias == ""


# -- equality / hashing -----------------------------------------------------------------------------------


def test_equal_nodes_compare_equal_and_hash_equal() -> None:
    left = PostgresExplain(SEQ_SCAN_WITH_FILTER[0]["Plan"])
    right = PostgresExplain(SEQ_SCAN_WITH_FILTER[0]["Plan"])

    assert left == right
    assert hash(left) == hash(right)


def test_nodes_with_different_relations_are_not_equal() -> None:
    left = PostgresExplain(SEQ_SCAN_WITH_FILTER[0]["Plan"])
    right = PostgresExplain(INDEX_SCAN[0]["Plan"])

    assert left != right


# -- as_qep(): scan and join conversion -----------------------------------------------------------------


def test_as_qep_converts_a_simple_scan() -> None:
    node = PostgresExplain(SEQ_SCAN_WITH_FILTER[0]["Plan"])

    plan = node.as_qep()

    assert plan.operator == ScanOperator.SequentialScan
    assert plan.base_table == TableReference("badges")
    assert plan.estimated_cost == 1558.14
    assert plan.estimated_cardinality == Cardinality(8254)
    assert plan.children == ()


def test_as_qep_uses_get_for_the_index_name() -> None:
    """`QueryPlan` does not re-expose `index` as a direct property the way it does `base_table` or
    `parallel_workers` -- it is only reachable via `.get("index")`.
    """
    plan = PostgresExplain(INDEX_SCAN[0]["Plan"]).as_qep()

    assert plan.get("index") == "users_pkey"


def test_as_qep_orders_join_children_outer_then_inner() -> None:
    plan = PostgresExplain(NESTED_LOOP_JOIN[0]["Plan"]).as_qep()

    assert plan.operator == JoinOperator.NestedLoopJoin
    assert len(plan.children) == 2
    outer_child, inner_child = plan.children
    assert outer_child.base_table == TableReference("users", "u")
    assert inner_child.base_table == TableReference("posts", "p")


def test_as_qep_raises_on_unknown_parent_relationship() -> None:
    raw = {
        "Node Type": "Nested Loop",
        "Plans": [{**SEQ_SCAN_WITH_FILTER[0]["Plan"], "Parent Relationship": "Bogus"}],
    }

    with pytest.raises(ValueError, match="Unknown parent relationship"):
        PostgresExplain(raw).as_qep()


def test_as_qep_scales_estimated_cardinality_below_a_parallel_gather() -> None:
    """Postgres reports the per-worker row estimate on the node below a Gather (17697 here); PostBOUND
    re-derives the total by multiplying by (workers planned + 1) to account for the leader process, giving
    17697 * 5 = 88485 -- independent of whatever total Postgres's own Gather-level estimate (70789) says.
    """
    plan = PostgresExplain(PARALLEL_GATHER[0]["Plan"]).as_qep()

    scan_child = plan.children[0]
    assert scan_child.estimated_cardinality == Cardinality(88485)
    # The Gather node itself reports its own already-aggregated total and must not be re-scaled again.
    assert plan.estimated_cardinality == Cardinality(70789)


def test_as_qep_reads_parallel_workers_from_workers_planned() -> None:
    plan = PostgresExplain(PARALLEL_GATHER[0]["Plan"]).as_qep()

    assert plan.parallel_workers == 4


def test_as_qep_parallel_workers_is_zero_for_sequential_nodes() -> None:
    """Absence of parallelism is indicated by 0, per `QueryPlan.parallel_workers`'s own docstring."""
    plan = PostgresExplain(SEQ_SCAN_WITH_FILTER[0]["Plan"]).as_qep()

    assert plan.parallel_workers == 0


def test_as_qep_attaches_subplan_by_parent_relationship() -> None:
    plan = PostgresExplain(SUBPLAN_SCALAR[0]["Plan"]).as_qep()

    assert plan.children == ()
    subplan = plan.subplan
    assert subplan is not None
    assert subplan.root.operator is None  # Aggregate has no PostBOUND-modeled physical operator
    assert subplan.root.children[0].base_table == TableReference("posts", "p")


def test_as_qep_subplan_target_name_is_populated() -> None:
    plan = PostgresExplain(SUBPLAN_SCALAR[0]["Plan"]).as_qep()
    subplan = plan.subplan
    assert subplan is not None

    assert subplan.target_name == "SubPlan 1"


def test_as_qep_analyze_plan_carries_execution_measures() -> None:
    plan = PostgresExplain(ANALYZE_INDEX_SCAN[0]["Plan"]).as_qep()

    assert plan.actual_cardinality == Cardinality(1)
    assert plan.execution_time == pytest.approx(0.019 / 1000)
    assert plan.get("cache_hits") == 3
    assert plan.get("cache_misses") == 0


# -- PostgresPlan: whole-plan parsing --------------------------------------------------------------------


def test_postgres_plan_accepts_the_real_top_level_list_format() -> None:
    """`EXPLAIN (FORMAT JSON)` always returns a JSON array at the top level -- exactly what every fixture
    in this module already is.
    """
    plan = PostgresPlan(SEQ_SCAN_WITH_FILTER)

    assert plan.root.relation_name == "badges"


def test_postgres_plan_accepts_an_unwrapped_bare_dict() -> None:
    plan = PostgresPlan(SEQ_SCAN_WITH_FILTER[0])

    assert plan.root.relation_name == "badges"


def test_postgres_plan_rejects_data_without_a_plan_key() -> None:
    with pytest.raises(ValueError, match="missing 'Plan' key"):
        PostgresPlan({"Planning Time": 1.0})


def test_postgres_plan_converts_planning_time_to_seconds() -> None:
    plan = PostgresPlan(ANALYZE_INDEX_SCAN)

    assert plan.planning_time == pytest.approx(0.046 / 1000)


def test_postgres_plan_planning_time_defaults_to_nan_without_analyze() -> None:
    plan = PostgresPlan(SEQ_SCAN_WITH_FILTER)

    assert math.isnan(plan.planning_time)


def test_postgres_plan_as_qep_is_cached() -> None:
    plan = PostgresPlan(SEQ_SCAN_WITH_FILTER)

    assert plan.as_qep() is plan.as_qep()


def test_postgres_plan_is_analyze_delegates_to_root_node() -> None:
    plan = PostgresPlan(ANALYZE_INDEX_SCAN)

    assert plan.is_analyze() is True


# -- PostgresPlan attribute delegation chain --------------------------------------------------------------


def test_postgres_plan_delegates_unknown_attribute_to_root_explain_node() -> None:
    """`node_type` is not defined on PostgresPlan itself, only on the wrapped PostgresExplain."""
    plan = PostgresPlan(SEQ_SCAN_WITH_FILTER)

    assert plan.node_type == "Seq Scan"


def test_postgres_plan_delegates_further_to_the_normalized_query_plan() -> None:
    """`is_linear` exists only on `QueryPlan`, not on `PostgresPlan` or `PostgresExplain`.

    It must fall through the full two-step delegation chain in `__getattribute__`.
    """
    assert not hasattr(PostgresExplain, "is_linear")

    plan = PostgresPlan(SEQ_SCAN_WITH_FILTER)

    assert plan.is_linear() is True


def test_postgres_plan_raises_attribute_error_for_truly_unknown_attribute() -> None:
    plan = PostgresPlan(SEQ_SCAN_WITH_FILTER)

    with pytest.raises(AttributeError):
        _ = plan.this_attribute_does_not_exist_anywhere
