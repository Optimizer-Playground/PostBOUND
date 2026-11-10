"""Tests for `postbound.duckdb._duckdb` -- the offline part of the DuckDB backend: parsing EXPLAIN output.

The plan parser and the scan-to-table binding are pure functions of the EXPLAIN JSON and the parsed query, so they
are tested here at tier 0 with output captured verbatim from DuckDB (quacklab 2.0.2, in-memory database). Each
fixture names the statements that produced it; to recapture, run them in a fresh in-memory database after::

    CREATE TABLE r (a INTEGER, b INTEGER);
    INSERT INTO r VALUES (1, 10), (2, 20), (3, 30);

Whether DuckDB accepts the generated EXPLAIN statements, and the `DuckDBOptimizer` code paths that run them, are
covered against a real engine in `tests/test_duckdb.py`.
"""

from __future__ import annotations

import json

import pytest

from postbound import ScanOperator, TableReference, parser
from postbound.duckdb import parse_duckdb_plan
from postbound.duckdb._duckdb import _bind_node_to_table

# -- fixtures -----------------------------------------------------------------------------------------------


def parse(sql: str):
    return parser.parse_query(sql, bind_columns=False)


# EXPLAIN (FORMAT JSON) SELECT * FROM r WHERE r.a = 1
# (identical output for `SELECT * FROM r AS x WHERE x.a = 1` -- DuckDB plans carry no aliases)
EXPLAIN_FILTER_SCAN = [
    {
        "name": "SEQ_SCAN",
        "children": [],
        "extra_info": {
            "Table": "memory.main.r",
            "Type": "Sequential Scan",
            "Projections": ["a", "b"],
            "Filters": "(a = 1)",
            "Estimated Cardinality": "1",
        },
    }
]

# EXPLAIN (ANALYZE, FORMAT JSON) SELECT * FROM r WHERE r.a = 1
# The operator tree is wrapped in a profiling summary, and starts with RESULT_COLLECTOR -> EXPLAIN_ANALYZE.
EXPLAIN_ANALYZE_FILTER_SCAN = {
    "io": {
        "total_bytes_read": 0,
        "total_bytes_written": 0,
        "total_read_operations": 0,
        "total_read_time": 0.0,
        "total_write_operations": 0,
        "total_write_time": 0.0,
    },
    "operator": [
        {
            "extra_info": {},
            "intermediate_rows": 0,
            "intermediate_size_bytes": 0,
            "name": "RESULT_COLLECTOR",
            "timing": 0.0,
            "type": "RESULT_COLLECTOR",
            "children": [
                {
                    "extra_info": {},
                    "intermediate_rows": 0,
                    "intermediate_size_bytes": 0,
                    "name": "EXPLAIN_ANALYZE",
                    "timing": 1.25e-7,
                    "type": "EXPLAIN_ANALYZE",
                    "children": [
                        {
                            "extra_info": {
                                "Table": "memory.main.r",
                                "Type": "Sequential Scan",
                                "Projections": ["a", "b"],
                                "Filters": "(a = 1)",
                                "Estimated Cardinality": "1",
                            },
                            "intermediate_rows": 1,
                            "intermediate_size_bytes": 8,
                            "name": "SEQ_SCAN",
                            "row_groups_scanned": 1,
                            "rows_scanned": 3,
                            "timing": 0.000008375,
                            "total_row_groups_to_scan": 1,
                            "type": "TABLE_SCAN",
                        }
                    ],
                }
            ],
        }
    ],
    "optimizer": {
        "aggregate_function_rewriter": 0.000001125,
        "aggregate_reuse": 2.91e-7,
        "build_side_probe_side": 2.91e-7,
        "column_lifetime": 4.16e-7,
        "common_aggregate": 8.3e-8,
        "common_subexpressions": 1.67e-7,
        "common_subplan": 0.000001125,
        "cte_filter_pusher": 8.4e-8,
        "cte_inlining": 8.3e-8,
        "deliminator": 1.66e-7,
        "duplicate_groups": 2.91e-7,
        "empty_result_pullup": 4.1e-8,
        "expression_rewriter": 0.000007834,
        "filter_pullup": 8.4e-8,
        "filter_pushdown": 0.000004333,
        "grouping_sets": 1.25e-7,
        "in_clause": 1.25e-7,
        "join_elimination": 2.08e-7,
        "join_filter_pushdown": 8.3e-8,
        "join_order": 0.00000475,
        "late_materialization": 1.25e-7,
        "limit_pushdown": 8.4e-8,
        "outer_join_simplification": 4.58e-7,
        "partial_aggregate_pushdown": 1.25e-7,
        "partitioned_execution": 8.3e-8,
        "projection_placement": 6.66e-7,
        "projection_pullup": 3.75e-7,
        "redundant_order_keys": 3.33e-7,
        "regex_range": 4.2e-8,
        "reorder_filter": 8.3e-8,
        "row_group_pruner": 4.2e-8,
        "row_number_rewriter": 8.3e-8,
        "sampling_pushdown": 8.3e-8,
        "scalar_fn_pushdown": 4.2e-8,
        "statistics_propagation": 0.000003,
        "top_n": 4.2e-8,
        "top_n_window_elimination": 2.5e-7,
        "total_time": 0.000046375,
        "type_pushdown": 4.2e-8,
        "unnest_rewriter": 1.25e-7,
        "unused_columns": 0.000002291,
        "window_self_join": 4.1e-8,
    },
    "physical_planner": {
        "column_binding": 4.16e-7,
        "create_plan": 0.000001542,
        "resolve_types": 8.4e-8,
        "total_time": 0.000002958,
    },
    "planner": {"binding_time": 0.000013667, "total_time": 0.000027125},
    "query": {
        "cpu_time": 0.0000085,
        "sql": "EXPLAIN (ANALYZE, FORMAT JSON)\nSELECT *\nFROM r\nWHERE r.a = 1;",
        "total_intermediate_rows": 1,
        "total_intermediate_size_bytes": 8,
        "total_row_groups_scanned": 1,
        "total_row_groups_to_scan": 1,
        "total_rows_scanned": 3,
        "total_time": 0.000136625,
    },
    "system": {
        "blocked_thread_time": 0.0,
        "peak_buffer_memory": 0,
        "peak_streaming_buffer_size": 0,
        "peak_temp_dir_size": 0,
        "total_bytes_spilled": 0,
        "total_memory_allocated": 0,
    },
}


def scan_node_for(table: str) -> dict:
    """Synthetic: the captured scan node with its `Table` entry replaced.

    DuckDB always reports `catalog.schema.table`; the other shapes exist only to reach the remaining branches of
    `_bind_node_to_table`, which accepts them defensively.
    """
    node = EXPLAIN_FILTER_SCAN[0]
    return {**node, "extra_info": {**node["extra_info"], "Table": table}}


# -- _bind_node_to_table ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("table", "sql"),
    [
        ("r", "SELECT * FROM r WHERE r.a = 1"),
        ("main.r", "SELECT * FROM main.r WHERE r.a = 1"),
        ("main.r", "SELECT * FROM r WHERE r.a = 1"),
        ("memory.main.r", "SELECT * FROM memory.main.r WHERE r.a = 1"),
    ],
    ids=["bare-name", "schema-qualified", "schema-qualified-fallback-to-name", "catalog-qualified"],
)
def test_bind_node_to_table_matches_each_qualification_level(table: str, sql: str) -> None:
    query = parse(sql)

    bound = _bind_node_to_table(scan_node_for(table), query=query)

    assert bound == query.tables().pop()


def test_bind_node_to_table_falls_back_to_the_final_component_for_deeper_names() -> None:
    query = parse("SELECT * FROM r WHERE r.a = 1")

    with pytest.warns(UserWarning, match="Unexpected table format"):
        bound = _bind_node_to_table(scan_node_for("db.catalog.main.r"), query=query)

    assert bound == TableReference("r")


def test_bind_node_to_table_returns_none_for_a_table_the_query_does_not_reference() -> None:
    query = parse("SELECT * FROM s WHERE s.a = 1")

    assert _bind_node_to_table(EXPLAIN_FILTER_SCAN[0], query=query) is None


def test_bind_node_to_table_matches_a_schema_qualified_query_against_a_catalog_qualified_scan() -> None:
    query = parse("SELECT * FROM main.r WHERE r.a = 1")

    assert _bind_node_to_table(EXPLAIN_FILTER_SCAN[0], query=query) == TableReference("r", schema="main")


# -- parse_duckdb_plan --------------------------------------------------------------------------------------


def test_parse_duckdb_plan_unwraps_the_profiling_root_of_explain_analyze() -> None:
    """The ANALYZE output is a single object: a profiling summary with the operator tree below `operator`, which
    starts with the `RESULT_COLLECTOR` and `EXPLAIN_ANALYZE` operators before the actual plan. All wrappers must be
    skipped and the measurements of the scan must be kept.
    """
    query = parse("SELECT * FROM r WHERE r.a = 1")

    plan = parse_duckdb_plan(EXPLAIN_ANALYZE_FILTER_SCAN, query=query)

    assert plan.operator == ScanOperator.SequentialScan
    assert plan.base_table == TableReference("r")
    assert plan.actual_cardinality == 1
    assert plan.execution_time == pytest.approx(0.000008375)


EXPLAIN_JSON, ANALYZE_JSON = json.dumps(EXPLAIN_FILTER_SCAN), json.dumps(EXPLAIN_ANALYZE_FILTER_SCAN)


@pytest.mark.parametrize(
    ("raw_plan", "analyzed"),
    [
        ([("physical_plan", EXPLAIN_JSON)], False),
        (("physical_plan", EXPLAIN_JSON), False),
        (EXPLAIN_JSON, False),
        (EXPLAIN_FILTER_SCAN, False),
        (EXPLAIN_FILTER_SCAN[0], False),
        ([("analyzed_plan", ANALYZE_JSON)], True),
        (("analyzed_plan", ANALYZE_JSON), True),
        (ANALYZE_JSON, True),
        (EXPLAIN_ANALYZE_FILTER_SCAN, True),
        (EXPLAIN_ANALYZE_FILTER_SCAN["operator"][0], True),
    ],
    ids=[
        "explain-raw-result-set",
        "explain-simplified-result-set",
        "explain-json-string",
        "explain-parsed-json",
        "explain-operator-tree",
        "analyze-raw-result-set",
        "analyze-simplified-result-set",
        "analyze-json-string",
        "analyze-parsed-json",
        "analyze-operator-tree",
    ],
)
def test_parse_duckdb_plan_unwraps_every_representation_of_the_explain_output(raw_plan: object, analyzed: bool) -> None:
    """The result set of `execute_query` can be passed as-is, but each intermediate representation works as well."""
    query = parse("SELECT * FROM r WHERE r.a = 1")

    plan = parse_duckdb_plan(raw_plan, query=query)

    assert plan.operator == ScanOperator.SequentialScan
    assert plan.base_table == TableReference("r")
    assert plan.actual_cardinality.is_valid() is analyzed


@pytest.mark.parametrize(
    "raw_plan",
    [
        [("physical_plan", EXPLAIN_JSON), ("physical_plan", EXPLAIN_JSON)],
        ("physical_plan", EXPLAIN_JSON, "extra column"),
        "physical_plan",
        42,
    ],
    ids=["two-rows", "three-columns", "no-json", "no-explain-output"],
)
def test_parse_duckdb_plan_rejects_input_that_is_no_explain_output(raw_plan: object) -> None:
    with pytest.raises(ValueError):
        parse_duckdb_plan(raw_plan)


def test_parse_duckdb_plan_rejects_a_node_without_an_operator_name() -> None:
    """An unknown wrapper object (e.g. a changed profiling format) must fail loudly instead of yielding a bogus plan."""
    with pytest.raises(ValueError, match="operator name is missing"):
        parse_duckdb_plan({"children": [EXPLAIN_FILTER_SCAN[0]]})


# -- regression tests --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("sql", "expected"),
    [
        ("SELECT * FROM r WHERE r.a = 1", TableReference("r")),
        ("SELECT * FROM r AS x WHERE x.a = 1", TableReference("r", "x")),
    ],
    ids=["plain", "aliased"],
)
def test_parse_duckdb_plan_binds_a_catalog_qualified_scan_to_the_query_table(
    sql: str, expected: TableReference
) -> None:
    """Regression guard for the v0.22.1 DuckDB table-binding fix: `_bind_node_to_table` compared the scan node's
    `Table` entry with `TableReference.full_name` by plain string equality. DuckDB reports fully qualified names
    (`memory.main.r`), while a parsed query references just `r`, so no scan node was ever bound and every DuckDB
    plan came out without base tables.
    """
    query = parse(sql)

    plan = parse_duckdb_plan(EXPLAIN_FILTER_SCAN[0], query=query)

    assert plan.base_table == expected
    assert plan.tables() == {expected}
