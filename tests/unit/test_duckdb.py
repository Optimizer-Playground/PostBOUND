"""Tests for `postbound.duckdb._duckdb` -- the offline part of the DuckDB backend: parsing EXPLAIN output.

The plan parser and the scan-to-table binding are pure functions of the EXPLAIN JSON and the parsed query, so they
are tested here at tier 0 with output captured verbatim from DuckDB (quacklab 1.5, in-memory database). Each
fixture names the statements that produced it; to recapture, run them in a fresh in-memory database after::

    CREATE TABLE r (a INTEGER, b INTEGER);
    INSERT INTO r VALUES (1, 10), (2, 20), (3, 30);

Whether DuckDB accepts the generated EXPLAIN statements, and the `DuckDBOptimizer` code paths that run them, are
covered against a real engine in `tests/test_duckdb.py`.
"""

from __future__ import annotations

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
            "Filters": "a=1",
            "Estimated Cardinality": "1",
        },
    }
]

# EXPLAIN (ANALYZE, FORMAT JSON) SELECT * FROM r WHERE r.a = 1
EXPLAIN_ANALYZE_FILTER_SCAN = {
    "total_memory_allocated": 0,
    "total_bytes_written": 0,
    "total_bytes_read": 0,
    "system_peak_temp_dir_size": 0,
    "system_peak_buffer_memory": 100352,
    "rows_returned": 0,
    "result_set_size": 0,
    "latency": 0.000134792,
    "wal_replay_entry_count": 0,
    "extra_info": {},
    "commit_local_storage_latency": 0.0,
    "attach_load_storage_latency": 0.0,
    "query_name": "EXPLAIN (ANALYZE, FORMAT JSON)\nSELECT *\nFROM r\nWHERE r.a = 1;",
    "cpu_time": 0.0000024570000000000004,
    "checkpoint_latency": 0.0,
    "cumulative_cardinality": 1,
    "waiting_to_attach_latency": 0.0,
    "write_to_wal_latency": 0.0,
    "attach_replay_wal_latency": 0.0,
    "blocked_thread_time": 0.0,
    "cumulative_rows_scanned": 3,
    "children": [
        {
            "system_peak_temp_dir_size": 0,
            "result_set_size": 0,
            "operator_type": "EXPLAIN_ANALYZE",
            "operator_timing": 4.1e-8,
            "operator_rows_scanned": 0,
            "extra_info": {},
            "system_peak_buffer_memory": 0,
            "cumulative_rows_scanned": 3,
            "cpu_time": 0.0000024570000000000004,
            "operator_name": "EXPLAIN_ANALYZE",
            "cumulative_cardinality": 1,
            "operator_cardinality": 0,
            "children": [
                {
                    "cumulative_cardinality": 1,
                    "operator_cardinality": 1,
                    "cumulative_rows_scanned": 3,
                    "operator_name": "SEQ_SCAN",
                    "cpu_time": 0.0000024160000000000002,
                    "operator_rows_scanned": 3,
                    "system_peak_buffer_memory": 0,
                    "extra_info": {
                        "Table": "memory.main.r",
                        "Type": "Sequential Scan",
                        "Projections": ["a", "b"],
                        "Filters": "a=1",
                        "Estimated Cardinality": "1",
                    },
                    "operator_timing": 0.0000024160000000000002,
                    "system_peak_temp_dir_size": 0,
                    "result_set_size": 8,
                    "operator_type": "TABLE_SCAN",
                    "children": [],
                }
            ],
        }
    ],
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
    """The ANALYZE output is a single object: an unnamed profiling root, then the `EXPLAIN_ANALYZE` operator,
    then the actual plan. Both wrappers must be skipped and the measurements of the scan must be kept.
    """
    query = parse("SELECT * FROM r WHERE r.a = 1")

    plan = parse_duckdb_plan(EXPLAIN_ANALYZE_FILTER_SCAN, query=query)

    assert plan.operator == ScanOperator.SequentialScan
    assert plan.base_table == TableReference("r")
    assert plan.actual_cardinality == 1
    assert plan.execution_time == pytest.approx(0.0000024160000000000002)


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
