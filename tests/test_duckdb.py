"""Tests for `postbound.duckdb` -- the DuckDB backend, run against a real in-memory DuckDB (via quacklab).

All tests here are tier 1 (`embedded`): they need a real DuckDB engine, but no server or database file, because
the behaviours under test are about whether DuckDB *accepts* the SQL that PostBOUND generates. A double would
share the backend's assumptions about DuckDB's dialect and could only confirm a bug, not catch it. The synthetic
schema is created per test, so nothing is downloaded or written to disk.

`PreciseStatistics.histogram` is run here as well, because its SQL and the shape of the result set (in particular
how NULLs are grouped and sorted) are decided by the engine. Its decision logic is covered offline in
`tests/unit/test_db_stats.py`, which scripts the result sets observed here.

The pure formatting part of the EXPLAIN regression (the *duckdb* flavor of `format_quick`) is covered offline in
`tests/unit/test_qal_formatter.py`; the parsing of EXPLAIN output and the scan-to-table binding in
`tests/unit/test_duckdb.py`.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from postbound import ColumnReference, ScanOperator, TableReference, parser, transform
from postbound.db import Histogram, PreciseStatistics
from postbound.duckdb import DuckDBDatabase

pytestmark = pytest.mark.embedded

# -- fixtures -----------------------------------------------------------------------------------------------


def parse(sql: str):
    return parser.parse_query(sql, bind_columns=False)


FILTER_QUERY = parse("SELECT * FROM r WHERE r.a = 1")


@pytest.fixture
def duckdb_instance() -> Iterator[DuckDBDatabase]:
    """An in-memory DuckDB with a tiny table `r(a, b)`. The constructor does not register it on the pool."""
    instance = DuckDBDatabase(Path(":memory:"))
    instance.execute_query("CREATE TABLE r (a INTEGER, b INTEGER)")
    instance.execute_query("INSERT INTO r VALUES (1, 10), (2, 20), (3, 30)")
    yield instance
    instance.close()


def create_column(instance: DuckDBDatabase, values: list[int | None]) -> ColumnReference:
    """Fills a fresh table `h(v)` with `values` and returns the reference to its column."""
    instance.execute_query("CREATE TABLE h (v INTEGER)")
    rows = ", ".join("(NULL)" if value is None else f"({value})" for value in values)
    instance.execute_query(f"INSERT INTO h VALUES {rows}")
    return ColumnReference("v", TableReference("h"))


# -- PreciseStatistics.histogram ----------------------------------------------------------------------------


def test_precise_histogram_builds_equi_depth_buckets_on_duckdb(duckdb_instance: DuckDBDatabase) -> None:
    column = create_column(duckdb_instance, [1, 2, 3, 4, 5, 6, 7, 8])

    hist = PreciseStatistics(duckdb_instance).histogram(column, n_bins=4)

    assert hist == Histogram([2, 4, 6, 8], [2, 2, 2, 2], lower=1, bucket_interpolation="approx-uni")


# -- regression tests --------------------------------------------------------------------------------------


def test_hint_service_formats_explain_queries_without_postgres_only_options(duckdb_instance: DuckDBDatabase) -> None:
    """Regression guard for the v0.22.1 DuckDB EXPLAIN fix: `DuckDBHintService.format_query` used to call
    `format_quick(..., flavor="postgres")`. Since v0.22.0 that flavor writes `EXPLAIN (SETTINGS, SUMMARY,
    VERBOSE, FORMAT JSON)`, which DuckDB does not support, so the backend could not run any EXPLAIN query.
    """
    explain_query = transform.as_explain(FILTER_QUERY)

    formatted = duckdb_instance.hinting().format_query(explain_query)

    header = formatted.splitlines()[0]
    assert header == "EXPLAIN (FORMAT JSON)"


def test_query_plan_of_a_parsed_query_runs_on_duckdb(duckdb_instance: DuckDBDatabase) -> None:
    """Regression guard for the v0.22.1 DuckDB EXPLAIN fix: `DuckDBOptimizer.query_plan` wraps a `SqlQuery` in an
    EXPLAIN and formats it through the hint service, which used the Postgres flavor and thereby emitted the
    Postgres-only `SETTINGS`/`SUMMARY`/`VERBOSE` options. DuckDB rejected the statement (`NotImplementedException: Unimplemented explain type: settings`)
    before planning even started. (Plain-string queries took a separate hand-written `EXPLAIN (FORMAT JSON)` path
    and were unaffected, which is why this test passes a parsed query.)
    """
    plan = duckdb_instance.optimizer().query_plan(FILTER_QUERY)

    assert plan.operator == ScanOperator.SequentialScan


def test_execute_query_runs_a_parsed_explain_query_on_duckdb(duckdb_instance: DuckDBDatabase) -> None:
    """Regression guard for the v0.22.1 DuckDB EXPLAIN fix, on the `DuckDBDatabase.execute_query` path: it formats
    every `SqlQuery` through the hint service as well, so executing an `EXPLAIN` / `EXPLAIN ANALYZE` query object
    directly failed with the same `NotImplementedException` as `query_plan`.
    """
    explain_queries = [transform.as_explain(FILTER_QUERY), transform.as_explain_analyze(FILTER_QUERY)]

    result_sets = [duckdb_instance.execute_query(query, raw=True) for query in explain_queries]

    assert [result_set[0][0] for result_set in result_sets] == ["physical_plan", "analyzed_plan"]


def test_analyze_plan_returns_the_measured_plan(duckdb_instance: DuckDBDatabase) -> None:
    """Regression guard for the v0.22.1 DuckDB `analyze_plan` fix: `DuckDBOptimizer.analyze_plan` passed `parsed[0]`
    to `parse_duckdb_plan`, assuming the JSON *list* that plain `EXPLAIN (FORMAT JSON)` returns. With ANALYZE,
    DuckDB returns a single JSON *object* (a profiling root with the plan below `children`), so indexing it with `0`
    raised a `KeyError` and no query could be analyzed.
    """
    plan = duckdb_instance.optimizer().analyze_plan(FILTER_QUERY)

    assert plan.operator == ScanOperator.SequentialScan
    assert plan.actual_cardinality == 1


def test_query_plan_binds_scan_nodes_to_their_tables(duckdb_instance: DuckDBDatabase) -> None:
    """Regression guard for the v0.22.1 DuckDB table-binding fix, end to end: `_bind_node_to_table` compared the
    `Table` entry of a scan node with `TableReference.full_name` by plain string equality, but DuckDB reports fully
    qualified names (`memory.main.r` here) while the query references just `r`. Every scan node stayed unbound, so
    `query_plan` returned plans without any tables. The offline tests in `tests/unit/test_duckdb.py` pin the captured
    output; this one guards against DuckDB changing the reported name format underneath them.
    """
    plan = duckdb_instance.optimizer().query_plan(FILTER_QUERY)

    assert plan.base_table == TableReference("r")
    assert plan.tables() == {TableReference("r")}


def test_precise_histogram_excludes_nulls_on_duckdb(duckdb_instance: DuckDBDatabase) -> None:
    """Regression guard for the unreleased histogram fixes: DuckDB returns the NULL group of the histogram's GROUP BY
    query last as ``(None, 0)``. With fewer rows than bins, `_infer_histogram_bounds` turned it into the upper bound of
    the histogram, and every estimate at or above the largest value raised a `TypeError` comparing with *None*.
    """
    column = create_column(duckdb_instance, [1, 1, 2, 3, None, None])

    hist = PreciseStatistics(duckdb_instance).histogram(column)

    assert list(hist) == [(1, 2), (2, 1), (3, 1)]
    assert hist.frequency_below(5) == 4


def test_precise_histogram_of_a_single_distinct_value_on_duckdb(duckdb_instance: DuckDBDatabase) -> None:
    """Regression guard for the unreleased histogram fixes: the one-row result set was simplified to a plain tuple
    before the bounds were inferred, which crashed. See `test_histogram_of_a_column_with_a_single_distinct_value` in
    `tests/unit/test_db_stats.py`.
    """
    column = create_column(duckdb_instance, [7, 7])

    hist = PreciseStatistics(duckdb_instance).histogram(column)

    assert list(hist) == [(7, 2)]
