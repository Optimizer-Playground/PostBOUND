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

import json
import math
import warnings
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, cast

import pytest

from postbound import ColumnReference, ScanOperator, TableReference, db, parser, transform
from postbound.db import ForeignKeyRef, Histogram, PreciseStatistics, PrimaryKeyRef
from postbound.duckdb import DuckDBDatabase, parse_duckdb_plan
from postbound.qal import SqlQuery
from tests.unit.test_duckdb import EXPLAIN_ANALYZE_FILTER_SCAN, EXPLAIN_FILTER_SCAN

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


# -- DuckDBSchema ---------------------------------------------------------------------------------------------

SCHEMA_DDL = [
    "CREATE TABLE p (a INTEGER, b INTEGER, c INTEGER, PRIMARY KEY (b, a))",
    "CREATE TABLE s (id INTEGER PRIMARY KEY, u INTEGER UNIQUE, x INTEGER, y INTEGER, UNIQUE (x, y))",
    "CREATE TABLE ch (a INTEGER, b INTEGER, k INTEGER, FOREIGN KEY (b, a) REFERENCES p (b, a), "
    "FOREIGN KEY (k) REFERENCES s (id))",
    "CREATE INDEX s_x_idx ON s (x)",
    "CREATE INDEX s_yx_idx ON s (y, x)",
]
P, S, CH = TableReference("p"), TableReference("s"), TableReference("ch")


@pytest.fixture
def schema_instance(duckdb_instance: DuckDBDatabase) -> DuckDBDatabase:
    for statement in SCHEMA_DDL:
        duckdb_instance.execute_query(statement)
    return duckdb_instance


def test_duckdb_schema_lists_tables_and_columns_in_order(schema_instance: DuckDBDatabase) -> None:
    schema = schema_instance.schema()

    assert schema.tables() == {TableReference("r"), P, S, CH}
    assert [col.name for col in schema.columns(S)] == ["id", "u", "x", "y"]


def test_duckdb_schema_primary_key_of_follows_the_key_definition_order(schema_instance: DuckDBDatabase) -> None:
    """The key is declared as ``(b, a)`` while the table defines ``a`` before ``b``."""
    schema = schema_instance.schema()

    pk = schema.primary_key_of(P)

    assert pk is not None and [col.name for col in pk.columns] == ["b", "a"]
    assert schema.primary_key_of(S) == PrimaryKeyRef([ColumnReference("id", S)])
    assert schema.primary_key_of(CH) is None


def test_duckdb_schema_is_primary_key_for_composite_keys(schema_instance: DuckDBDatabase) -> None:
    schema = schema_instance.schema()
    a, b = ColumnReference("a", P), ColumnReference("b", P)

    assert schema.is_primary_key([a, b]) is True
    assert schema.is_primary_key(a) is False
    assert schema.is_primary_key(ColumnReference("id", S)) is True


def test_duckdb_schema_is_unique_for_single_and_composite_constraints(schema_instance: DuckDBDatabase) -> None:
    schema = schema_instance.schema()
    x, y = ColumnReference("x", S), ColumnReference("y", S)

    assert schema.is_unique(ColumnReference("u", S)) is True
    assert schema.is_unique([y, x]) is True
    assert schema.is_unique(x) is False


def test_duckdb_schema_datatype_and_nullability(schema_instance: DuckDBDatabase) -> None:
    schema = schema_instance.schema()

    assert schema.datatype(ColumnReference("x", S)) == "integer"
    assert schema.is_nullable(ColumnReference("x", S)) is True
    assert schema.is_nullable(ColumnReference("id", S)) is False


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


# -- identifier casing ----------------------------------------------------------------------------------------

# DuckDB keeps the spelling of the DDL in its catalog but resolves identifiers case-insensitively. The schema mirrors
# the Stats database (``users.Id``, ``postHistory``), while the references use the lower-cased spelling that the parser
# produces for unquoted identifiers.
MIXED_CASE_DDL = [
    "CREATE TABLE users (Id INTEGER PRIMARY KEY, DisplayName VARCHAR UNIQUE, Age INTEGER)",
    "CREATE TABLE postHistory (Id INTEGER PRIMARY KEY, UserId INTEGER, FOREIGN KEY (UserId) REFERENCES users (Id))",
    "CREATE INDEX postHistory_UserId_idx ON postHistory (UserId)",
    "INSERT INTO users VALUES (1, 'a', 20), (2, 'b', NULL)",
    "INSERT INTO postHistory VALUES (1, 1), (2, 1), (3, 2)",
]
USERS, POST_HISTORY = TableReference("users"), TableReference("posthistory")
USERS_ID, USERS_AGE = ColumnReference("id", USERS), ColumnReference("age", USERS)
PH_USER_ID = ColumnReference("userid", POST_HISTORY)


@pytest.fixture
def mixed_case_instance(duckdb_instance: DuckDBDatabase) -> DuckDBDatabase:
    for statement in MIXED_CASE_DDL:
        duckdb_instance.execute_query(statement)
    return duckdb_instance


@pytest.mark.parametrize(
    ("lookup", "expected"),
    [
        (lambda d: d.schema().columns(POST_HISTORY), [ColumnReference("id", POST_HISTORY), PH_USER_ID]),
        (lambda d: d.schema().lookup_column("age", [POST_HISTORY, USERS]), USERS),
        (lambda d: d.schema().is_view(POST_HISTORY), False),
        (lambda d: d.schema().is_primary_key(USERS_ID), True),
        (lambda d: d.schema().primary_key_of(USERS), PrimaryKeyRef([USERS_ID])),
        (lambda d: d.schema().is_unique(ColumnReference("displayname", USERS)), True),
        (lambda d: d.schema().datatype(ColumnReference("displayname", USERS)), "varchar"),
        (lambda d: d.schema().is_nullable(USERS_ID), False),
        (lambda d: d.schema().foreign_keys_on(PH_USER_ID), {ForeignKeyRef([PH_USER_ID], [USERS_ID])}),
        (lambda d: d.schema().indexes_on(PH_USER_ID), {"postHistory_UserId_idx", "postHistory_userid_id_fkey"}),
        (lambda d: d.schema().has_secondary_index(PH_USER_ID), True),
        (lambda d: d.schema()[USERS_ID].primary_key, True),
        (lambda d: d.schema().as_graph().has_edge(POST_HISTORY, USERS), True),
        (lambda d: d.schema().join_equivalence_keys()[PH_USER_ID], {PH_USER_ID, USERS_ID}),
        (lambda d: d.statistics().total_rows(POST_HISTORY), 3),
        (lambda d: d.statistics().num_distinct(PH_USER_ID), 2),
        (lambda d: d.statistics().null_frac(USERS_AGE), 0.5),
        (lambda d: d.statistics().min_max(USERS_AGE), (20, 20)),
        (lambda d: list(d.statistics().most_common_values(PH_USER_ID)), [(1, 2), (2, 1)]),
    ],
    ids=[
        "columns",
        "lookup_column",
        "is_view",
        "is_primary_key",
        "primary_key_of",
        "is_unique",
        "datatype",
        "is_nullable",
        "foreign_keys_on",
        "indexes_on",
        "has_secondary_index",
        "column_info",
        "as_graph",
        "join_equivalence_keys",
        "total_rows",
        "num_distinct",
        "null_frac",
        "min_max",
        "most_common_values",
    ],
)
def test_duckdb_lookups_ignore_the_case_of_identifiers(
    mixed_case_instance: DuckDBDatabase,
    monkeypatch: pytest.MonkeyPatch,
    lookup: Callable[[DuckDBDatabase], object],
    expected: object,
) -> None:
    """Every schema and statistics lookup accepts the lower-cased spelling of a mixed-case DuckDB catalog.

    The cases that used to fail have dedicated regression tests (tier 0 for the comparisons in the generic
    `DatabaseSchema`, below for the SQL of `DuckDBSchema`); this guard checks the complete surface end to end.
    """
    monkeypatch.setattr(db, "enable_emulation_fallback", True)

    assert lookup(mixed_case_instance) == expected


# -- plan format drift ----------------------------------------------------------------------------------------

# The plan parser is tested offline against captured EXPLAIN output (`tests/unit/test_duckdb.py`). These tests run it
# against the live engine instead, so that a DuckDB/quacklab upgrade that changes the JSON format fails here rather
# than silently producing plans without measurements. They use only operators that `parse_duckdb_plan` knows, so an
# "Unknown node type" warning signals a new wrapper or a renamed operator.
DRIFT_QUERIES = [
    "SELECT * FROM r WHERE r.a = 1",
    "SELECT * FROM r JOIN s ON r.a = s.a",
    "SELECT count(*) FROM r JOIN s ON r.a = s.a WHERE r.b < 30",
    "SELECT r.a, count(*) FROM r GROUP BY r.a",
]
DRIFT_IDS = ["filter-scan", "join", "aggregate-over-join", "group-by"]


@pytest.fixture
def plan_instance(duckdb_instance: DuckDBDatabase) -> DuckDBDatabase:
    """Extends `r` by a second table `s(a)` that joins with two of its rows."""
    duckdb_instance.execute_query("CREATE TABLE s (a INTEGER)")
    duckdb_instance.execute_query("INSERT INTO s VALUES (1), (2)")
    return duckdb_instance


def json_shape(value: object) -> object:
    """The structure of a JSON value -- its keys and nesting -- with every scalar replaced by its kind."""
    match value:
        case dict():
            return {key: json_shape(item) for key, item in value.items()}
        case list():
            return [json_shape(item) for item in value]
        case bool():
            return "bool"
        case int() | float():
            return "number"
        case _:
            return type(value).__name__


@pytest.mark.parametrize("sql", DRIFT_QUERIES, ids=DRIFT_IDS)
def test_query_plan_parses_the_live_explain_format(plan_instance: DuckDBDatabase, sql: str) -> None:
    query = parse(sql)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        plan = plan_instance.optimizer().query_plan(query)

    assert plan.tables() == query.tables()
    assert all(node.estimated_cardinality.is_valid() for node in plan.iternodes())


@pytest.mark.parametrize("sql", DRIFT_QUERIES, ids=DRIFT_IDS)
def test_analyze_plan_parses_the_live_profiling_format(plan_instance: DuckDBDatabase, sql: str) -> None:
    """Every operator carries its measurements, and the root produces exactly the rows of the query result."""
    query = parse(sql)
    n_result_rows = len(plan_instance.execute_query(query, raw=True))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        plan = plan_instance.optimizer().analyze_plan(query)

    assert plan.tables() == query.tables()
    assert plan.actual_cardinality == n_result_rows
    assert all(node.actual_cardinality.is_valid() for node in plan.iternodes())
    assert all(math.isfinite(node.execution_time) for node in plan.iternodes())


EXPLAIN_RESULT_SETS = [
    (transform.as_explain, True),
    (transform.as_explain, False),
    (transform.as_explain_analyze, True),
    (transform.as_explain_analyze, False),
]
EXPLAIN_RESULT_SET_IDS = ["explain-raw", "explain-simplified", "analyze-raw", "analyze-simplified"]


@pytest.mark.parametrize(("explain", "raw"), EXPLAIN_RESULT_SETS, ids=EXPLAIN_RESULT_SET_IDS)
def test_parse_duckdb_plan_accepts_the_result_set_of_execute_query(
    duckdb_instance: DuckDBDatabase, explain: Callable[[SqlQuery], SqlQuery], raw: bool
) -> None:
    """The workflow of the DuckDB docs: the result set of an EXPLAIN query is passed to the parser as-is."""
    result_set = duckdb_instance.execute_query(explain(FILTER_QUERY), raw=raw)

    plan = parse_duckdb_plan(result_set, query=FILTER_QUERY)

    assert plan == duckdb_instance.optimizer().query_plan(FILTER_QUERY)
    assert plan.actual_cardinality.is_valid() is (explain is transform.as_explain_analyze)


def test_offline_explain_fixtures_match_the_live_format(duckdb_instance: DuckDBDatabase) -> None:
    """The fixtures of `tests/unit/test_duckdb.py` still have the structure of the live EXPLAIN output.

    If this fails after a DuckDB/quacklab upgrade, re-capture the fixtures (see the docstring of that module) and
    check whether `parse_duckdb_plan` needs to follow. For the ANALYZE output, only the top-level sections of the
    profiling summary are compared: the timings within them (e.g. one entry per optimizer pass) change with every
    release but are irrelevant for parsing.
    """
    raw_explain = duckdb_instance.execute_query(transform.as_explain(FILTER_QUERY), raw=True)[0][1]
    raw_analyze = duckdb_instance.execute_query(transform.as_explain_analyze(FILTER_QUERY), raw=True)[0][1]

    live_explain, live_analyze = json.loads(raw_explain), json.loads(raw_analyze)

    assert json_shape(live_explain) == json_shape(EXPLAIN_FILTER_SCAN)
    assert set(live_analyze) == set(EXPLAIN_ANALYZE_FILTER_SCAN)
    assert json_shape(live_analyze["operator"]) == json_shape(EXPLAIN_ANALYZE_FILTER_SCAN["operator"])


# -- regression tests --------------------------------------------------------------------------------------------


def test_duckdb_schema_all_foreign_keys_on_resolves_composite_and_simple_keys(schema_instance: DuckDBDatabase) -> None:
    """Regression guard for the composite key rework: the generic implementation relied on Postgres semantics of
    ``constraint_column_usage`` and reported every key of ``ch`` as referencing ``ch`` itself. The foreign key is
    declared as ``(b, a)`` while ``p`` defines ``a`` before ``b``, so this also checks the column pairing.
    """
    fks = schema_instance.schema().all_foreign_keys_on(CH)

    assert {fk.referenced_table for fk in fks} == {P, S}
    composite = next(fk for fk in fks if fk.referenced_table == P)
    assert dict(composite.mapping()) == {
        ColumnReference("b", CH): ColumnReference("b", P),
        ColumnReference("a", CH): ColumnReference("a", P),
    }
    simple = next(fk for fk in fks if fk.referenced_table == S)
    assert simple == ForeignKeyRef([ColumnReference("k", CH)], [ColumnReference("id", S)])


def test_duckdb_schema_foreign_keys_on_selects_by_the_exact_column_set(schema_instance: DuckDBDatabase) -> None:
    schema = schema_instance.schema()
    a, b, k = (ColumnReference(name, CH) for name in "abk")

    assert {fk.referenced_table for fk in schema.foreign_keys_on([a, b])} == {P}
    assert {fk.referenced_table for fk in schema.foreign_keys_on(k)} == {S}
    assert schema.foreign_keys_on(a) == set()
    assert schema.foreign_keys_on(ColumnReference("k", TableReference("ch", "c"))) == schema.foreign_keys_on(k)


def test_duckdb_schema_has_no_foreign_keys_for_tables_without_any(schema_instance: DuckDBDatabase) -> None:
    assert schema_instance.schema().all_foreign_keys_on(S) == set()


def test_duckdb_schema_as_graph_contains_composite_foreign_keys(schema_instance: DuckDBDatabase) -> None:
    graph = schema_instance.schema().as_graph()

    assert {tuple(sorted(col.name for col in fk.fk_cols)) for fk in graph.edges[CH, P]["foreign_keys"]} == {("a", "b")}
    assert graph.has_edge(CH, S)


def test_duckdb_schema_join_equivalence_follows_simple_foreign_keys_only(schema_instance: DuckDBDatabase) -> None:
    classes = schema_instance.schema().join_equivalence_keys()

    assert classes[ColumnReference("k", CH)] == {ColumnReference("k", CH), ColumnReference("id", S)}
    assert ColumnReference("a", CH) not in classes


def test_duckdb_schema_all_indexes_on_lists_constraint_backed_and_explicit_indexes(
    schema_instance: DuckDBDatabase,
) -> None:
    """Regression guard for the composite key rework: the statement filtered the table functions by columns that do
    not exist (``catalog_name``, and a ``constraint_type`` on ``duckdb_indexes()``) and was rejected by DuckDB.
    """
    indexes = schema_instance.schema().all_indexes_on(S)

    assert indexes["s_id_pkey"] == {ColumnReference("id", S)}
    assert indexes["s_u_key"] == {ColumnReference("u", S)}
    assert indexes["s_x_y_key"] == {ColumnReference("x", S), ColumnReference("y", S)}
    assert indexes["s_x_idx"] == {ColumnReference("x", S)}
    assert indexes["s_yx_idx"] == {ColumnReference("x", S), ColumnReference("y", S)}


def test_duckdb_schema_all_indexes_on_does_not_report_not_null_constraints(schema_instance: DuckDBDatabase) -> None:
    """Regression guard for the composite key rework: ``duckdb_constraints()`` also lists NOT NULL and CHECK
    constraints, which made every NOT NULL column look indexed.
    """
    schema = schema_instance.schema()

    assert set(schema.all_indexes_on(P)) == {"p_b_a_pkey"}
    assert schema.has_secondary_index(ColumnReference("a", P)) is False
    assert schema.has_secondary_index(ColumnReference("x", S)) is True


def test_duckdb_schema_all_indexes_on_lists_foreign_keys_on_their_own_columns(schema_instance: DuckDBDatabase) -> None:
    indexes = schema_instance.schema().all_indexes_on(CH)

    assert sorted(sorted(col.name for col in cols) for cols in indexes.values()) == [["a", "b"], ["k"]]


def test_duckdb_schema_indexes_on_matches_composite_indexes(schema_instance: DuckDBDatabase) -> None:
    schema = schema_instance.schema()
    x, y = ColumnReference("x", S), ColumnReference("y", S)

    assert schema.indexes_on(x) == {"s_x_idx"}
    assert schema.indexes_on([y, x]) == {"s_yx_idx", "s_x_y_key"}  # the explicit index and the UNIQUE constraint


def test_duckdb_schema_table_info_combines_keys_and_indexes(schema_instance: DuckDBDatabase) -> None:
    info = schema_instance.schema()[CH]

    assert info.primary_key is None
    assert {fk.referenced_table for fk in info.outgoing_foreign_keys} == {P, S}
    assert schema_instance.schema()[P].incoming_foreign_keys == {
        fk for fk in info.outgoing_foreign_keys if fk.referenced_table == P
    }
    assert info["k"].indexed is True  # foreign keys count as indexed, like in the generic implementation


def test_duckdb_database_describe_lists_keys_and_indexes_of_every_table(schema_instance: DuckDBDatabase) -> None:
    """Regression guard for the composite key rework: `describe()` failed with the binder errors of the schema lookups."""
    description = schema_instance.describe()

    schema_info = cast(dict[str, Any], description)["schema_info"]
    tables = {entry["table"]: entry for entry in schema_info}
    assert set(tables) == {"r", "p", "s", "ch"}
    assert tables["s"]["primary_key"] is not None
    assert tables["ch"]["foreign_keys"]
    assert tables["s"]["indexes"]


def test_duckdb_schema_all_foreign_keys_on_finds_mixed_case_tables(mixed_case_instance: DuckDBDatabase) -> None:
    """Regression guard for the DuckDB identifier casing bug: ``duckdb_constraints()`` was filtered with
    ``table_name = ?``, which compares the catalog spelling (``postHistory``) to the lower-cased name of a parsed query
    as raw strings. The table therefore seemed to have no foreign keys.
    """
    fks = mixed_case_instance.schema().all_foreign_keys_on(POST_HISTORY)

    assert fks == {ForeignKeyRef([PH_USER_ID], [USERS_ID])}


def test_duckdb_schema_all_indexes_on_finds_mixed_case_tables(mixed_case_instance: DuckDBDatabase) -> None:
    """Regression guard for the DuckDB identifier casing bug: see
    `test_duckdb_schema_all_foreign_keys_on_finds_mixed_case_tables`. Both ``duckdb_constraints()`` and
    ``duckdb_indexes()`` were filtered by the raw table name, so the table seemed to have no indexes.
    """
    indexes = mixed_case_instance.schema().all_indexes_on(POST_HISTORY)

    assert {name: cols for name, cols in indexes.items()} == {
        "postHistory_id_pkey": {ColumnReference("id", POST_HISTORY)},
        "postHistory_userid_id_fkey": {PH_USER_ID},
        "postHistory_UserId_idx": {PH_USER_ID},
    }


def test_analyze_plan_reports_the_measurements_of_the_quacklab_2_profiling_format(
    duckdb_instance: DuckDBDatabase,
) -> None:
    """Regression guard for the quacklab 2 profiling format: since quacklab 2.0, ``EXPLAIN (ANALYZE, FORMAT JSON)``
    nests the operator tree below the ``operator`` key of a profiling summary, starts it with a ``RESULT_COLLECTOR``
    operator, and renamed ``operator_cardinality``/``operator_timing`` to ``intermediate_rows``/``timing``.
    `parse_duckdb_plan` looked for the plan below ``children`` and failed with a `KeyError`. Unwrapping alone was not
    enough: the parser would have read the old keys and silently reported *NaN* for every measurement.
    """
    plan = duckdb_instance.optimizer().analyze_plan(FILTER_QUERY)

    assert plan.operator == ScanOperator.SequentialScan
    assert plan.actual_cardinality == 1
    assert math.isfinite(plan.execution_time) and plan.execution_time > 0


@pytest.mark.parametrize(("explain", "raw"), EXPLAIN_RESULT_SETS, ids=EXPLAIN_RESULT_SET_IDS)
def test_parse_plan_accepts_the_result_set_of_an_explain_query(
    duckdb_instance: DuckDBDatabase, explain: Callable[[SqlQuery], SqlQuery], raw: bool
) -> None:
    """Regression guard for `DuckDBOptimizer.parse_plan` rejecting every real result set: it unwrapped the result set
    down to the JSON column, which is a *string*, and then gave up because it expected a parsed dictionary. Every
    call therefore warned "Could not parse plan" and returned *None*.
    """
    result_set = duckdb_instance.execute_query(explain(FILTER_QUERY), raw=raw)

    plan = duckdb_instance.optimizer().parse_plan(result_set, query=FILTER_QUERY)

    assert plan == duckdb_instance.optimizer().query_plan(FILTER_QUERY)
