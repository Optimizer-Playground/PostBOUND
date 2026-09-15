"""Tests for the test doubles themselves.

The doubles are load-bearing: once real tests depend on them, a silently wrong double produces confidently
wrong results everywhere. These tests pin the two properties that matter most -- that the doubles refuse
what they were not told about, and that `StaticSchema` really does run the production `DatabaseSchema`
composition rather than replacing it.
"""

from __future__ import annotations

import pytest

from postbound import Cardinality
from postbound._core import ColumnReference, TableReference
from postbound.db import Database, DatabaseSchema
from tests.doubles import (
    FakeDatabase,
    FakeStatistics,
    ScriptedCursor,
    StaticSchema,
    UnexpectedQueryError,
    column_spec,
)

TITLE = TableReference("title")
MOVIE_INFO = TableReference("movie_info")

SCHEMA_SPEC = {
    "title": [
        column_spec("id", "integer", nullable=False, primary_key=True),
        column_spec("title", "text"),
        column_spec("production_year", "integer", indexed=True),
    ],
    "movie_info": [
        column_spec("id", "integer", nullable=False, primary_key=True),
        column_spec("movie_id", "integer", references="title.id"),
        column_spec("info", "text"),
    ],
}


# -- ScriptedCursor ----------------------------------------------------------------------------------


def test_scripted_cursor_returns_scripted_rows() -> None:
    cursor = ScriptedCursor({"information_schema.tables": [("title",), ("movie_info",)]})

    cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = %s", ["public"])

    assert cursor.fetchall() == [("title",), ("movie_info",)]


def test_scripted_cursor_records_sql_and_parameters() -> None:
    cursor = ScriptedCursor({"information_schema.columns": []})

    cursor.execute("SELECT column_name FROM information_schema.columns WHERE table_name = %s", ["title"])

    assert len(cursor.calls) == 1
    assert cursor.calls[0].parameters == ("title",)
    assert cursor.calls[0].mentions("information_schema.columns", "table_name")


def test_scripted_cursor_matching_ignores_formatting() -> None:
    """Whitespace and case must not affect matching -- the production SQL is textwrap.dedent-ed."""
    cursor = ScriptedCursor({"FROM information_schema.TABLES": [("title",)]})

    cursor.execute("SELECT   table_name\n  FROM\n    information_schema.tables\n")

    assert cursor.fetchall() == [("title",)]


def test_scripted_cursor_rejects_unscripted_query() -> None:
    """An unknown query must raise, not return empty.

    Returning [] would let a test assert on silence and pass for the wrong reason.
    """
    cursor = ScriptedCursor({"information_schema.tables": []})

    with pytest.raises(UnexpectedQueryError, match="unscripted query"):
        cursor.execute("SELECT * FROM pg_stats")


def test_scripted_cursor_non_strict_returns_empty() -> None:
    cursor = ScriptedCursor(strict=False)

    cursor.execute("SELECT * FROM anything")

    assert cursor.fetchall() == []


def test_scripted_cursor_fetchone_returns_first_row() -> None:
    cursor = ScriptedCursor({"pg_class": [("42",), ("43",)]})

    cursor.execute("SELECT reltuples FROM pg_class")

    assert cursor.fetchone() == ("42",)


def test_scripted_cursor_fetchone_on_empty_result() -> None:
    cursor = ScriptedCursor({"pg_class": []})

    cursor.execute("SELECT reltuples FROM pg_class")

    assert cursor.fetchone() is None


# -- StaticSchema ------------------------------------------------------------------------------------


@pytest.fixture
def schema() -> StaticSchema:
    return StaticSchema(SCHEMA_SPEC)


def test_static_schema_reports_tables(schema: StaticSchema) -> None:
    assert schema.tables() == {TITLE, MOVIE_INFO}


def test_static_schema_reports_columns(schema: StaticSchema) -> None:
    assert sorted(col.name for col in schema.columns(TITLE)) == ["id", "production_year", "title"]


@pytest.mark.parametrize(
    ("table", "column", "expected"),
    [
        ("title", "id", True),
        ("title", "production_year", False),
        ("movie_info", "id", True),
        ("movie_info", "movie_id", False),
    ],
)
def test_static_schema_primary_keys(schema: StaticSchema, table: str, column: str, expected: bool) -> None:
    assert schema.is_primary_key(ColumnReference(column, TableReference(table))) is expected


def test_static_schema_has_index_composes_production_logic(schema: StaticSchema) -> None:
    """`has_index` is *not* overridden -- it is the real implementation over the two leaf lookups.

    This is the point of the double: production composition stays under test instead of being replaced.
    """
    assert "has_index" not in vars(StaticSchema), "has_index must stay inherited from DatabaseSchema"

    assert schema.has_index(ColumnReference("id", TITLE)) is True  # via is_primary_key
    assert schema.has_index(ColumnReference("production_year", TITLE)) is True  # via has_secondary_index
    assert schema.has_index(ColumnReference("title", TITLE)) is False


def test_static_schema_lookup_column_composes_production_logic(schema: StaticSchema) -> None:
    """`lookup_column` is inherited and resolves through `columns()`."""
    assert "lookup_column" not in vars(StaticSchema)

    resolved = schema.lookup_column(ColumnReference("production_year", TITLE), [TITLE, MOVIE_INFO])

    assert resolved == TITLE


def test_base_lookup_column_resolves_an_unqualified_column(schema: StaticSchema) -> None:
    """`lookup_column` fetches the first matching table from the leaf lookups."""
    assert schema.lookup_column("production_year", [TITLE, MOVIE_INFO]) is TITLE
    assert schema.lookup_column(ColumnReference("production_year"), [TITLE, MOVIE_INFO]) is TITLE
    assert schema.lookup_column("production_year", [TITLE], expect_match=True) is TITLE


def test_static_schema_table_info_composes_production_logic(schema: StaticSchema) -> None:
    """``schema[table]`` builds a `TableInfo` purely from the leaf lookups."""
    info = schema[TITLE]

    assert info.table == TITLE
    assert info.primary_key == ColumnReference("id", TITLE)
    assert {col.column.name for col in info.columns} == {"id", "title", "production_year"}
    assert info["title"].datatype == "text"
    assert info["id"].nullable is False


def test_static_schema_foreign_keys(schema: StaticSchema) -> None:
    fks = schema.foreign_keys_on(ColumnReference("movie_id", MOVIE_INFO))

    assert fks == {ColumnReference("id", TITLE)}


def test_static_schema_is_iterable_as_mapping(schema: StaticSchema) -> None:
    """`DatabaseSchema` is a Mapping; iteration must yield the tables."""
    assert set(schema) == {TITLE, MOVIE_INFO}
    assert len(schema) == 2


def test_static_schema_rejects_unknown_column(schema: StaticSchema) -> None:
    with pytest.raises(KeyError):
        schema.datatype(ColumnReference("nonexistent", TITLE))


# -- FakeDatabase ------------------------------------------------------------------------------------


def test_fake_database_satisfies_the_abc() -> None:
    """Instantiating proves every abstract member is implemented.

    This is the reason the doubles are hand-written rather than MagicMock: adding an abstract method to
    `Database` breaks this test immediately instead of silently passing.
    """
    instance = FakeDatabase()

    assert isinstance(instance, Database)
    assert isinstance(instance.schema(), DatabaseSchema)


def test_fake_database_returns_scripted_result() -> None:
    database = FakeDatabase(results={"count(*)": [(42,)]})

    assert database.execute_query("SELECT count(*) FROM title") == 42


def test_fake_database_raw_result_is_unsimplified() -> None:
    database = FakeDatabase(results={"count(*)": [(42,)]})

    assert database.execute_query("SELECT count(*) FROM title", raw=True) == [(42,)]


def test_fake_database_records_executed_queries() -> None:
    database = FakeDatabase(results={"count(*)": [(1,)]})

    database.execute_query("SELECT  count(*)\nFROM title")

    assert database.executed_queries == ["select count(*) from title"]


def test_fake_database_rejects_unscripted_query() -> None:
    database = FakeDatabase()

    with pytest.raises(UnexpectedQueryError, match="unscripted query"):
        database.execute_query("SELECT * FROM title")


def test_fake_database_expect_result_is_chainable() -> None:
    database = FakeDatabase().expect_result("min(id)", [(1,)]).expect_result("max(id)", [(9,)])

    assert database.execute_query("SELECT min(id) FROM title") == 1
    assert database.execute_query("SELECT max(id) FROM title") == 9


def test_fake_database_tracks_lifecycle() -> None:
    database = FakeDatabase()

    database.reset_connection()
    database.reset_connection()
    database.close()

    assert database.reset_count == 2
    assert database.closed is True


def test_fake_database_exposes_static_schema() -> None:
    database = FakeDatabase(schema_spec=SCHEMA_SPEC)

    assert database.schema().tables() == {TITLE, MOVIE_INFO}


# -- FakeStatistics ----------------------------------------------------------------------------------


def test_fake_statistics_returns_configured_values() -> None:
    stats = FakeStatistics(
        total_rows={"title": 2_528_312},
        num_distinct={"title.production_year": 133},
        null_frac={"title.production_year": 0.02},
    )

    assert stats.total_rows(TITLE) == Cardinality(2_528_312)
    assert stats.num_distinct(ColumnReference("production_year", TITLE)) == 133
    assert stats.null_frac(ColumnReference("production_year", TITLE)) == pytest.approx(0.02)


def test_fake_statistics_returns_none_for_unknown() -> None:
    """*None* is the interface's "maintained but no value" signal, not an error."""
    stats = FakeStatistics()

    assert stats.total_rows(TITLE) is None
    assert stats.min_max(ColumnReference("id", TITLE)) is None
    assert stats.histogram(ColumnReference("id", TITLE)) is None


# -- pool isolation ----------------------------------------------------------------------------------


def test_fake_db_fixture_registers_current_database(fake_db: FakeDatabase) -> None:
    from postbound.db import DatabasePool

    assert DatabasePool.get_instance().current_database() is fake_db


def test_pool_is_clean_between_tests() -> None:
    """The previous test registered a database; it must not still be visible here.

    Pool leakage is not hypothetical -- it previously made `test_relalg` bind columns against whichever
    database an earlier module happened to leave behind.
    """
    from postbound.db import DatabasePool

    assert DatabasePool.get_instance().empty()
