"""Tests for the generic `DatabaseSchema` implementation, driven by a `ScriptedCursor`.

`DatabaseSchema` declares no abstract methods: all of its lookups are default implementations that issue
*information_schema* SQL through ``self._db.cursor()``. Roughly 900 lines of production code therefore run
here without a server -- and because the cursor records what it was asked, these tests assert on the
generated SQL as well as on the interpretation of the rows.

That second half matters: it is what catches placeholder bugs (``%s`` versus DuckDB's ``?``) and mismatched
parameter counts, which are invisible to a test that only checks the return value.
"""

from __future__ import annotations

import pytest

from postbound._core import ColumnReference, TableReference
from postbound.db import Database, DatabaseSchema
from tests.doubles import FakeDatabase, ScriptedCursor

TITLE = TableReference("title")


def schema_over(cursor: ScriptedCursor, *, placeholder: str = "%s") -> DatabaseSchema:
    """Builds a stock `DatabaseSchema` whose cursor is the given double."""
    database: Database = FakeDatabase(cursor_double=cursor)
    return DatabaseSchema(database, prep_placeholder=placeholder)


# -- tables() ----------------------------------------------------------------------------------------


def test_tables_reads_information_schema() -> None:
    cursor = ScriptedCursor({"information_schema.tables": [("title",), ("movie_info",)]})

    tables = schema_over(cursor).tables()

    assert tables == {TableReference("title"), TableReference("movie_info")}
    assert cursor.calls[0].mentions("information_schema.tables")


def test_tables_defaults_to_current_catalog_and_schema() -> None:
    """With no catalog/schema given, the SQL must use the server-side functions and bind no parameters."""
    cursor = ScriptedCursor({"information_schema.tables": []})

    schema_over(cursor).tables()

    call = cursor.calls[0]
    assert call.mentions("current_database()", "current_schema()")
    assert call.parameters == ()


def test_tables_binds_explicit_catalog_and_schema() -> None:
    cursor = ScriptedCursor({"information_schema.tables": []})

    schema_over(cursor).tables(catalog="imdb", schema="public")

    call = cursor.calls[0]
    assert call.parameters == ("imdb", "public")
    assert not call.mentions("current_database()")


def test_tables_binds_only_the_supplied_qualifier() -> None:
    """A parameter list that drifts out of step with the placeholders is the classic failure here."""
    cursor = ScriptedCursor({"information_schema.tables": []})

    schema_over(cursor).tables(schema="public")

    call = cursor.calls[0]
    assert call.parameters == ("public",)
    assert call.mentions("current_database()")
    assert call.normalized.count("%s") == len(call.parameters)


# -- placeholder dialects ----------------------------------------------------------------------------


@pytest.mark.parametrize("placeholder", ["%s", "?"])
def test_placeholder_dialect_is_respected(placeholder: str) -> None:
    """DuckDB binds with ``?`` while Postgres and MySQL use ``%s``.

    The count of placeholders must always match the number of bound parameters, whichever dialect is in use.
    """
    cursor = ScriptedCursor({"information_schema.tables": []})

    schema_over(cursor, placeholder=placeholder).tables(catalog="imdb", schema="public")

    call = cursor.calls[0]
    assert call.normalized.count(placeholder) == len(call.parameters) == 2


# -- columns() ---------------------------------------------------------------------------------------


def test_columns_returns_bound_columns_in_order() -> None:
    cursor = ScriptedCursor({"information_schema.columns": [("id",), ("title",), ("production_year",)]})

    columns = schema_over(cursor).columns(TITLE)

    assert [col.name for col in columns] == ["id", "title", "production_year"]
    assert all(col.table == TITLE for col in columns)


def test_columns_of_unknown_table_is_empty() -> None:
    cursor = ScriptedCursor({"information_schema.columns": []})

    assert schema_over(cursor).columns(TableReference("nope")) == []


def test_columns_rejects_virtual_tables() -> None:
    # Exported from the top-level package; the DatabaseSchema.columns docstring calls it
    # `postbound.qal.VirtualTableError`, which is not where it actually lives.
    from postbound import VirtualTableError

    cursor = ScriptedCursor({"information_schema.columns": []})
    virtual = TableReference.create_virtual("sq")

    with pytest.raises(VirtualTableError):
        schema_over(cursor).columns(virtual)


# -- index lookups -----------------------------------------------------------------------------------


def test_is_primary_key_true_when_constraint_present() -> None:
    cursor = ScriptedCursor({"table_constraints": [("id",)]})

    assert schema_over(cursor).is_primary_key(ColumnReference("id", TITLE)) is True


def test_is_primary_key_false_when_no_constraint() -> None:
    cursor = ScriptedCursor({"table_constraints": []})

    assert schema_over(cursor).is_primary_key(ColumnReference("title", TITLE)) is False


def test_has_index_short_circuits_on_primary_key() -> None:
    """`has_index` is ``is_primary_key(col) or has_secondary_index(col)``.

    A primary key must therefore answer without ever querying for secondary indexes.
    """
    cursor = ScriptedCursor({"table_constraints": [("id",)]}, strict=True)

    assert schema_over(cursor).has_index(ColumnReference("id", TITLE)) is True
    assert cursor.calls_mentioning("pg_index") == []


# -- datatypes ---------------------------------------------------------------------------------------


def test_datatype_reads_information_schema() -> None:
    cursor = ScriptedCursor({"information_schema.columns": [("integer",)]})

    assert schema_over(cursor).datatype(ColumnReference("id", TITLE)) == "integer"


@pytest.mark.parametrize(
    ("datatype", "expected"),
    [
        ("text", True),
        ("character varying", True),
        ("integer", False),
        ("numeric", False),
    ],
)
def test_is_string_column_classifies_datatype(datatype: str, expected: bool) -> None:
    """`is_string_column` is pure classification layered on `datatype()`."""
    cursor = ScriptedCursor({"information_schema.columns": [(datatype,)]})

    assert schema_over(cursor).is_string_column(ColumnReference("c", TITLE)) is expected
