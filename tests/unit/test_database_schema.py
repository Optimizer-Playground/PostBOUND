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

from postbound import UnboundColumnError
from postbound._core import ColumnReference, TableReference
from postbound.db import Database, DatabaseSchema, ForeignKeyRef, PrimaryKeyRef
from tests.doubles import FakeDatabase, ScriptedCursor

TITLE = TableReference("title")
ID = ColumnReference("id", TITLE)
KIND = ColumnReference("kind", TITLE)
YEAR = ColumnReference("production_year", TITLE)


def schema_over(cursor: ScriptedCursor, *, placeholder: str = "%s") -> DatabaseSchema:
    """Builds a stock `DatabaseSchema` whose cursor is the given double."""
    database: Database = FakeDatabase(cursor_double=cursor)
    return DatabaseSchema(database, prep_placeholder=placeholder, default_schema="public", default_catalog="fake")


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


# -- primary keys ------------------------------------------------------------------------------------

PK_FRAGMENT = "constraint_type = 'primary key'"


def test_is_primary_key_accepts_a_composite_key_in_any_order() -> None:
    cursor = ScriptedCursor({PK_FRAGMENT: [("id",), ("kind",)]})

    assert schema_over(cursor).is_primary_key([KIND, ID]) is True


@pytest.mark.parametrize(
    "candidate",
    [[ID], [ID, KIND, YEAR], [ID, YEAR]],
    ids=["part-of-the-key", "superset-of-the-key", "overlapping-columns"],
)
def test_is_primary_key_requires_the_exact_column_set(candidate: list[ColumnReference]) -> None:
    cursor = ScriptedCursor({PK_FRAGMENT: [("id",), ("kind",)]})

    assert schema_over(cursor).is_primary_key(candidate) is False


def test_is_primary_key_of_a_single_column_is_false_for_a_composite_key() -> None:
    cursor = ScriptedCursor({PK_FRAGMENT: [("id",), ("kind",)]})

    assert schema_over(cursor).is_primary_key(ID) is False


def test_is_primary_key_rejects_an_empty_column_list() -> None:
    with pytest.raises(ValueError, match="empty column list"):
        schema_over(ScriptedCursor()).is_primary_key([])


def test_is_primary_key_rejects_columns_of_different_tables() -> None:
    with pytest.raises(ValueError, match="same table"):
        schema_over(ScriptedCursor()).is_primary_key([ID, ColumnReference("id", TableReference("movie_info"))])


def test_is_primary_key_rejects_unbound_columns() -> None:
    with pytest.raises(UnboundColumnError):
        schema_over(ScriptedCursor()).is_primary_key(ColumnReference("id"))


def test_is_primary_key_ignores_table_aliases() -> None:
    cursor = ScriptedCursor({PK_FRAGMENT: [("id",)]})

    assert schema_over(cursor).is_primary_key(ColumnReference("id", TableReference("title", "t"))) is True
    assert cursor.calls[0].parameters == ("title",)


def test_is_primary_key_binds_table_catalog_and_schema_in_template_order() -> None:
    cursor = ScriptedCursor({PK_FRAGMENT: []})
    column = ColumnReference("id", TableReference("title", schema="imdb", catalog="prod"))

    schema_over(cursor).is_primary_key(column)

    call = cursor.calls[0]
    assert call.parameters == ("title", "prod", "imdb")
    assert not call.mentions("current_database()", "current_schema()")


def test_primary_key_of_returns_the_columns_in_key_definition_order() -> None:
    cursor = ScriptedCursor({PK_FRAGMENT: [("kind",), ("id",)]})

    pk = schema_over(cursor).primary_key_of(TITLE)

    assert pk == PrimaryKeyRef([KIND, ID])
    assert pk is not None
    assert [col.name for col in pk.columns] == ["kind", "id"]
    assert cursor.calls[0].mentions("order by kcu.ordinal_position")


def test_primary_key_of_returns_none_for_a_table_without_primary_key() -> None:
    cursor = ScriptedCursor({PK_FRAGMENT: []})

    assert schema_over(cursor).primary_key_of(TITLE) is None


def test_primary_key_of_accepts_a_table_name_and_ignores_aliases() -> None:
    cursor = ScriptedCursor({PK_FRAGMENT: [("id",)]})
    schema = schema_over(cursor)

    assert schema.primary_key_of("title") == schema.primary_key_of(TableReference("title", "t"))
    assert [call.parameters for call in cursor.calls] == [("title",), ("title",)]


def test_primary_key_of_binds_table_catalog_and_schema_in_template_order() -> None:
    cursor = ScriptedCursor({PK_FRAGMENT: []})

    schema_over(cursor).primary_key_of(TableReference("title", schema="imdb", catalog="prod"))

    assert cursor.calls[0].parameters == ("title", "prod", "imdb")


# -- foreign keys ------------------------------------------------------------------------------------

FK_FRAGMENT = "constraint_type = 'foreign key'"
MOVIE_INFO = TableReference("movie_info")
MOVIE_ID = ColumnReference("movie_id", MOVIE_INFO)
KIND_ID = ColumnReference("kind_id", MOVIE_INFO)
INFO_TYPE = ColumnReference("info_type", MOVIE_INFO)

# (constraint, fk column, target catalog, target schema, target table, target column) -- the shape of the query
FK_ROWS = [
    ("fk_movie", "movie_id", "fake", "public", "title", "id"),
    ("fk_kind", "kind_id", "fake", "public", "title", "kind"),
    ("fk_kind", "info_type", "fake", "public", "title", "production_year"),
]


def test_all_foreign_keys_groups_the_rows_of_a_constraint_into_one_composite_key() -> None:
    cursor = ScriptedCursor({FK_FRAGMENT: FK_ROWS})

    fks = schema_over(cursor).all_foreign_keys_on(MOVIE_INFO)

    assert fks == {
        ForeignKeyRef([MOVIE_ID], [ID]),
        ForeignKeyRef([KIND_ID, INFO_TYPE], [KIND, YEAR]),
    }


def test_all_foreign_keys_keeps_the_column_pairing_of_a_composite_key() -> None:
    cursor = ScriptedCursor({FK_FRAGMENT: FK_ROWS})

    fks = schema_over(cursor).all_foreign_keys_on(MOVIE_INFO)

    composite = next(fk for fk in fks if not fk.is_simple())
    assert [col.name for col in composite.fk_cols] == ["kind_id", "info_type"]
    assert [col.name for col in composite.referenced_cols] == ["kind", "production_year"]


def test_all_foreign_keys_is_empty_without_constraints() -> None:
    cursor = ScriptedCursor({FK_FRAGMENT: []})

    assert schema_over(cursor).all_foreign_keys_on(MOVIE_INFO) == set()


def test_all_foreign_keys_omits_the_default_catalog_and_schema_of_the_target() -> None:
    cursor = ScriptedCursor({FK_FRAGMENT: [("fk", "movie_id", "fake", "public", "title", "id")]})

    (fk,) = schema_over(cursor).all_foreign_keys_on(MOVIE_INFO)

    assert fk.referenced_table == TITLE
    assert fk.referenced_table.schema == ""
    assert fk.referenced_table.catalog == ""


def test_all_foreign_keys_keeps_a_non_default_catalog_and_schema_of_the_target() -> None:
    cursor = ScriptedCursor({FK_FRAGMENT: [("fk", "movie_id", "archive", "imdb", "title", "id")]})

    (fk,) = schema_over(cursor).all_foreign_keys_on(MOVIE_INFO)

    assert fk.referenced_table == TableReference("title", schema="imdb", catalog="archive")


@pytest.mark.parametrize(
    ("table", "expected"),
    [
        (MOVIE_INFO, ("movie_info",)),
        (TableReference("movie_info", schema="imdb"), ("movie_info", "imdb")),
        (TableReference("movie_info", catalog="prod"), ("movie_info", "prod")),
        (TableReference("movie_info", schema="imdb", catalog="prod"), ("movie_info", "imdb", "prod")),
    ],
    ids=["unqualified", "schema-only", "catalog-only", "fully-qualified"],
)
def test_all_foreign_keys_binds_parameters_in_template_order(table: TableReference, expected: tuple) -> None:
    """The template mentions the table, then the schema, then the catalog -- and each exactly once.

    Since the query was rewritten to use `referential_constraints`, the parameter list must follow that order.
    """
    cursor = ScriptedCursor({FK_FRAGMENT: []})

    schema_over(cursor).all_foreign_keys_on(table)

    call = cursor.calls[0]
    assert call.parameters == expected
    assert call.normalized.count("%s") == len(expected)


def test_all_foreign_keys_respects_the_placeholder_dialect() -> None:
    cursor = ScriptedCursor({FK_FRAGMENT: []})
    table = TableReference("movie_info", schema="imdb", catalog="prod")

    schema_over(cursor, placeholder="?").all_foreign_keys_on(table)

    call = cursor.calls[0]
    assert call.normalized.count("?") == len(call.parameters) == 3
    assert "%s" not in call.normalized


def test_all_foreign_keys_accepts_a_table_name_and_ignores_aliases() -> None:
    cursor = ScriptedCursor({FK_FRAGMENT: FK_ROWS})
    schema = schema_over(cursor)

    assert schema.all_foreign_keys_on("movie_info") == schema.all_foreign_keys_on(TableReference("movie_info", "mi"))
    assert [call.parameters for call in cursor.calls] == [("movie_info",), ("movie_info",)]


def test_foreign_keys_on_selects_the_keys_that_consist_of_exactly_the_given_columns() -> None:
    cursor = ScriptedCursor({FK_FRAGMENT: FK_ROWS})
    schema = schema_over(cursor)

    assert schema.foreign_keys_on(MOVIE_ID) == {ForeignKeyRef([MOVIE_ID], [ID])}
    assert schema.foreign_keys_on([INFO_TYPE, KIND_ID]) == {ForeignKeyRef([KIND_ID, INFO_TYPE], [KIND, YEAR])}


def test_foreign_keys_on_a_part_of_a_composite_key_finds_nothing() -> None:
    cursor = ScriptedCursor({FK_FRAGMENT: FK_ROWS})

    assert schema_over(cursor).foreign_keys_on(KIND_ID) == set()


def test_foreign_keys_on_rejects_an_empty_column_list() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        schema_over(ScriptedCursor()).foreign_keys_on([])


def test_foreign_keys_on_rejects_columns_of_different_tables() -> None:
    with pytest.raises(ValueError, match="same table"):
        schema_over(ScriptedCursor()).foreign_keys_on([MOVIE_ID, ID])


def test_foreign_keys_on_rejects_unbound_columns() -> None:
    with pytest.raises(UnboundColumnError):
        schema_over(ScriptedCursor()).foreign_keys_on(ColumnReference("movie_id"))


# -- unique constraints ------------------------------------------------------------------------------

UNIQUE_FRAGMENT = "constraint_type = 'unique'"


def test_is_unique_matches_a_composite_constraint_in_any_order() -> None:
    cursor = ScriptedCursor({UNIQUE_FRAGMENT: [("uq_kind_year", "kind"), ("uq_kind_year", "production_year")]})

    assert schema_over(cursor).is_unique([YEAR, KIND]) is True


@pytest.mark.parametrize(
    "candidate",
    [KIND, [KIND, YEAR, ID]],
    ids=["part-of-the-constraint", "superset-of-the-constraint"],
)
def test_is_unique_requires_the_exact_column_set_of_one_constraint(candidate) -> None:
    cursor = ScriptedCursor({UNIQUE_FRAGMENT: [("uq_kind_year", "kind"), ("uq_kind_year", "production_year")]})

    assert schema_over(cursor).is_unique(candidate) is False


def test_is_unique_does_not_merge_columns_of_different_constraints() -> None:
    """Two single-column constraints do not make the *pair* unique as a constraint of its own."""
    cursor = ScriptedCursor({UNIQUE_FRAGMENT: [("uq_kind", "kind"), ("uq_year", "production_year")]})
    schema = schema_over(cursor)

    assert schema.is_unique([KIND, YEAR]) is False
    assert schema.is_unique(KIND) is True


def test_is_unique_queries_only_unique_constraints() -> None:
    """Primary keys are deliberately excluded: `ColumnInfo.unique` documents UNIQUE constraints only."""
    cursor = ScriptedCursor({UNIQUE_FRAGMENT: []})

    schema_over(cursor).is_unique(ID)

    assert not cursor.calls[0].mentions("'primary key'")


def test_is_unique_rejects_invalid_arguments() -> None:
    schema = schema_over(ScriptedCursor())

    with pytest.raises(ValueError, match="cannot be empty"):
        schema.is_unique([])
    with pytest.raises(ValueError, match="same table"):
        schema.is_unique([ID, MOVIE_ID])
    with pytest.raises(UnboundColumnError):
        schema.is_unique(ColumnReference("id"))


# -- indexes -----------------------------------------------------------------------------------------

INDEX_FRAGMENT = "select tc.constraint_name, kcu.column_name"

INDEX_ROWS = [
    ("title_pkey", "id"),
    ("title_kind_year_key", "kind"),
    ("title_kind_year_key", "production_year"),
    ("fk_kind", "kind"),
]


def test_all_indexes_on_groups_the_columns_by_index_name() -> None:
    cursor = ScriptedCursor({INDEX_FRAGMENT: INDEX_ROWS})

    indexes = schema_over(cursor).all_indexes_on(TITLE)

    assert dict(indexes) == {
        "title_pkey": {ID},
        "title_kind_year_key": {KIND, YEAR},
        "fk_kind": {KIND},
    }


def test_all_indexes_on_is_empty_for_a_table_without_indexes() -> None:
    cursor = ScriptedCursor({INDEX_FRAGMENT: []})

    assert dict(schema_over(cursor).all_indexes_on(TITLE)) == {}


def test_indexes_on_matches_the_exact_column_set_of_an_index() -> None:
    cursor = ScriptedCursor({INDEX_FRAGMENT: INDEX_ROWS})
    schema = schema_over(cursor)

    assert schema.indexes_on(ID) == {"title_pkey"}
    assert schema.indexes_on(KIND) == {"fk_kind"}
    assert schema.indexes_on([YEAR, KIND]) == {"title_kind_year_key"}


def test_indexes_on_a_column_ignores_composite_indexes_containing_it() -> None:
    cursor = ScriptedCursor({INDEX_FRAGMENT: INDEX_ROWS})

    assert schema_over(cursor).indexes_on(YEAR) == set()


def test_indexes_on_rejects_invalid_arguments() -> None:
    schema = schema_over(ScriptedCursor())

    with pytest.raises(ValueError, match="cannot be empty"):
        schema.indexes_on([])
    with pytest.raises(ValueError, match="same table"):
        schema.indexes_on([ID, MOVIE_ID])
    with pytest.raises(UnboundColumnError):
        schema.indexes_on(ColumnReference("id"))


def test_has_secondary_index_is_false_for_the_primary_key_index() -> None:
    cursor = ScriptedCursor({PK_FRAGMENT: [("id",)], INDEX_FRAGMENT: INDEX_ROWS})

    assert schema_over(cursor).has_secondary_index(ID) is False


def test_has_secondary_index_is_true_for_a_non_primary_key_index() -> None:
    cursor = ScriptedCursor({PK_FRAGMENT: [("id",)], INDEX_FRAGMENT: INDEX_ROWS})

    assert schema_over(cursor).has_secondary_index(KIND) is True


def test_has_secondary_index_supports_composite_indexes() -> None:
    cursor = ScriptedCursor({PK_FRAGMENT: [("id",)], INDEX_FRAGMENT: INDEX_ROWS})
    schema = schema_over(cursor)

    assert schema.has_secondary_index([YEAR, KIND]) is True
    assert schema.has_secondary_index([YEAR, ID]) is False


def test_has_index_covers_composite_primary_keys_and_composite_secondary_indexes() -> None:
    cursor = ScriptedCursor({PK_FRAGMENT: [("id",), ("kind",)], INDEX_FRAGMENT: INDEX_ROWS})
    schema = schema_over(cursor)

    assert schema.has_index([KIND, ID]) is True  # composite primary key
    assert schema.has_index([KIND, YEAR]) is True  # composite unique index
    assert schema.has_index([YEAR, ID]) is False  # no such index


# -- regression tests --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("table", "expected"),
    [
        (TITLE, ("title",)),
        (TableReference("title", schema="imdb"), ("title", "imdb")),
        (TableReference("title", schema="imdb", catalog="prod"), ("title", "prod", "imdb")),
    ],
    ids=["unqualified", "schema-only", "fully-qualified"],
)
def test_all_indexes_on_binds_exactly_one_parameter_per_placeholder(table: TableReference, expected: tuple) -> None:
    """Regression guard for the composite key rework: the UNION was removed, but ``params *= 2`` stayed, so psycopg
    rejected the statement (``the query has 1 placeholders but 2 parameters were passed``).
    """
    cursor = ScriptedCursor({INDEX_FRAGMENT: []})

    schema_over(cursor).all_indexes_on(table)

    call = cursor.calls[0]
    assert call.parameters == expected
    assert call.normalized.count("%s") == len(expected)


def test_all_indexes_on_restricts_the_constraints_to_key_constraints() -> None:
    """Regression guard for the composite key rework: removing the UNION also removed the constraint type filter, so
    CHECK and NOT NULL constraints were reported as indexes.
    """
    cursor = ScriptedCursor({INDEX_FRAGMENT: []})

    schema_over(cursor).all_indexes_on(TITLE)

    assert cursor.calls[0].mentions("tc.constraint_type in (")


def test_all_foreign_keys_does_not_join_the_target_through_the_referencing_table() -> None:
    """Regression guard for the composite key rework: the join with the referenced columns demanded
    ``tc.table_name = ccu.table_name``, but that view describes the *referenced* table, so only self-references were
    found.
    """
    cursor = ScriptedCursor({FK_FRAGMENT: []})

    schema_over(cursor).all_foreign_keys_on(MOVIE_INFO)

    assert not cursor.calls[0].mentions("constraint_column_usage", "ccu.table_name")


def test_all_foreign_keys_resolves_the_target_columns_through_the_referenced_constraint() -> None:
    """Regression guard for the composite key rework: ``position_in_unique_constraint`` (the position within the
    referenced *key*) was compared with ``columns.ordinal_position`` (the position within the referenced *table*), so
    ``ch(b, a) REFERENCES p(b, a)`` with ``PRIMARY KEY (b, a)`` on ``p(a, b)`` came back with crossed columns. The
    position now has to be matched against the ``key_column_usage`` rows of the referenced constraint.
    """
    cursor = ScriptedCursor({FK_FRAGMENT: []})

    schema_over(cursor).all_foreign_keys_on(MOVIE_INFO)

    call = cursor.calls[0]
    assert call.mentions(
        "information_schema.referential_constraints", "kcu.position_in_unique_constraint = ref.ordinal_position"
    )
    assert not call.mentions("information_schema.columns")


def test_is_primary_key_lists_the_unbound_columns_when_reporting_them() -> None:
    """Regression guard for the composite key rework: the message was built with ``if col.is_bound()`` (inverted) and
    therefore listed the bound columns -- an empty list when every column is unbound.
    """
    unbound = ColumnReference("id")

    with pytest.raises(UnboundColumnError) as excinfo:
        schema_over(ScriptedCursor()).is_primary_key([ID, unbound])

    assert "id" in str(excinfo.value)
    assert str(ID.table) not in str(excinfo.value)


def test_all_indexes_on_selects_only_from_joined_tables() -> None:
    """Regression guard for the composite key rework: the second half of a UNION selected ``ccu.column_name`` without
    joining ``ccu``, so the statement was rejected by every server.
    """
    cursor = ScriptedCursor({INDEX_FRAGMENT: []})

    schema_over(cursor).all_indexes_on(TITLE)

    sql = cursor.calls[0].normalized
    assert " union " not in sql
    assert "ccu" not in sql
    assert cursor.calls[0].mentions("join information_schema.key_column_usage kcu")


def test_all_indexes_on_reads_the_columns_of_foreign_keys_from_the_constrained_side() -> None:
    """Regression guard for the composite key rework: the columns were read from ``constraint_column_usage``, which
    lists the *referenced* columns of a foreign key. ``fk2(pid) REFERENCES pu(id)`` was missing from the indexes and
    ``self(par) REFERENCES self(id)`` was reported as an index on ``self.id`` (PostgreSQL 18). ``key_column_usage``
    lists the constrained columns of primary key, UNIQUE and foreign key constraints alike.
    """
    cursor = ScriptedCursor({INDEX_FRAGMENT: []})

    schema_over(cursor).all_indexes_on(TITLE)

    call = cursor.calls[0]
    assert call.mentions("'primary key', 'foreign key', 'unique'")
    assert call.mentions("kcu.column_name", "tc.table_name = kcu.table_name")
    assert not call.mentions("constraint_column_usage")


def test_foreign_keys_on_ignores_the_table_alias_of_the_given_columns() -> None:
    """Regression guard for the composite key rework: the requested columns kept their alias and therefore never
    matched the alias-free columns of the foreign keys.
    """
    cursor = ScriptedCursor({FK_FRAGMENT: FK_ROWS})
    aliased = ColumnReference("movie_id", TableReference("movie_info", "mi"))

    assert schema_over(cursor).foreign_keys_on(aliased) == {ForeignKeyRef([MOVIE_ID], [ID])}


def test_indexes_on_ignores_the_table_alias_of_the_given_columns() -> None:
    """Regression guard for the composite key rework: see `test_foreign_keys_on_ignores_the_table_alias_of_the_given_columns`."""
    cursor = ScriptedCursor({INDEX_FRAGMENT: INDEX_ROWS})
    aliased = ColumnReference("id", TableReference("title", "t"))

    assert schema_over(cursor).indexes_on(aliased) == {"title_pkey"}


def test_all_foreign_keys_scopes_the_key_lookup_to_the_constraint_table() -> None:
    """Regression guard for the composite key rework: constraint names are only unique per table on Postgres, so
    joining ``key_column_usage`` on the constraint name alone mixed up the keys of different tables.
    """
    cursor = ScriptedCursor({FK_FRAGMENT: []})

    schema_over(cursor).all_foreign_keys_on(MOVIE_INFO)

    assert cursor.calls[0].mentions("and tc.table_name = kcu.table_name")


def test_is_primary_key_ignores_the_case_of_the_column_names() -> None:
    """Regression guard for the DuckDB identifier casing bug: the key columns from the catalog were compared to the
    requested column names as raw strings. DuckDB keeps the spelling of the DDL (the Stats database declares
    ``users.Id``) while the parser folds unquoted identifiers to lower case, so ``users.id`` was not recognized as the
    primary key -- even though `ColumnReference` itself already compares names case-insensitively.
    """
    cursor = ScriptedCursor({"table_constraints": [("Id",)]})

    assert schema_over(cursor).is_primary_key(ColumnReference("id", TITLE)) is True


def test_is_unique_ignores_the_case_of_the_column_names() -> None:
    """Regression guard for the DuckDB identifier casing bug: see
    `test_is_primary_key_ignores_the_case_of_the_column_names`. The columns of the UNIQUE constraints were compared to
    the requested column names as raw strings.
    """
    cursor = ScriptedCursor({UNIQUE_FRAGMENT: [("uq_kind_year", "Kind"), ("uq_kind_year", "Production_Year")]})

    assert schema_over(cursor).is_unique([KIND, YEAR]) is True


def test_lookup_column_ignores_the_case_of_the_column_names() -> None:
    """Regression guard for the DuckDB identifier casing bug: see
    `test_is_primary_key_ignores_the_case_of_the_column_names`. The column names of the candidate tables were compared
    to the requested name as raw strings, so a lower-cased column from a parsed query never found its table.
    """
    cursor = ScriptedCursor({"information_schema.columns": [("Id",), ("Production_Year",)]})

    assert schema_over(cursor).lookup_column("production_year", [TITLE]) == TITLE
