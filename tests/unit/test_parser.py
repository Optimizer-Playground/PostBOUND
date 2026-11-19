"""Tests for `postbound.parser` -- the translation of SQL text into qal objects.

Everything here is tier 0: queries reference the abstract tables ``r`` and ``s`` with explicitly qualified columns and
are parsed with ``bind_columns=False``, so no schema or database is needed.

For now, this module only pins known parser bugs (see the `Documents a real bug` tests). The bugs are listed under
*Known bugs* in ``CHANGELOG.md``.
"""

from __future__ import annotations

import pytest

from postbound import parser
from postbound.parser import ParserError
from postbound.qal import FunctionTableSource, SelectStatement, SqlQuery


def parse(sql: str) -> SqlQuery:
    return parser.parse_query(sql, bind_columns=False)


# -- known bugs -----------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM r, generate_series(1, 3) AS g WHERE r.a = g.g",
        "SELECT g.g FROM generate_series(1, 3) AS g",
        "SELECT * FROM generate_series(1, 3) AS g ORDER BY g.g",
        "SELECT * FROM r JOIN generate_series(1, 3) AS g ON r.a = g.g",
        "SELECT * FROM r, LATERAL generate_series(1, r.a) AS g WHERE g.g > 1",
        "SELECT * FROM r, unnest(r.arr) AS u WHERE u.u = 1",
        "SELECT * FROM r WHERE r.a IN (SELECT g.g FROM generate_series(1, 3) AS g)",
        "SELECT * FROM r, (SELECT g.g FROM generate_series(1, 3) AS g) AS sq WHERE r.a = sq.g",
    ],
    ids=["where", "select", "order-by", "join-on", "lateral", "unnest", "in-subquery", "from-subquery"],
)
def test_parser_cannot_resolve_columns_of_a_table_function(sql: str) -> None:
    """Documents a real bug, not the intended behaviour.

    `_pglast_parse_from_entry` turns a ``RangeFunction`` into a `FunctionTableSource`, but -- unlike the ``RangeVar``
    (`QueryNamespace.register_table`) and ``RangeSubselect`` (`QueryNamespace.open_nested`) branches -- never registers
    the function's alias in the `QueryNamespace`. `_pglast_create_bound_colref` therefore cannot resolve the table of a
    qualified column like ``g.g`` and raises. This applies anywhere a table function occurs, including (nested)
    subqueries. Correct behaviour: the alias is registered, so that ``g.g`` binds to the virtual table ``g``.
    """
    with pytest.raises(ParserError, match="Table not found"):
        parse(sql)


def test_parser_drops_the_column_aliases_of_a_table_function() -> None:
    """Documents a real bug, not the intended behaviour.

    `_pglast_parse_from_entry` only reads ``alias.aliasname`` of a ``RangeFunction`` and ignores ``alias.colnames``,
    and `FunctionTableSource` has no way to store them. ``generate_series(1, 3) AS g(x)`` is therefore formatted back as
    ``... AS g``, so the output column is called ``g`` instead of ``x`` -- any reference to ``g.x`` breaks.
    Correct behaviour: the column aliases are kept and formatted.
    """
    query = parse("SELECT * FROM r, generate_series(1, 3) AS g(x)")

    assert isinstance(query, SelectStatement)
    assert query.from_clause is not None
    function_source = query.from_clause.items[1]
    assert isinstance(function_source, FunctionTableSource)
    assert str(query) == "SELECT * FROM r, GENERATE_SERIES(1, 3) AS g;"


@pytest.mark.parametrize(
    "sql",
    ["SELECT * FROM r, (SELECT 1) WHERE true", "SELECT * FROM r, (VALUES (1), (2)) WHERE true"],
    ids=["subquery", "values"],
)
def test_parser_rejects_an_unaliased_subquery_in_the_from_clause(sql: str) -> None:
    """Documents a real bug, not the intended behaviour.

    Since Postgres 16, subqueries and *VALUES* lists in the *FROM* clause no longer need an alias. For an unaliased
    ``RangeSubselect``, `_pglast_parse_from_entry` (or `_pglast_parse_values` for *VALUES*) calls
    `QueryNamespace.open_nested` with ``alias=""``, which builds
    ``TableReference.create_virtual("")`` and fails its "full name or alias" check. Correct behaviour: the query parses,
    and the source binds no table (as `SubqueryTableSource.bound_tables` already supports a missing target table).
    """
    with pytest.raises(ValueError, match="Full name or alias required"):
        parse(sql)
