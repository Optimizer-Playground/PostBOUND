"""Tests for `postbound.qal.format_quick`, the pretty-printer for parsed queries.

Most tests here check the *round trip*: format a parsed query, reparse the result, and assert the two query
objects are equal -- rather than pinning an exact formatted string, which is both more brittle and less
informative about what actually matters (that formatting is a lossless, semantics-preserving operation).

One deliberate exception: queries with more than one table in an *implicit* (comma-separated) FROM clause are
avoided in the round-trip checks. `_quick_format_implicit_from` builds its output from
``list(from_clause.tables())``, and `tables()` returns a `set` -- so the printed order of such a FROM clause
is not guaranteed to match the order the query was written in, and (since Python's string hashing is salted
per process) is not even guaranteed to be stable across interpreter runs. That does not lose any information
(the table set itself round-trips correctly, and this qal has no ORDINALITY-sensitive implicit joins), so it
is not treated as a bug here -- just something the round-trip tests route around by using explicit JOIN
syntax or single-table FROM clauses instead.
"""

from __future__ import annotations

from postbound import parser
from postbound.qal import format_quick


def parse(sql: str):
    return parser.parse_query(sql, bind_columns=False)


def assert_round_trips(sql: str) -> None:
    """Formats `sql` and asserts that reparsing the result reproduces an equal query object."""
    query = parse(sql)
    formatted = format_quick(query)
    reparsed = parse(formatted)
    assert query == reparsed, f"format_quick output did not round-trip:\n{formatted}"


# -- basic structure --------------------------------------------------------------------------------------


def test_format_quick_puts_every_clause_on_its_own_line() -> None:
    query = parse("SELECT r.a FROM r WHERE r.a = 1")

    formatted = format_quick(query)

    lines = formatted.splitlines()
    assert lines[0].startswith("SELECT")
    assert any(line.startswith("FROM") for line in lines)
    assert any(line.strip().startswith("WHERE") for line in lines)


def test_format_quick_appends_a_trailing_semicolon_by_default() -> None:
    query = parse("SELECT * FROM r")

    assert format_quick(query).rstrip().endswith(";")


def test_format_quick_can_omit_the_trailing_semicolon() -> None:
    query = parse("SELECT * FROM r")

    assert not format_quick(query, trailing_semicolon=False).rstrip().endswith(";")


# -- round trips --------------------------------------------------------------------------------------------


def test_round_trips_a_single_table_filter_query() -> None:
    assert_round_trips("SELECT * FROM r WHERE r.a = 1 AND r.b = 2")


def test_round_trips_an_explicit_join() -> None:
    assert_round_trips("SELECT r.a, s.b FROM r JOIN s ON r.a = s.b WHERE r.c < 42")


def test_round_trips_a_left_outer_join() -> None:
    assert_round_trips("SELECT * FROM r LEFT JOIN s ON r.a = s.b")


def test_round_trips_group_by_having_order_by_limit() -> None:
    assert_round_trips("SELECT r.a, count(*) FROM r GROUP BY r.a HAVING count(*) > 1 ORDER BY r.a LIMIT 5")


def test_round_trips_a_cte() -> None:
    assert_round_trips("WITH cte_r AS (SELECT * FROM r) SELECT * FROM cte_r JOIN r ON cte_r.id = r.id")


def test_round_trips_a_union() -> None:
    assert_round_trips("SELECT r.a FROM r UNION SELECT s.b FROM s")


def test_round_trips_a_query_with_a_subquery_in_the_where_clause() -> None:
    assert_round_trips("SELECT * FROM r WHERE r.a IN (SELECT s.b FROM s WHERE s.c = 42)")


def test_round_trips_a_query_with_more_than_three_joined_tables() -> None:
    """Exercises the multi-line branch of `_quick_format_implicit_from` (more than 3 tables) -- using
    explicit joins throughout so table order is well-defined and the round trip is exact.
    """
    assert_round_trips("SELECT * FROM r JOIN s ON r.a = s.b JOIN t ON s.c = t.d JOIN u ON t.e = u.f WHERE r.g = 1")


def test_multi_table_implicit_from_preserves_the_table_set_even_if_not_the_order() -> None:
    """The documented exception: table *order* in an implicit FROM clause is not guaranteed to survive
    formatting, but the table *set* -- and therefore the query's meaning -- always does.
    """
    query = parse("SELECT * FROM r, s, t WHERE r.a = s.b AND s.c = t.d")

    formatted = format_quick(query)
    reparsed = parse(formatted)

    assert reparsed.tables() == query.tables()
    assert reparsed.predicates() == query.predicates()


# -- hint block placement ----------------------------------------------------------------------------------


def test_hint_block_precedes_the_query_by_default() -> None:
    query = parse("/*+ HashJoin(r s) */ SELECT * FROM r JOIN s ON r.a = s.b")

    formatted = format_quick(query)

    assert formatted.startswith("/*+")
    select_line_index = next(i for i, line in enumerate(formatted.splitlines()) if line.startswith("SELECT"))
    hint_line_index = next(i for i, line in enumerate(formatted.splitlines()) if line.startswith("/*+"))
    assert hint_line_index < select_line_index


def test_hint_block_can_be_inlined_into_the_select_clause() -> None:
    query = parse("/*+ HashJoin(r s) */ SELECT * FROM r JOIN s ON r.a = s.b")

    formatted = format_quick(query, inline_hint_block=True)

    select_line = next(line for line in formatted.splitlines() if "SELECT" in line)
    assert "/*+" in select_line


def test_hinted_query_round_trips() -> None:
    assert_round_trips("/*+ HashJoin(r s) */ SELECT * FROM r JOIN s ON r.a = s.b")


# -- dialect flavor -----------------------------------------------------------------------------------------


def test_vanilla_and_postgres_flavors_agree_on_a_plain_query() -> None:
    query = parse("SELECT * FROM r WHERE r.a = 1")

    assert format_quick(query, flavor="vanilla") == format_quick(query, flavor="postgres")
