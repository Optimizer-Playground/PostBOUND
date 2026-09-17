"""Tests for `SqlQuery`/`SelectStatement`/`SetQuery` in `postbound.qal` -- the aggregate, query-wide API
(`tables()`, `columns()`, `predicates()`, `joins()`, `filters()`, `subqueries()`, `has_simple_from()`,
`is_ordered()`, `is_set_query()`, `clauses()`, `hints`, `bound_tables()`/`is_dependent()`) rather than the
individual clause/predicate/expression classes covered by the other `test_qal_*.py` modules.

As elsewhere, queries are obtained by parsing real SQL through the actual parser.
"""

from __future__ import annotations

from postbound import parser
from postbound._core import ColumnReference, TableReference
from postbound.qal import Limit, SelectStatement, SetOperator, SetQuery, is_select_query, is_set_query

R = TableReference("r")
S = TableReference("s")
T = TableReference("t")


def parse(sql: str):
    return parser.parse_query(sql, bind_columns=False)


# -- tables() / columns() ------------------------------------------------------------------------------


def test_tables_collects_every_table_in_the_query() -> None:
    query = parse("SELECT r.a FROM r, s WHERE r.a = s.b")

    assert query.tables() == {R, S}


def test_columns_collects_every_referenced_column_across_all_clauses() -> None:
    query = parse("SELECT r.a FROM r, s WHERE r.a = s.b AND r.c = 1")

    assert query.columns() == {
        ColumnReference("a", R),
        ColumnReference("b", S),
        ColumnReference("c", R),
    }


def test_tables_includes_a_cte_as_a_virtual_table() -> None:
    query = parse("WITH cte_r AS (SELECT * FROM r) SELECT * FROM cte_r")

    assert TableReference.create_virtual("cte_r") in query.tables()


# -- filters() / joins() (aggregate view; the classification rules themselves are covered in
# test_qal_predicates.py) -------------------------------------------------------------------------------


def test_filters_and_joins_partition_the_where_clause() -> None:
    query = parse("SELECT * FROM r, s, t WHERE r.a = s.b AND s.c = t.d AND r.e = 1")

    assert len(query.filters()) == 1
    assert len(query.joins()) == 2


def test_dependent_subquery_predicate_counts_as_a_filter_not_a_join() -> None:
    query = parse("SELECT * FROM r WHERE r.a IN (SELECT s.b FROM s WHERE r.c = s.d)")

    assert len(query.joins()) == 0
    assert len(query.filters()) == 1


def test_query_without_a_where_clause_has_no_filters_or_joins() -> None:
    query = parse("SELECT * FROM r")

    assert not query.filters()
    assert not query.joins()


# -- subqueries() ---------------------------------------------------------------------------------------


def test_subqueries_finds_a_subquery_in_the_where_clause() -> None:
    query = parse("SELECT * FROM r WHERE r.a IN (SELECT s.b FROM s)")

    subqueries = query.subqueries()
    assert len(subqueries) == 1
    assert S in next(iter(subqueries)).tables()


def test_subqueries_is_empty_for_a_query_without_any() -> None:
    query = parse("SELECT * FROM r WHERE r.a = 42")

    assert not query.subqueries()


def test_subqueries_finds_a_subquery_in_an_explicit_join() -> None:
    """Regression guard for the `JoinTableSource.__match_args__` bug (fixed in 74a1448): subquery collection
    walks table sources with the same positional pattern match that used to never fire for explicit joins.
    """
    query = parse("SELECT * FROM r JOIN (SELECT * FROM s) AS sub ON r.a = sub.b")

    subqueries = query.subqueries()
    assert len(subqueries) == 1


# -- is_dependent() ---------------------------------------------------------------------------------------


def test_dependent_subquery_is_recognized_as_dependent() -> None:
    query = parse("SELECT * FROM r WHERE EXISTS (SELECT * FROM s WHERE r.a = s.b)")

    subquery = next(iter(query.subqueries()))
    assert subquery.is_dependent() is True


def test_independent_query_is_not_dependent() -> None:
    query = parse("SELECT * FROM r WHERE r.a = 42")

    assert query.is_dependent() is False


# -- has_simple_from() ------------------------------------------------------------------------------------


def test_has_simple_from_true_for_comma_separated_tables() -> None:
    query = parse("SELECT * FROM r, s, t WHERE r.a = s.b AND s.b = t.c")

    assert query.has_simple_from() is True


def test_has_simple_from_true_regardless_of_aliasing() -> None:
    query = parse("SELECT * FROM r AS ra, s AS sa")

    assert query.has_simple_from() is True


def test_has_simple_from_false_for_an_explicit_join() -> None:
    query = parse("SELECT * FROM r JOIN s ON r.a = s.b WHERE r.c LIKE '%42%' AND r.c < s.b")

    assert query.has_simple_from() is False


# -- is_ordered() / is_explain() / is_set_query() --------------------------------------------------------


def test_is_ordered_reflects_the_presence_of_an_orderby_clause() -> None:
    assert parse("SELECT * FROM r ORDER BY r.a").is_ordered() is True
    assert parse("SELECT * FROM r").is_ordered() is False


def test_is_explain_reflects_the_presence_of_an_explain_clause() -> None:
    assert parse("EXPLAIN SELECT * FROM r").is_explain() is True
    assert parse("SELECT * FROM r").is_explain() is False


def test_is_set_query_true_for_union_except_intersect() -> None:
    assert parse("SELECT r.a FROM r UNION SELECT s.b FROM s").is_set_query() is True
    assert parse("SELECT r.a FROM r EXCEPT SELECT s.b FROM s").is_set_query() is True
    assert parse("SELECT r.a FROM r INTERSECT SELECT s.b FROM s").is_set_query() is True
    assert parse("SELECT * FROM r").is_set_query() is False


# -- clauses() --------------------------------------------------------------------------------------------


def test_clauses_lists_every_present_clause_in_order() -> None:
    query = parse("SELECT r.a FROM r WHERE r.a = 1 ORDER BY r.a LIMIT 5")

    clause_types = [type(clause).__name__ for clause in query.clauses()]

    assert clause_types == ["Select", "From", "Where", "OrderBy", "Limit"]


def test_clauses_skip_omits_the_requested_types() -> None:
    query = parse("SELECT r.a FROM r WHERE r.a = 1 ORDER BY r.a LIMIT 5")

    remaining = query.clauses(skip=[Limit])

    assert not any(isinstance(clause, Limit) for clause in remaining)
    assert len(remaining) == 4


# -- hints ------------------------------------------------------------------------------------------------


def test_query_without_hints_has_none() -> None:
    query = parse("SELECT * FROM r")

    assert query.hints is None


def test_query_with_a_hint_block_exposes_it() -> None:
    query = parse("/*+ HashJoin(r s) */ SELECT * FROM r, s WHERE r.a = s.b")

    assert query.hints is not None
    assert "HashJoin(r s)" in query.hints.query_hints


# -- SelectStatement / SetQuery / type guards --------------------------------------------------------------


def test_plain_query_is_a_select_statement() -> None:
    query = parse("SELECT * FROM r")

    assert isinstance(query, SelectStatement)
    assert is_select_query(query) is True
    assert is_set_query(query) is False


def test_union_query_is_a_set_query() -> None:
    query = parse("SELECT r.a FROM r UNION SELECT s.b FROM s")

    assert isinstance(query, SetQuery)
    assert is_select_query(query) is False
    assert is_set_query(query) is True
    assert query.set_operation == SetOperator.Union


def test_set_query_exposes_both_operand_queries() -> None:
    query = parse("SELECT r.a FROM r UNION SELECT s.b FROM s")

    assert isinstance(query, SetQuery)
    assert R in query.lhs.tables()
    assert S in query.rhs.tables()


def test_set_query_tables_and_columns_union_both_sides() -> None:
    query = parse("SELECT r.a FROM r UNION SELECT s.b FROM s")

    assert isinstance(query, SetQuery)
    assert query.tables() == {R, S}
    assert query.columns() == {ColumnReference("a", R), ColumnReference("b", S)}


def test_set_query_is_ordered_reflects_a_trailing_orderby() -> None:
    query = parse("SELECT r.a FROM r UNION SELECT s.b FROM s ORDER BY 1")

    assert isinstance(query, SetQuery)
    assert query.is_ordered() is True
