"""Tests for the `SqlClause` hierarchy in `postbound.qal`: SELECT/FROM/WHERE/GROUP BY/HAVING/ORDER BY/LIMIT,
table sources, CTEs, hints, and EXPLAIN.

As with expressions and predicates, clauses are obtained by parsing real SQL through the actual parser.
"""

from __future__ import annotations

import pytest

from postbound import parser
from postbound._core import ColumnReference, TableReference
from postbound.qal import (
    DirectTableSource,
    FunctionTableSource,
    Hint,
    JoinTableSource,
    JoinType,
    SubqueryTableSource,
    ValuesTableSource,
    all_simple_from,
)

R = TableReference("r")
S = TableReference("s")


def parse(sql: str):
    return parser.parse_query(sql, bind_columns=False)


# -- Select / Projection ------------------------------------------------------------------------------------


def test_select_targets_are_ordered_as_written() -> None:
    query = parse("SELECT r.a, r.b AS x FROM r")

    names = [target.target_name for target in query.select_clause.targets]
    assert names == ["", "x"]


def test_select_is_star_for_a_bare_star() -> None:
    assert parse("SELECT * FROM r").select_clause.is_star() is True
    assert parse("SELECT r.a FROM r").select_clause.is_star() is False


def test_select_is_count_star_for_count_star() -> None:
    assert parse("SELECT count(*) FROM r").select_clause.is_count_star() is True
    assert parse("SELECT count(r.a) FROM r").select_clause.is_count_star() is False


def test_select_is_distinct() -> None:
    assert parse("SELECT DISTINCT r.a FROM r").select_clause.is_distinct() is True
    assert parse("SELECT r.a FROM r").select_clause.is_distinct() is False


def test_select_distinct_on_captures_the_distinguishing_columns() -> None:
    query = parse("SELECT DISTINCT ON (r.a) r.a, r.b FROM r")

    distinct_on = query.select_clause.distinct_on
    assert distinct_on is not None
    assert {col.column for col in distinct_on} == {ColumnReference("a", R)}


def test_select_columns_union_every_target() -> None:
    query = parse("SELECT r.a, r.b + 1 FROM r")

    assert query.select_clause.columns() == {ColumnReference("a", R), ColumnReference("b", R)}


# -- From / TableSource -------------------------------------------------------------------------------------


def test_direct_table_source_wraps_a_table_reference() -> None:
    query = parse("SELECT * FROM r AS ra")

    source = query.from_clause.items[0]
    assert isinstance(source, DirectTableSource)
    assert source.table == TableReference("r", "ra")


def test_subquery_table_source_carries_the_nested_query() -> None:
    query = parse("SELECT * FROM (SELECT * FROM s) AS sub")

    source = query.from_clause.items[0]
    assert isinstance(source, SubqueryTableSource)
    assert source.target_table == TableReference.create_virtual("sub")
    assert S in source.query.tables()
    assert source.lateral is False


def test_subquery_table_source_lateral_flag() -> None:
    query = parse("SELECT * FROM r, LATERAL (SELECT * FROM s WHERE s.a = r.a) AS sub")

    source = query.from_clause.items[1]
    assert isinstance(source, SubqueryTableSource)
    assert source.lateral is True


def test_values_table_source_carries_its_rows() -> None:
    query = parse("SELECT * FROM (VALUES (1, 2), (3, 4)) AS t (a, b)")

    source = query.from_clause.items[0]
    assert isinstance(source, ValuesTableSource)
    assert len(source.rows) == 2
    assert source.table == TableReference.create_virtual("t")


def test_function_table_source_carries_the_function_call() -> None:
    query = parse("SELECT * FROM my_table_function(42) AS foo")

    source = query.from_clause.items[0]
    assert isinstance(source, FunctionTableSource)
    assert source.function.function == "MY_TABLE_FUNCTION"
    assert source.target_table == TableReference.create_virtual("foo")


def test_join_table_source_captures_both_sides_and_the_condition() -> None:
    query = parse("SELECT * FROM r JOIN s ON r.a = s.b")

    source = query.from_clause.items[0]
    assert isinstance(source, JoinTableSource)
    assert source.join_type == JoinType.InnerJoin
    assert isinstance(source.lhs, DirectTableSource) and source.lhs.table == R
    assert isinstance(source.rhs, DirectTableSource) and source.rhs.table == S
    assert source.join_condition is not None
    assert source.join_condition.columns() == {ColumnReference("a", R), ColumnReference("b", S)}


def test_join_table_source_left_outer_join_type() -> None:
    query = parse("SELECT * FROM r LEFT JOIN s ON r.a = s.b")

    source = query.from_clause.items[0]
    assert source.join_type == JoinType.LeftJoin


def test_all_simple_from_accepts_plain_base_tables() -> None:
    query = parse("SELECT * FROM r, s AS sa")

    assert all_simple_from(query.from_clause.items) is True


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM r, (SELECT * FROM s) AS sub",
        "SELECT * FROM r JOIN s ON r.a = s.b",
        "SELECT * FROM r, (VALUES (1, 2)) AS t (a, b)",
        "SELECT * FROM r, my_table_function(42) AS foo",
    ],
    ids=["subquery", "explicit-join", "values", "table-function"],
)
def test_all_simple_from_rejects_any_other_table_source(sql: str) -> None:
    query = parse(sql)

    assert all_simple_from(query.from_clause.items) is False


def test_all_simple_from_holds_vacuously_for_no_table_sources() -> None:
    assert all_simple_from([]) is True


def test_join_table_source_pattern_match_extracts_all_four_fields() -> None:
    """Regression guard for the `__match_args__` bug fixed in commit 74a1448: a positional
    `case JoinTableSource(lhs, rhs, cond, join_type)` must actually match and bind, not silently fail (which
    is how Python's structural pattern matching handles an `AttributeError` from a bad `__match_args__`
    entry -- it never raises, the `case` just never matches).
    """
    query = parse("SELECT * FROM r JOIN s ON r.a = s.b")
    source = query.from_clause.items[0]

    match source:
        case JoinTableSource(lhs, rhs, cond, join_type):
            matched = True
        case _:
            matched = False

    assert matched, "case JoinTableSource(...) with positional arguments failed to match"
    assert lhs.table == R
    assert rhs.table == S
    assert cond is not None
    assert join_type == JoinType.InnerJoin


def test_sql_query_bound_tables_includes_tables_from_an_explicit_join() -> None:
    """Regression guard, same root cause as the pattern-match test above: `bound_tables()` used to crash
    outright for any query with an explicit JOIN, via `_collect_bound_tables_from_source`.
    """
    query = parse("SELECT * FROM r JOIN s ON r.a = s.b")

    assert query.bound_tables() == {R, S}


# -- Where / GroupBy / Having ------------------------------------------------------------------------------


def test_where_clause_root_predicate() -> None:
    query = parse("SELECT * FROM r WHERE r.a = 1")

    assert query.where_clause is not None
    assert query.where_clause.root.columns() == {ColumnReference("a", R)}


def test_groupby_clause_captures_grouping_columns() -> None:
    query = parse("SELECT r.a, count(*) FROM r GROUP BY r.a")

    assert query.groupby_clause is not None
    assert {col.column for col in query.groupby_clause.group_columns} == {ColumnReference("a", R)}


def test_groupby_clause_distinct() -> None:
    query = parse("SELECT DISTINCT r.a FROM r GROUP BY r.a")

    assert query.groupby_clause is not None


def test_having_clause_condition() -> None:
    query = parse("SELECT r.a, count(*) FROM r GROUP BY r.a HAVING count(*) > 1")

    assert query.having_clause is not None
    assert query.having_clause.condition.tables() == set()  # count(*) references no table


# -- OrderBy / Ordering -------------------------------------------------------------------------------------


def test_orderby_clause_captures_direction_per_expression() -> None:
    query = parse("SELECT * FROM r ORDER BY r.a DESC, r.b")

    orderings = query.orderby_clause.expressions
    assert len(orderings) == 2
    assert orderings[0].ascending is False
    assert orderings[1].ascending is None  # no explicit ASC/DESC was written


def test_query_is_ordered_iff_it_has_an_orderby_clause() -> None:
    assert parse("SELECT * FROM r ORDER BY r.a").is_ordered() is True
    assert parse("SELECT * FROM r").is_ordered() is False


# -- Limit --------------------------------------------------------------------------------------------------


def test_limit_clause_captures_limit_and_offset() -> None:
    query = parse("SELECT * FROM r LIMIT 10 OFFSET 5")

    assert query.limit_clause is not None
    assert query.limit_clause.limit == 10
    assert query.limit_clause.offset == 5


def test_limit_clause_fetch_next_syntax() -> None:
    """`FETCH NEXT n ROWS ONLY` is the SQL-standard spelling of `LIMIT n`."""
    query = parse("SELECT * FROM r FETCH NEXT 10 ROWS ONLY")

    assert query.limit_clause is not None
    assert query.limit_clause.limit == 10


def test_limit_clause_without_offset() -> None:
    query = parse("SELECT * FROM r LIMIT 10")

    assert query.limit_clause is not None
    assert query.limit_clause.offset is None


# -- CommonTableExpression ----------------------------------------------------------------------------------


def test_single_cte_is_recognized() -> None:
    query = parse("WITH cte_r AS (SELECT * FROM r) SELECT * FROM cte_r JOIN r ON cte_r.id = r.id")

    assert query.cte_clause is not None
    assert len(query.cte_clause.queries) == 1
    assert len(query.tables()) == 2  # r and the virtual cte_r table


def test_multiple_ctes_are_all_recognized() -> None:
    query = parse(
        "WITH cte_r AS (SELECT * FROM r), cte_s AS (SELECT min(s.c) FROM s WHERE s.c < 42) SELECT * FROM cte_r, cte_s"
    )

    assert query.cte_clause is not None
    assert len(query.cte_clause.queries) == 2


def test_cte_target_table_is_virtual() -> None:
    query = parse("WITH cte_r AS (SELECT * FROM r) SELECT * FROM cte_r")

    with_query = query.cte_clause.queries[0]
    assert with_query.target_table == TableReference.create_virtual("cte_r")
    assert with_query.target_table.virtual is True


# -- Hint -----------------------------------------------------------------------------------------------------


def test_hint_carries_preparatory_statements_and_query_hints() -> None:
    hint = Hint("SET x = 1;", "/*+ HashJoin(r s) */")

    assert hint.preparatory_statements == "SET x = 1;"
    assert hint.query_hints == "/*+ HashJoin(r s) */"


# -- Explain ----------------------------------------------------------------------------------------------------


def test_plain_explain_is_not_analyze() -> None:
    query = parse("EXPLAIN SELECT * FROM r")

    assert query.is_explain() is True
    assert query.explain is not None
    assert query.explain.analyze is False


def test_explain_analyze_with_json_format() -> None:
    query = parse("EXPLAIN (ANALYZE, FORMAT JSON) SELECT * FROM r")

    assert query.explain is not None
    assert query.explain.analyze is True
    assert query.explain.target_format == "json"


def test_non_explain_query_has_no_explain_clause() -> None:
    query = parse("SELECT * FROM r")

    assert query.is_explain() is False
    assert query.explain is None


# -- Set operation clauses (recognized via SetQuery, see test_qal_query.py for the query-level API) ------------


def test_union_produces_a_set_query() -> None:
    from postbound.qal import SetQuery

    query = parse("SELECT r.a FROM r UNION SELECT s.b FROM s")

    assert isinstance(query, SetQuery)


def test_except_and_intersect_also_produce_set_queries() -> None:
    from postbound.qal import SetOperator, SetQuery

    except_query = parse("SELECT r.a FROM r EXCEPT SELECT s.b FROM s")
    intersect_query = parse("SELECT r.a FROM r INTERSECT SELECT s.b FROM s")

    assert isinstance(except_query, SetQuery) and except_query.set_operation == SetOperator.Except
    assert isinstance(intersect_query, SetQuery) and intersect_query.set_operation == SetOperator.Intersect


# -- regression tests --------------------------------------------------------------------------------------


def test_predicates_of_an_outer_join_keep_the_inner_join_and_where_predicates() -> None:
    """Regression guard for df75e5a: `JoinTableSource.predicates()` raised a `ValueError` for every non-inner join,
    so `query.predicates()`, `joins()` and `filters()` crashed on any query containing a LEFT/RIGHT/FULL JOIN. The
    outer join condition is now left out of the predicate tree (with a warning), but everything else is kept.
    """
    query = parse("SELECT * FROM r JOIN s ON r.a = s.b LEFT JOIN t ON s.b = t.c WHERE r.x > 5")

    with pytest.warns(UserWarning, match="Outer join conditions are ignored"):
        predicates = query.predicates()

    assert predicates is not None
    assert str(predicates) == "r.a = s.b AND r.x > 5"


def test_predicates_of_a_natural_join_keep_the_where_predicates() -> None:
    """Regression guard for df75e5a: like outer joins, NATURAL joins made `JoinTableSource.predicates()` raise a
    `ValueError` because they are not `JoinType.InnerJoin`. Their implicit condition cannot be represented, but the
    remaining predicates must still be available.
    """
    query = parse("SELECT * FROM r NATURAL JOIN s WHERE r.x > 5")

    with pytest.warns(UserWarning, match="Natural join conditions are ignored"):
        predicates = query.predicates()

    assert predicates is not None
    assert str(predicates) == "r.x > 5"
