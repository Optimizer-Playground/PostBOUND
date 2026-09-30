"""Tests for `postbound.transform` -- the functions that build new `SqlQuery` objects from existing ones.

Everything here is tier 0. Inputs are obtained by parsing real SQL over the abstract tables ``r``, ``s``, ``t``
with explicitly qualified columns, so the parser binds them without a schema. The few transformations that need
schema knowledge (`expand_select_star`, `expand_natural_joins`, `normalize_query`) receive a `StaticSchema`
through their ``schema`` parameter; the `DatabasePool` fallback is exercised once with a `FakeDatabase`.

Many transformations assemble their output from sets (predicate conjuncts, FROM items, projections of moved
subqueries), and qal equality is order-sensitive. Such results are therefore compared order-independently via
`conjuncts()` / `join_pairs()` / `tables()` instead of against a parsed "expected" query.

Several tests pin real bugs rather than intended behaviour. Their docstrings start with
``Documents a real bug, not the intended behaviour.`` and the bugs are listed under *Known bugs* in
``CHANGELOG.md``. SQL formatting itself is covered by ``test_qal_formatter.py`` and is not re-asserted here.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import pytest

import postbound as pb
from postbound import parser, transform
from postbound._core import ColumnReference, TableReference
from postbound.db import DatabasePool
from postbound.qal import (
    AbstractPredicate,
    AndPredicate,
    BetweenPredicate,
    BinaryOperator,
    ColumnExpression,
    CommonTableExpression,
    CompoundPredicate,
    Explain,
    FunctionExpression,
    GroupBy,
    Hint,
    InPredicate,
    JoinTableSource,
    JoinType,
    Limit,
    NotPredicate,
    OrderBy,
    OrPredicate,
    Projection,
    Select,
    SelectStatement,
    SetOperator,
    SetQuery,
    SimpleFilter,
    SqlExpression,
    SqlQuery,
    StaticValueExpression,
    SubqueryTableSource,
    TableSource,
    ValuesWithQuery,
    Where,
)
from postbound.transform import _get_predicate_fragment
from tests.doubles import FakeDatabase, StaticSchema

# -- fixtures -----------------------------------------------------------------------------------------------

R = TableReference("r")
S = TableReference("s")
T = TableReference("t")


def col(name: str, table: TableReference) -> ColumnReference:
    return ColumnReference(name, table)


def parse(sql: str) -> SqlQuery:
    return parser.parse_query(sql, bind_columns=False)


def select(sql: str) -> SelectStatement:
    query = parse(sql)
    assert isinstance(query, SelectStatement)
    return query


def pred(condition: str) -> AbstractPredicate:
    """Parses a single predicate over the tables ``r``, ``s`` and ``t``."""
    where = select(f"SELECT * FROM r, s, t WHERE {condition}").where_clause
    assert where is not None
    return where.root


def conjuncts(query: SqlQuery) -> set[AbstractPredicate]:
    """The top-level conjuncts of the WHERE clause (a single predicate if it is not a conjunction)."""
    root = where_root(query)
    return set(root.children) if isinstance(root, AndPredicate) else {root}


def where_root(query: SqlQuery) -> AbstractPredicate:
    assert isinstance(query, SelectStatement)
    assert query.where_clause is not None
    return query.where_clause.root


def from_items(query: SqlQuery) -> Sequence[TableSource]:
    assert isinstance(query, SelectStatement)
    assert query.from_clause is not None
    return query.from_clause.items


def join_pairs(query: SqlQuery) -> set[frozenset[ColumnReference]]:
    """The column pairs of all join predicates, independent of which side each column is on."""
    return {frozenset(join.columns()) for join in query.joins()}


SCHEMA = StaticSchema({"r": ["a", "id"], "s": ["b", "id"], "t": ["id", "c"]})

CHAIN_JOIN = select("SELECT * FROM r, s, t WHERE r.a = s.b AND s.c = t.d AND r.a = 42 ORDER BY s.b")
EXPLICIT_JOIN = select("SELECT * FROM r JOIN s ON r.a = s.b JOIN t ON s.id = t.id WHERE r.a = 1")
UNION = parse("SELECT r.a FROM r UNION SELECT s.b FROM s")


# -- flatten_and_predicate ------------------------------------------------------------------------------------


def test_flatten_and_predicate_returns_a_non_conjunction_unchanged() -> None:
    predicate = pred("r.a = 1 OR r.b = 2")

    assert transform.flatten_and_predicate(predicate) is predicate


def test_flatten_and_predicate_moves_nested_conjuncts_to_the_top_level() -> None:
    predicate = pred("(r.a = s.b AND r.a = 42) AND s.b = 24")

    flattened = transform.flatten_and_predicate(predicate)

    assert isinstance(flattened, AndPredicate)
    assert set(flattened.children) == {pred("r.a = s.b"), pred("r.a = 42"), pred("s.b = 24")}


def test_flatten_and_predicate_stops_at_a_disjunction() -> None:
    predicate = pred("r.a = 1 AND (r.b = 2 OR (r.c = 3 AND r.d = 4))")

    flattened = transform.flatten_and_predicate(predicate)

    assert isinstance(flattened, AndPredicate)
    assert set(flattened.children) == {pred("r.a = 1"), pred("r.b = 2 OR (r.c = 3 AND r.d = 4)")}


def test_flatten_and_predicate_unwraps_a_conjunction_of_identical_children() -> None:
    predicate = pred("r.a = 1 AND (r.a = 1 AND r.a = 1)")

    assert transform.flatten_and_predicate(predicate) == pred("r.a = 1")


# -- explicit_to_implicit -------------------------------------------------------------------------------------


def test_explicit_to_implicit_moves_join_conditions_into_the_where_clause() -> None:
    implicit = transform.explicit_to_implicit(EXPLICIT_JOIN)

    assert implicit.has_simple_from()
    assert implicit.tables() == {R, S, T}
    assert conjuncts(implicit) == {pred("r.a = s.b"), pred("s.id = t.id"), pred("r.a = 1")}


def test_explicit_to_implicit_handles_a_mix_of_explicit_joins_and_plain_tables() -> None:
    query = select("SELECT * FROM r JOIN s ON r.a = s.b, t WHERE t.c = 1")

    implicit = transform.explicit_to_implicit(query)

    assert implicit.has_simple_from()
    assert implicit.tables() == {R, S, T}
    assert conjuncts(implicit) == {pred("r.a = s.b"), pred("t.c = 1")}


def test_explicit_to_implicit_keeps_the_where_clause_of_a_query_without_join_conditions() -> None:
    """A subquery in the FROM clause is not a "simple" FROM, but contributes no join predicate of its own."""
    query = select("SELECT * FROM (SELECT * FROM r) AS sub WHERE sub.a = 1")

    implicit = transform.explicit_to_implicit(query)

    assert implicit.from_clause == query.from_clause
    assert implicit.where_clause == query.where_clause


def test_explicit_to_implicit_keeps_the_other_clauses() -> None:
    query = select("SELECT r.a FROM r JOIN s ON r.a = s.b ORDER BY r.a LIMIT 5")

    implicit = transform.explicit_to_implicit(query)

    assert implicit.select_clause == query.select_clause
    assert implicit.orderby_clause == query.orderby_clause
    assert implicit.limit_clause == query.limit_clause


@pytest.mark.parametrize(
    "sql",
    ["SELECT * FROM r, s WHERE r.a = s.b", "SELECT 1"],
    ids=["already-implicit", "no-from-clause"],
)
def test_explicit_to_implicit_returns_queries_without_explicit_joins_unchanged(sql: str) -> None:
    query = select(sql)

    assert transform.explicit_to_implicit(query) is query


@pytest.mark.parametrize(
    "join",
    ["LEFT JOIN s ON r.a = s.b", "RIGHT JOIN s ON r.a = s.b", "FULL OUTER JOIN s ON r.a = s.b", "NATURAL LEFT JOIN s"],
    ids=["left", "right", "full", "natural-left"],
)
def test_explicit_to_implicit_rejects_outer_joins(join: str) -> None:
    query = select(f"SELECT * FROM r {join}")

    with pytest.raises(ValueError, match="outer joins cannot be re-written"):
        transform.explicit_to_implicit(query)


def test_explicit_to_implicit_rejects_a_natural_inner_join() -> None:
    query = select("SELECT * FROM r NATURAL JOIN s")

    with pytest.raises(ValueError, match="NATURAL JOIN is not supported"):
        transform.explicit_to_implicit(query)


def test_explicit_to_implicit_rewrites_both_sides_of_a_set_query() -> None:
    query = parse("SELECT * FROM r JOIN s ON r.a = s.b UNION ALL SELECT * FROM s JOIN t ON s.c = t.d")

    implicit = transform.explicit_to_implicit(query)

    assert isinstance(implicit, SetQuery)
    assert implicit.set_operation == SetOperator.UnionAll
    assert implicit.lhs.has_simple_from()
    assert implicit.rhs.has_simple_from()
    assert conjuncts(implicit.rhs) == {pred("s.c = t.d")}


def test_explicit_to_implicit_leaves_the_original_query_unchanged() -> None:
    original = str(EXPLICIT_JOIN)

    transform.explicit_to_implicit(EXPLICIT_JOIN)

    assert str(EXPLICIT_JOIN) == original
    assert not EXPLICIT_JOIN.has_simple_from()


def test_explicit_to_implicit_crashes_on_a_cross_join() -> None:
    query = select("SELECT * FROM r CROSS JOIN s")

    implicit = transform.explicit_to_implicit(query)

    assert isinstance(implicit, SelectStatement)
    assert implicit.has_simple_from()
    assert implicit.tables() == {R, S}
    assert implicit.where_clause is None


# -- _get_predicate_fragment ----------------------------------------------------------------------------------


def test_predicate_fragment_keeps_only_the_parts_referencing_the_given_tables() -> None:
    """The docstring example: the disjunction is broken up, no logical simplification is attempted."""
    predicate = pred("r.a > 100 AND (s.b = 42 OR r.a = 42)")

    fragment = _get_predicate_fragment(predicate, {R})

    assert isinstance(fragment, AndPredicate)
    assert set(fragment.children) == {pred("r.a > 100"), pred("r.a = 42")}


def test_predicate_fragment_is_none_when_everything_is_pruned() -> None:
    assert _get_predicate_fragment(pred("s.b = 1 AND s.c = 2"), {R}) is None


def test_predicate_fragment_keeps_a_negation_around_a_single_remaining_child() -> None:
    """Conjunctions/disjunctions with one remaining child are unwrapped, but a NOT must stay a NOT."""
    fragment = _get_predicate_fragment(pred("NOT (r.a = 1 AND s.b = 2)"), {R})

    assert fragment == CompoundPredicate.create_not(pred("r.a = 1"))


# -- extract_query_fragment -----------------------------------------------------------------------------------


def test_extract_query_fragment_keeps_the_induced_joins_filters_and_ordering() -> None:
    """The docstring example for tables R and S."""
    fragment = transform.extract_query_fragment(CHAIN_JOIN, [R, S])

    assert fragment is not None
    assert fragment.tables() == {R, S}
    assert conjuncts(fragment) == {pred("r.a = s.b"), pred("r.a = 42")}
    assert fragment.orderby_clause == CHAIN_JOIN.orderby_clause


def test_extract_query_fragment_for_a_single_table_without_filters_drops_the_where_clause() -> None:
    fragment = transform.extract_query_fragment(CHAIN_JOIN, S)

    assert fragment is not None
    assert fragment.tables() == {S}
    assert fragment.where_clause is None
    assert fragment.orderby_clause == CHAIN_JOIN.orderby_clause


def test_extract_query_fragment_returns_none_for_tables_not_in_the_query() -> None:
    assert transform.extract_query_fragment(select("SELECT * FROM r"), [S]) is None


def test_extract_query_fragment_breaks_disjunctions() -> None:
    """The docstring example: ``R.a = 42 OR S.b = 42`` loses its S branch for the R fragment."""
    query = select("SELECT * FROM r, s WHERE r.a < 100 AND (r.a = 42 OR s.b = 42)")

    fragment = transform.extract_query_fragment(query, R)

    assert fragment is not None
    assert conjuncts(fragment) == {pred("r.a < 100"), pred("r.a = 42")}


def test_extract_query_fragment_rewrites_explicit_joins_first() -> None:
    query = select("SELECT * FROM r JOIN s ON r.a = s.b JOIN t ON s.c = t.d")

    fragment = transform.extract_query_fragment(query, [R, S])

    assert fragment is not None
    assert fragment.has_simple_from()
    assert fragment.tables() == {R, S}
    assert conjuncts(fragment) == {pred("r.a = s.b")}


def test_extract_query_fragment_filters_group_by_and_having_and_keeps_the_limit() -> None:
    query = select(
        "SELECT s.b, count(*) FROM r, s WHERE r.a = s.b GROUP BY s.b, r.a "
        "HAVING count(*) > 1 AND min(r.a) > 2 ORDER BY s.b LIMIT 3"
    )

    fragment = transform.extract_query_fragment(query, S)

    assert fragment is not None
    assert fragment.groupby_clause == GroupBy.create_for([col("b", S)])
    assert fragment.having_clause is not None
    assert fragment.having_clause == select("SELECT * FROM r HAVING count(*) > 1").having_clause
    assert fragment.limit_clause == query.limit_clause


def test_extract_query_fragment_keeps_only_the_ctes_among_the_referenced_tables() -> None:
    query = select("WITH c AS (SELECT * FROM r), d AS (SELECT * FROM s) SELECT * FROM c, d WHERE c.a = d.b")
    cte_c = TableReference.create_virtual("c")

    fragment = transform.extract_query_fragment(query, cte_c)

    assert fragment is not None
    assert fragment.cte_clause is not None
    assert [cte.target_table for cte in fragment.cte_clause.queries] == [cte_c]


def test_extract_query_fragment_keep_projection_retains_targets_of_exactly_the_given_tables() -> None:
    query = select("SELECT r.a, s.b, count(*) FROM r, s")

    fragment = transform.extract_query_fragment(query, R)

    assert fragment is not None
    assert fragment.select_clause == select("SELECT r.a, count(*) FROM r").select_clause


def test_extract_query_fragment_keep_projection_falls_back_to_star_and_keeps_distinct() -> None:
    query = select("SELECT DISTINCT s.b FROM r, s")

    fragment = transform.extract_query_fragment(query, R)

    assert fragment is not None
    assert fragment.select_clause.is_star()
    assert fragment.select_clause.is_distinct()


@pytest.mark.parametrize(
    ("projection", "expected"),
    [("star", Select.star()), ("*", Select.star()), ("count_star", Select.count_star())],
    ids=["star", "literal-star", "count-star"],
)
def test_extract_query_fragment_projection_option_replaces_the_select_clause(projection, expected: Select) -> None:
    fragment = transform.extract_query_fragment(select("SELECT r.a FROM r"), R, projection=projection)

    assert fragment is not None
    assert fragment.select_clause == expected


def test_extract_query_fragment_rejects_an_unknown_projection() -> None:
    with pytest.raises(ValueError, match="Invalid projection type: bogus"):
        transform.extract_query_fragment(select("SELECT * FROM r"), R, projection=cast(Any, "bogus"))


def test_extract_query_fragment_extracts_both_sides_of_a_set_query() -> None:
    query = parse("SELECT * FROM r, s WHERE r.a = s.b UNION SELECT * FROM r, t")

    fragment = transform.extract_query_fragment(query, R)

    assert isinstance(fragment, SetQuery)
    assert fragment.set_operation == SetOperator.Union
    assert fragment.lhs.tables() == {R}
    assert fragment.rhs.tables() == {R}


def test_extract_query_fragment_of_a_set_query_returns_the_only_matching_side() -> None:
    query = parse("SELECT * FROM r, s UNION SELECT * FROM s")

    fragment = transform.extract_query_fragment(query, [R, S])

    assert isinstance(fragment, SelectStatement)
    assert fragment.tables() == {R, S}


def test_extract_query_fragment_keep_projection_keeps_targets_of_a_strict_subset_of_the_tables() -> None:
    query = select("SELECT r.a, s.b FROM r, s, t WHERE r.a = s.b AND s.c = t.d")

    fragment = transform.extract_query_fragment(query, [R, S])

    assert fragment is not None
    assert fragment.select_clause.columns() == {col("a", R), col("b", S)}


def test_extract_query_fragment_ignores_the_projection_option_for_set_queries() -> None:
    query = parse("SELECT r.a FROM r UNION SELECT r.b FROM r, s")

    fragment = transform.extract_query_fragment(query, R, projection="count_star")

    assert isinstance(fragment, SetQuery)
    assert fragment.lhs.select_clause == select("SELECT r.a FROM r").select_clause
    assert fragment.rhs.select_clause == select("SELECT r.b FROM r").select_clause


# -- extract_subquery -----------------------------------------------------------------------------------------


def test_extract_subquery_keeps_only_select_star_from_and_where() -> None:
    query = select(
        "SELECT r.a FROM r, s WHERE r.a = s.b AND r.c = 1 GROUP BY r.a HAVING count(*) > 1 ORDER BY r.a LIMIT 1"
    )

    subquery = transform.extract_subquery(query, R)

    assert subquery == select("SELECT * FROM r WHERE r.c = 1")


def test_extract_subquery_raises_when_the_tables_are_not_in_the_query() -> None:
    with pytest.raises(ValueError, match="Could not extract subquery"):
        transform.extract_subquery(select("SELECT * FROM r"), S)


# -- expand_to_query ------------------------------------------------------------------------------------------


def test_expand_to_query_selects_from_the_tables_of_the_predicate() -> None:
    query = transform.expand_to_query(pred("r.a = s.b"))

    assert query.select_clause.is_star()
    assert query.tables() == {R, S}
    assert conjuncts(query) == {pred("r.a = s.b")}


def test_expand_to_query_can_produce_a_count_star_query() -> None:
    query = transform.expand_to_query(pred("r.a = 1"), projection="count_star")

    assert query == select("SELECT COUNT(*) FROM r WHERE r.a = 1")


# -- move_into_subquery ---------------------------------------------------------------------------------------


def test_move_into_subquery_replaces_the_tables_by_a_virtual_subquery_table() -> None:
    query = select("SELECT r.x FROM r, s, t WHERE r.a = s.b AND s.c = t.d AND r.e = 1")
    sub = TableReference.create_virtual("r_s")

    moved = transform.move_into_subquery(query, [R, S], "r_s")

    assert moved.tables() - {R, S} == {T, sub}  # r and s now only appear inside the subquery
    assert moved.select_clause == Select.create_for([col("x", sub)])
    assert conjuncts(moved) == {transform.rename_columns_in_predicate(pred("s.c = t.d"), {col("c", S): col("c", sub)})}

    [subquery] = moved.subqueries()
    assert subquery.tables() == {R, S}
    assert conjuncts(subquery) == {pred("r.a = s.b"), pred("r.e = 1")}


def test_move_into_subquery_generates_a_default_subquery_name() -> None:
    query = select("SELECT * FROM r, s, t WHERE r.a = s.b AND s.c = t.d")

    moved = transform.move_into_subquery(query, [R, S])

    [source] = [src for src in from_items(moved) if isinstance(src, SubqueryTableSource)]
    assert source.target_name in {"r_s", "s_r"}


def test_move_into_subquery_rejects_a_query_without_a_from_clause() -> None:
    with pytest.raises(ValueError, match="without a FROM clause"):
        transform.move_into_subquery(select("SELECT 1"), [R, S])


def test_move_into_subquery_rejects_a_query_with_virtual_tables() -> None:
    query = select("WITH c AS (SELECT * FROM r) SELECT * FROM c, s, t WHERE c.a = s.b")

    with pytest.raises(ValueError, match="virtual tables"):
        transform.move_into_subquery(query, [S, T])


def test_move_into_subquery_rejects_fewer_than_two_tables() -> None:
    with pytest.raises(ValueError, match="At least two tables required"):
        transform.move_into_subquery(select("SELECT * FROM r, s WHERE r.a = s.b"), [R])


def test_move_into_subquery_rejects_tables_exporting_columns_of_the_same_name() -> None:
    query = select("SELECT r.a, s.a FROM r, s, t WHERE r.a = s.a AND s.a = t.d")

    with pytest.raises(ValueError, match="columns of the same name"):
        transform.move_into_subquery(query, [R, S])


def test_move_into_subquery_of_all_tables_drops_outer_where() -> None:
    query = select("SELECT * FROM r, s WHERE r.a = s.b")

    moved = transform.move_into_subquery(query, [R, S], "r_s")

    assert moved.where_clause is None


# -- add_ec_predicates ----------------------------------------------------------------------------------------


def test_add_ec_predicates_adds_the_transitively_implied_join() -> None:
    query = select("SELECT * FROM r, s, t WHERE r.a = s.b AND s.b = t.c AND r.e = 1")

    expanded = transform.add_ec_predicates(query)

    assert join_pairs(expanded) == {
        frozenset({col("a", R), col("b", S)}),
        frozenset({col("b", S), col("c", T)}),
        frozenset({col("a", R), col("c", T)}),
    }
    assert set(expanded.filters()) == {pred("r.e = 1")}


def test_add_ec_predicates_returns_a_query_without_predicates_unchanged() -> None:
    query = select("SELECT * FROM r, s")

    assert transform.add_ec_predicates(query) is query


def test_add_ec_predicates_rewrites_explicit_joins_first() -> None:
    query = select("SELECT * FROM r JOIN s ON r.a = s.b JOIN t ON s.b = t.c")

    expanded = transform.add_ec_predicates(query)

    assert expanded.has_simple_from()
    assert frozenset({col("a", R), col("c", T)}) in join_pairs(expanded)


def test_add_ec_predicates_rejects_non_binary_joins() -> None:
    query = select("SELECT * FROM r, s, t WHERE r.a BETWEEN s.b AND t.c")

    with pytest.raises(ValueError, match="non-binary joins"):
        transform.add_ec_predicates(query)


def test_add_ec_predicates_keeps_non_equi_joins() -> None:
    query = select("SELECT * FROM r, s, t WHERE r.a = s.b AND s.c < t.d")

    expanded = transform.add_ec_predicates(query)

    assert len(conjuncts(expanded)) == 2
    assert join_pairs(expanded) == {frozenset({col("a", R), col("b", S)}), frozenset({col("c", S), col("d", T)})}


# -- infer_between_predicates ---------------------------------------------------------------------------------


def test_infer_between_predicates_merges_a_closed_range() -> None:
    query = select("SELECT * FROM r WHERE r.a >= 1 AND r.a <= 5")

    rewritten = transform.infer_between_predicates(query)

    assert conjuncts(rewritten) == {pred("r.a BETWEEN 1 AND 5")}
    assert isinstance(where_root(rewritten), BetweenPredicate)


def test_infer_between_predicates_keeps_unrelated_predicates() -> None:
    query = select("SELECT * FROM r, s WHERE r.a >= 1 AND r.a <= 5 AND r.a = s.b AND r.c = 2")

    rewritten = transform.infer_between_predicates(query)

    assert conjuncts(rewritten) == {pred("r.a BETWEEN 1 AND 5"), pred("r.a = s.b"), pred("r.c = 2")}


@pytest.mark.parametrize(
    "condition",
    ["r.a >= 1 AND r.c = 2", "r.a <= 5 AND r.c = 2", "r.a >= 1 AND r.b <= 5"],
    ids=["lower-bound-only", "upper-bound-only", "bounds-on-different-columns"],
)
def test_infer_between_predicates_leaves_open_ranges_alone(condition: str) -> None:
    query = select(f"SELECT * FROM r WHERE {condition}")

    rewritten = transform.infer_between_predicates(query)

    assert conjuncts(rewritten) == conjuncts(query)


def test_infer_between_predicates_returns_a_query_without_where_clause_unchanged() -> None:
    query = select("SELECT * FROM r")

    assert transform.infer_between_predicates(query) is query


def test_infer_between_predicates_rewrites_ranges_below_a_disjunction() -> None:
    query = select("SELECT * FROM r WHERE (r.a >= 1 AND r.a <= 5) OR r.b = 1")

    rewritten = transform.infer_between_predicates(query)

    root = where_root(rewritten)
    assert isinstance(root, OrPredicate)
    assert set(root.children) == {pred("r.a BETWEEN 1 AND 5"), pred("r.b = 1")}


def test_infer_between_predicates_rewrites_ranges_below_a_negation() -> None:
    query = select("SELECT * FROM r WHERE NOT (r.a >= 1 AND r.a <= 5)")

    rewritten = transform.infer_between_predicates(query)

    assert where_root(rewritten) == NotPredicate(pred("r.a BETWEEN 1 AND 5"))


@pytest.mark.parametrize(
    "condition",
    ["1 <= r.a AND r.a <= 5", "r.a >= 1 AND 5 >= r.a", "1 <= r.a AND 5 >= r.a"],
    ids=["lower-bound-mirrored", "upper-bound-mirrored", "both-mirrored"],
)
def test_infer_between_predicates_merges_bounds_with_the_column_on_the_right(condition: str) -> None:
    """Regression guard for `_BetweenPredCreator` swapping the operands of a bound with the column on the right
    without mirroring the operator: ``1 <= r.a`` was recorded as the upper bound ``r.a <= 1``.
    """
    query = select(f"SELECT * FROM r WHERE {condition}")

    rewritten = transform.infer_between_predicates(query)

    assert conjuncts(rewritten) == {pred("r.a BETWEEN 1 AND 5")}


@pytest.mark.parametrize(
    ("condition", "expected_bounds"),
    [
        ("r.a <= 1 AND r.a <= 5 AND r.a >= 0", {("<=", 1), ("<=", 5), (">=", 0)}),
        ("r.a >= 1 AND r.a >= 2 AND r.a <= 5", {(">=", 1), (">=", 2), ("<=", 5)}),
        ("r.a <= 1 AND 5 >= r.a AND r.a >= 0", {("<=", 1), ("<=", 5), (">=", 0)}),
        ("r.a <= 1 AND r.a <= 5 AND r.a >= 0 AND r.a >= 2", {("<=", 1), ("<=", 5), (">=", 0), (">=", 2)}),
    ],
    ids=["upper", "lower", "upper-mirrored", "both-directions"],
)
def test_infer_between_predicates_keeps_every_bound_of_the_same_direction(
    condition: str, expected_bounds: set[tuple[str, int]]
) -> None:
    """Regression guard for `_BetweenPredCreator` only blocking a column with a second bound of the same direction,
    without keeping that bound: one of the two bounds silently disappeared from the query.

    Whether a surviving bound is rendered as ``r.a <= 5`` or kept as the original ``5 >= r.a`` depends on the
    (hash-based) child order after `flatten_and_predicate`, so the bounds are compared in their simplified form.
    """
    query = select(f"SELECT * FROM r WHERE {condition}")

    rewritten = transform.infer_between_predicates(query)

    bounds = [SimpleFilter.wrap(child) for child in conjuncts(rewritten)]
    assert all(bound.column == col("a", R) for bound in bounds)
    assert len(bounds) == len(expected_bounds)
    assert {(bound.operation.value, bound.value) for bound in bounds} == expected_bounds


@pytest.mark.parametrize(
    "bound",
    ["r.a + 1 <= 5", "CAST(r.a AS integer) <= 5", "upper(r.a) >= 'x'"],
    ids=["arithmetic", "cast", "function"],
)
def test_infer_between_predicates_keeps_a_bound_on_a_computed_expression(bound: str) -> None:
    """Regression guard for `_BetweenPredCreator` dropping a ``<=`` / ``>=`` child without a bare column operand:
    it used to ``break`` out of the loop (losing every later conjunct as well), and afterwards simply never
    re-added the child.
    """
    query = select(f"SELECT * FROM r WHERE r.b = 3 AND {bound} AND r.c = 4")

    rewritten = transform.infer_between_predicates(query)

    assert conjuncts(rewritten) == {pred("r.b = 3"), pred(bound), pred("r.c = 4")}


# -- as_star_query / as_count_star_query ----------------------------------------------------------------------


def test_as_star_query_replaces_only_the_select_clause() -> None:
    query = select("SELECT r.a FROM r WHERE r.a = 1 ORDER BY r.a")

    star = transform.as_star_query(query)

    assert star == select("SELECT * FROM r WHERE r.a = 1 ORDER BY r.a")
    assert query.select_clause == select("SELECT r.a FROM r").select_clause


def test_as_star_query_returns_a_set_query_unchanged() -> None:
    assert transform.as_star_query(UNION) is UNION


def test_as_count_star_query_replaces_the_select_clause() -> None:
    query = select("SELECT r.a FROM r WHERE r.a = 1")

    assert transform.as_count_star_query(query) == select("SELECT COUNT(*) FROM r WHERE r.a = 1")


def test_as_count_star_query_wraps_a_set_query_in_a_subquery() -> None:
    counted = transform.as_count_star_query(UNION)

    assert isinstance(counted, SelectStatement)
    assert counted.select_clause == Select.count_star()
    [source] = from_items(counted)
    assert isinstance(source, SubqueryTableSource)
    assert source.query == UNION


# -- drop_hints -----------------------------------------------------------------------------------------------


def test_drop_hints_removes_the_hint_block() -> None:
    query = select("/*+ SeqScan(r) */ SELECT * FROM r")

    dropped = transform.drop_hints(query)

    assert dropped.hints is None
    assert dropped == select("SELECT * FROM r")


def test_drop_hints_returns_a_query_without_hints_unchanged() -> None:
    query = select("SELECT * FROM r")

    assert transform.drop_hints(query) is query


def test_drop_hints_can_remove_only_the_preparatory_statements() -> None:
    query = transform.add_clause(select("SELECT * FROM r"), Hint("SET enable_nestloop = off;", "/*+ SeqScan(r) */"))

    dropped = transform.drop_hints(query, preparatory_statements_only=True)

    assert dropped.hints == Hint("", "/*+ SeqScan(r) */")


# -- as_explain / as_explain_analyze --------------------------------------------------------------------------


def test_as_explain_adds_a_plain_explain_by_default() -> None:
    explained = transform.as_explain(select("SELECT * FROM r"))

    assert explained.explain == Explain.plan()
    assert explained.explain is not None
    assert not explained.explain.analyze


def test_as_explain_uses_the_given_explain_block() -> None:
    explain = Explain(analyze=True, target_format="TEXT")

    explained = transform.as_explain(select("SELECT * FROM r"), explain)

    assert explained.explain == explain


def test_as_explain_supports_set_queries() -> None:
    explained = transform.as_explain(UNION)

    assert isinstance(explained, SetQuery)
    assert explained.explain == Explain.plan()


def test_as_explain_analyze_adds_an_analyze_block() -> None:
    explained = transform.as_explain_analyze(select("SELECT * FROM r"))

    assert explained.explain == Explain.explain_analyze()
    assert explained.explain is not None
    assert explained.explain.analyze


# -- remove_predicate -----------------------------------------------------------------------------------------


def test_remove_predicate_of_none_is_none() -> None:
    assert transform.remove_predicate(None, pred("r.a = 1")) is None


def test_remove_predicate_removing_the_whole_predicate_is_none() -> None:
    assert transform.remove_predicate(pred("r.a = 1"), pred("r.a = 1")) is None


def test_remove_predicate_keeps_an_unrelated_base_predicate() -> None:
    predicate = pred("r.a = 1")

    assert transform.remove_predicate(predicate, pred("r.b = 2")) is predicate


@pytest.mark.parametrize("compound", ["r.a = 1 AND r.b = 2", "r.a = 1 OR r.b = 2"], ids=["conjunction", "disjunction"])
def test_remove_predicate_unwraps_a_compound_with_a_single_remaining_child(compound: str) -> None:
    assert transform.remove_predicate(pred(compound), pred("r.b = 2")) == pred("r.a = 1")


def test_remove_predicate_keeps_the_remaining_children_of_a_larger_conjunction() -> None:
    remaining = transform.remove_predicate(pred("r.a = 1 AND r.b = 2 AND r.c = 3"), pred("r.b = 2"))

    assert isinstance(remaining, AndPredicate)
    assert set(remaining.children) == {pred("r.a = 1"), pred("r.c = 3")}


def test_remove_predicate_removes_a_nested_predicate() -> None:
    remaining = transform.remove_predicate(pred("r.a = 1 AND (r.b = 2 OR r.c = 3)"), pred("r.c = 3"))

    assert isinstance(remaining, AndPredicate)
    assert set(remaining.children) == {pred("r.a = 1"), pred("r.b = 2")}


def test_remove_predicate_drops_a_negation_whose_child_is_removed() -> None:
    assert transform.remove_predicate(pred("NOT r.a = 1"), pred("r.a = 1")) is None


def test_remove_predicate_keeps_a_negation_around_the_remaining_child() -> None:
    remaining = transform.remove_predicate(pred("NOT (r.a = 1 AND r.b = 2)"), pred("r.b = 2"))

    assert remaining == NotPredicate(pred("r.a = 1"))


# -- add_clause / drop_clause / replace_clause ----------------------------------------------------------------


def test_add_clause_adds_a_missing_clause() -> None:
    query = select("SELECT * FROM r")

    limited = transform.add_clause(query, Limit(limit=10))

    assert limited.limit_clause == Limit(limit=10)
    assert query.limit_clause is None


def test_add_clause_overwrites_a_clause_of_the_same_type() -> None:
    query = select("SELECT * FROM r LIMIT 5")

    limited = transform.add_clause(query, Limit(limit=10))

    assert limited.limit_clause == Limit(limit=10)


def test_add_clause_accepts_several_clauses() -> None:
    where = Where(pred("r.a = 1"))

    updated = transform.add_clause(select("SELECT * FROM r"), [where, Limit(limit=10)])

    assert updated.where_clause == where
    assert updated.limit_clause == Limit(limit=10)


@pytest.mark.parametrize(
    "description",
    [Limit, Limit(limit=1), [Limit, OrderBy.create_for([ColumnExpression(col("a", R))])]],
    ids=["by-type", "by-instance-of-a-different-value", "mixed-iterable"],
)
def test_drop_clause_removes_clauses_by_type(description) -> None:
    query = select("SELECT * FROM r WHERE r.a = 1 LIMIT 5")

    dropped = transform.drop_clause(query, description)

    assert dropped == select("SELECT * FROM r WHERE r.a = 1")


def test_drop_clause_can_remove_several_clause_types_at_once() -> None:
    query = select("SELECT * FROM r WHERE r.a = 1 ORDER BY r.a LIMIT 5")

    dropped = transform.drop_clause(query, [OrderBy, Limit, Where])

    assert dropped == select("SELECT * FROM r")


def test_replace_clause_uses_the_replacement_instead_of_the_original() -> None:
    query = select("SELECT * FROM r WHERE r.a = 1")

    replaced = transform.replace_clause(query, Where(pred("r.a = 2")))

    assert replaced == select("SELECT * FROM r WHERE r.a = 2")


def test_replace_clause_ignores_replacements_for_clauses_not_in_the_query() -> None:
    query = select("SELECT * FROM r")

    replaced = transform.replace_clause(query, [Where(pred("r.a = 2")), Select.count_star()])

    assert replaced == select("SELECT COUNT(*) FROM r")


# -- replace_expressions / replace_predicate ------------------------------------------------------------------


def _rename_r_a(expression: SqlExpression) -> SqlExpression:
    if isinstance(expression, ColumnExpression) and expression.column == col("a", R):
        return ColumnExpression(col("z", R))
    return expression


def test_replace_expressions_with_the_identity_reproduces_the_query() -> None:
    query = select("SELECT r.a, count(*) FROM r, s WHERE r.a = s.b GROUP BY r.a HAVING count(*) > 1 ORDER BY r.a")

    assert transform.replace_expressions(query, lambda expr: expr) == query


def test_replace_expressions_applies_the_replacement_in_every_clause() -> None:
    query = select("SELECT r.a FROM r WHERE r.a = 1 GROUP BY r.a ORDER BY r.a")

    replaced = transform.replace_expressions(query, _rename_r_a)

    assert replaced == select("SELECT r.z FROM r WHERE r.z = 1 GROUP BY r.z ORDER BY r.z")


def test_replace_expressions_only_passes_top_level_expressions_to_the_replacement() -> None:
    """Nested expressions are the replacement's own responsibility (see the formatter's prettifier)."""
    seen: list[SqlExpression] = []

    def record(expression: SqlExpression) -> SqlExpression:
        seen.append(expression)
        return _rename_r_a(expression)

    replaced = transform.replace_expressions(select("SELECT count(r.a) FROM r"), record)

    assert replaced == select("SELECT count(r.a) FROM r")
    assert [type(expr) for expr in seen] == [FunctionExpression]


def test_replace_expressions_descends_into_subqueries_in_the_from_clause() -> None:
    query = select("SELECT * FROM (SELECT r.a FROM r WHERE r.a = 1) AS sub")

    replaced = transform.replace_expressions(query, _rename_r_a)

    [subquery] = replaced.subqueries()
    assert subquery == select("SELECT r.z FROM r WHERE r.z = 1")


def test_replace_expressions_descends_into_both_sides_of_a_set_query() -> None:
    query = parse("SELECT r.a FROM r UNION SELECT r.a FROM r WHERE r.a = 1")

    replaced = transform.replace_expressions(query, _rename_r_a)

    assert replaced == parse("SELECT r.z FROM r UNION SELECT r.z FROM r WHERE r.z = 1")


def test_replace_expressions_rejects_non_static_values_in_a_values_cte() -> None:
    query = select("WITH v(x) AS (VALUES (1)) SELECT * FROM v")

    def to_column(expression: SqlExpression) -> SqlExpression:
        return ColumnExpression(col("a", R)) if isinstance(expression, StaticValueExpression) else expression

    with pytest.raises(ValueError, match="non-static values"):
        transform.replace_expressions(query, to_column)


def test_replace_predicate_replaces_a_where_predicate() -> None:
    query = select("SELECT * FROM r, s WHERE r.a = s.b AND r.c = 1")

    replaced = transform.replace_predicate(query, pred("r.c = 1"), pred("r.c = 2"))

    assert replaced == select("SELECT * FROM r, s WHERE r.a = s.b AND r.c = 2")


# -- rename_columns_in_expression / _predicate / _clause / _query ---------------------------------------------

RENAME_A = {col("a", R): col("z", R)}


def test_rename_columns_in_expression_of_none_is_none() -> None:
    assert transform.rename_columns_in_expression(None, RENAME_A) is None


def test_rename_columns_in_expression_leaves_unmapped_columns_alone() -> None:
    expression = ColumnExpression(col("b", R))

    assert transform.rename_columns_in_expression(expression, RENAME_A) is expression


@pytest.mark.parametrize(
    ("sql", "expected"),
    [
        ("SELECT r.a FROM r", "SELECT r.z FROM r"),
        ("SELECT CAST(r.a AS text) FROM r", "SELECT CAST(r.z AS text) FROM r"),
        ("SELECT count(DISTINCT r.a) FROM r", "SELECT count(DISTINCT r.z) FROM r"),
        ("SELECT CASE WHEN r.a = 1 THEN r.a ELSE 0 END FROM r", "SELECT CASE WHEN r.z = 1 THEN r.z ELSE 0 END FROM r"),
        ("SELECT sum(r.a) OVER (PARTITION BY r.a) FROM r", "SELECT sum(r.z) OVER (PARTITION BY r.z) FROM r"),
        ("SELECT (SELECT max(r.a) FROM r) FROM r", "SELECT (SELECT max(r.z) FROM r) FROM r"),
    ],
    ids=["column", "cast", "function", "case", "window", "subquery"],
)
def test_rename_columns_in_expression_renames_nested_columns(sql: str, expected: str) -> None:
    [projection] = select(sql).select_clause.targets
    [expected_projection] = select(expected).select_clause.targets

    renamed = transform.rename_columns_in_expression(projection.expression, RENAME_A)

    assert renamed == expected_projection.expression


def test_rename_columns_in_expression_renames_columns_in_arithmetic_expressions() -> None:
    [projection] = select("SELECT r.a + 1 FROM r").select_clause.targets
    [expected_projection] = select("SELECT r.z + 1 FROM r").select_clause.targets

    renamed = transform.rename_columns_in_expression(projection.expression, RENAME_A)

    assert renamed == expected_projection.expression


def test_deprecated_rename_columns_in_expression_alias_warns_and_delegates() -> None:
    with pytest.warns(FutureWarning, match="deprecated"):
        renamed = transform._rename_columns_in_expression(ColumnExpression(col("a", R)), dict(RENAME_A))

    assert renamed == ColumnExpression(col("z", R))


def test_rename_columns_in_predicate_of_none_is_none() -> None:
    assert transform.rename_columns_in_predicate(None, RENAME_A) is None


@pytest.mark.parametrize(
    ("condition", "expected"),
    [
        ("r.a = s.b", "r.z = s.b"),
        ("r.a IN (1, 2)", "r.z IN (1, 2)"),
        ("r.a IS NULL", "r.z IS NULL"),
        ("r.a = 1 AND (r.a = 2 OR s.b = 3)", "r.z = 1 AND (r.z = 2 OR s.b = 3)"),
    ],
    ids=["binary", "in", "unary", "and-or"],
)
def test_rename_columns_in_predicate_renames_every_predicate_type(condition: str, expected: str) -> None:
    assert transform.rename_columns_in_predicate(pred(condition), RENAME_A) == pred(expected)


def test_rename_columns_in_predicate_renames_between_bounds() -> None:
    renamed = transform.rename_columns_in_predicate(pred("r.a BETWEEN 1 AND r.b"), RENAME_A)

    assert renamed == pred("r.z BETWEEN 1 AND r.b")


def test_rename_columns_in_predicate_renames_a_negated_predicate() -> None:
    renamed = transform.rename_columns_in_predicate(pred("NOT r.a = 1"), RENAME_A)

    assert renamed == pred("NOT r.z = 1")


def test_rename_columns_in_predicate_keeps_the_not_in_operator() -> None:
    """The operator is asserted directly because `InPredicate.__eq__` ignores it (pinned separately in
    ``test_qal_predicates.py``), so comparing against a parsed ``NOT IN`` predicate would not catch a regression
    that silently turns it into `IN`.
    """
    original = pred("r.a NOT IN (1, 2)")
    assert isinstance(original, InPredicate)
    assert original.operator == BinaryOperator.NotIn

    renamed = transform.rename_columns_in_predicate(original, RENAME_A)

    assert isinstance(renamed, InPredicate)
    assert renamed.column == ColumnExpression(col("z", R))
    assert renamed.operator == BinaryOperator.NotIn


def test_rename_columns_in_clause_of_none_is_none() -> None:
    assert transform.rename_columns_in_clause(None, RENAME_A) is None


@pytest.mark.parametrize(
    "clause",
    [Limit(limit=1), Explain.plan(), Hint("", "/*+ SeqScan(r) */")],
    ids=["limit", "explain", "hint"],
)
def test_rename_columns_in_clause_returns_column_free_clauses_unchanged(clause) -> None:
    assert transform.rename_columns_in_clause(clause, RENAME_A) is clause


def test_rename_columns_in_clause_keeps_projection_aliases() -> None:
    renamed = transform.rename_columns_in_clause(select("SELECT r.a AS x FROM r").select_clause, RENAME_A)

    assert renamed == select("SELECT r.z AS x FROM r").select_clause


def test_rename_columns_in_clause_renames_columns_of_a_values_cte() -> None:
    values = TableReference.create_virtual("v")
    cte = select("WITH v(x) AS (VALUES (1)) SELECT * FROM v").cte_clause

    renamed = transform.rename_columns_in_clause(cte, {col("x", values): col("y", values)})

    assert isinstance(renamed, CommonTableExpression)
    [values_cte] = renamed.queries
    assert isinstance(values_cte, ValuesWithQuery)
    assert [c.name for c in values_cte.cols] == ["y"]


def test_rename_columns_in_clause_moves_a_values_cte_to_another_table() -> None:
    values = TableReference.create_virtual("v")
    cte = select("WITH v(x) AS (VALUES (1)) SELECT * FROM v").cte_clause

    renamed = transform.rename_columns_in_clause(cte, {col("x", values): col("x", R)})

    assert isinstance(renamed, CommonTableExpression)
    [values_cte] = renamed.queries
    assert isinstance(values_cte, ValuesWithQuery)
    assert values_cte.cols == (col("x", R),)


def test_rename_columns_in_clause_keeps_an_unaffected_values_cte() -> None:
    query = select("WITH v(x) AS (VALUES (1), (2)) SELECT * FROM r, v WHERE r.a = v.x")

    renamed = transform.rename_columns_in_query(query, RENAME_A)

    assert renamed == select("WITH v(x) AS (VALUES (1), (2)) SELECT * FROM r, v WHERE r.z = v.x")


def test_rename_columns_in_query_renames_every_clause() -> None:
    query = select(
        "WITH c AS (SELECT r.a FROM r) SELECT r.a FROM r JOIN c ON r.a = c.a "
        "WHERE r.a = 1 GROUP BY r.a HAVING min(r.a) > 1 ORDER BY r.a"
    )

    renamed = transform.rename_columns_in_query(query, RENAME_A)

    assert col("a", R) not in renamed.columns()
    assert renamed == select(
        "WITH c AS (SELECT r.z FROM r) SELECT r.z FROM r JOIN c ON r.z = c.a "
        "WHERE r.z = 1 GROUP BY r.z HAVING min(r.z) > 1 ORDER BY r.z"
    )


def test_rename_columns_in_query_renames_both_sides_of_a_set_query() -> None:
    renamed = transform.rename_columns_in_query(parse("SELECT r.a FROM r UNION SELECT r.a FROM r"), RENAME_A)

    assert renamed == parse("SELECT r.z FROM r UNION SELECT r.z FROM r")


# -- rename_table ---------------------------------------------------------------------------------------------

X = TableReference("x")


def test_rename_table_replaces_the_table_and_its_columns() -> None:
    query = select("SELECT r.a FROM r, s WHERE r.a = s.b")

    renamed = transform.rename_table(query, R, X)

    assert renamed == select("SELECT x.a FROM x, s WHERE x.a = s.b")
    assert query.tables() == {R, S}


def test_rename_table_accepts_a_dict_of_renamings() -> None:
    query = select("SELECT r.a FROM r, s WHERE r.a = s.b")

    renamed = transform.rename_table(query, {R: X, S: T})

    assert renamed == select("SELECT x.a FROM x, t WHERE x.a = t.b")


def test_rename_table_can_prefix_column_names_with_the_old_table() -> None:
    query = select("SELECT r.a FROM r, s WHERE r.a = s.b")

    renamed = transform.rename_table(query, R, X, prefix_column_names=True)

    assert renamed == select("SELECT x.r_a FROM x, s WHERE x.r_a = s.b")


def test_rename_table_requires_a_target_for_a_single_table() -> None:
    with pytest.raises(ValueError, match="both from_table and target_table must be set"):
        transform.rename_table(select("SELECT * FROM r"), R)


def test_rename_table_renames_inside_explicit_joins_and_subqueries() -> None:
    query = select("SELECT r.* FROM r JOIN s ON r.a = s.b WHERE r.a IN (SELECT r.a FROM r)")

    renamed = transform.rename_table(query, R, X)

    assert renamed == select("SELECT x.* FROM x JOIN s ON x.a = s.b WHERE x.a IN (SELECT x.a FROM x)")


def test_rename_table_merges_into_a_target_that_is_already_in_the_query() -> None:
    query = select("SELECT * FROM r, s WHERE r.a = s.b")

    renamed = transform.rename_table(query, R, S)

    assert renamed == select("SELECT * FROM s WHERE s.a = s.b")


# -- merge_tables ---------------------------------------------------------------------------------------------

MAT_VIEW = TableReference("mv")


def test_merge_tables_of_a_single_table_is_a_renaming() -> None:
    query = select("SELECT r.a FROM r, s WHERE r.a = s.b")

    merged = transform.merge_tables(query, [R], target=MAT_VIEW)

    assert merged == select("SELECT mv.a FROM mv, s WHERE mv.a = s.b")


def test_merge_tables_merges_more_than_one_table_of_an_implicit_from_clause() -> None:
    query = select("SELECT * FROM r, s WHERE r.a = s.b AND r.c = 1")

    merged = transform.merge_tables(query, [R, S], target=MAT_VIEW)

    assert merged == select("SELECT * FROM mv WHERE mv.a = mv.b AND mv.c = 1")


def test_merge_tables_rejects_tables_joined_by_an_explicit_join() -> None:
    """Documents the intended behaviour, not a bug"""
    query = select("SELECT * FROM r JOIN s ON r.a = s.b WHERE r.c = 1")

    with pytest.raises(ValueError, match="Cannot rename explicit JOINs"):
        transform.merge_tables(query, [R, S], target=MAT_VIEW)


# -- expand_select_star ---------------------------------------------------------------------------------------


def test_expand_select_star_lists_the_columns_of_every_table() -> None:
    query = select("SELECT * FROM r, s WHERE r.a = s.b")

    expanded = transform.expand_select_star(query, schema=SCHEMA)

    assert expanded == select("SELECT r.a, r.id, s.b, s.id FROM r, s WHERE r.a = s.b")


def test_expand_select_star_expands_a_table_star_to_that_table_only() -> None:
    expanded = transform.expand_select_star(select("SELECT s.* FROM r, s"), schema=SCHEMA)

    assert expanded.select_clause == select("SELECT s.b, s.id FROM s").select_clause


def test_expand_select_star_keeps_explicit_columns_and_aliases_and_distinct() -> None:
    query = select("SELECT DISTINCT r.a, r.a + 1 AS x FROM r")

    assert transform.expand_select_star(query, schema=SCHEMA) == query


def test_expand_select_star_expands_both_sides_of_a_set_query() -> None:
    query = parse("SELECT * FROM r UNION ALL SELECT * FROM s")

    expanded = transform.expand_select_star(query, schema=SCHEMA)

    assert expanded == parse("SELECT r.a, r.id FROM r UNION ALL SELECT s.b, s.id FROM s")


def test_expand_select_star_rejects_a_function_table_source() -> None:
    query = select("SELECT * FROM generate_series(1, 3) AS g")

    with pytest.raises(ValueError, match="function table source"):
        transform.expand_select_star(query, schema=SCHEMA)


def test_expand_select_star_falls_back_to_the_schema_of_the_current_database() -> None:
    """The pool is restored by the autouse `isolated_database_pool` fixture."""
    DatabasePool.get_instance().register_database("fake", FakeDatabase(schema_spec={"r": ["a", "id"]}))

    expanded = transform.expand_select_star(select("SELECT * FROM r"), schema=None)

    assert expanded == select("SELECT r.a, r.id FROM r")


def test_expand_select_star_over_a_cte_references_the_columns_inside_the_cte() -> None:
    query = select("WITH c AS (SELECT 42 AS foo, r.a FROM r) SELECT * FROM c")

    expanded = transform.expand_select_star(query, schema=SCHEMA)

    assert expanded.select_clause == select("WITH c AS (SELECT 42) SELECT c.foo, c.a FROM c").select_clause


def test_expand_select_star_keeps_top_level_unnamed_expressions() -> None:
    expanded = transform.expand_select_star(select("SELECT r.a, count(*) FROM r"), schema=SCHEMA)

    [_, placeholder] = expanded.select_clause.targets
    assert placeholder == Projection.count_star()


# -- expand_natural_joins -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("natural", "expected_type"),
    [
        ("NATURAL JOIN", JoinType.InnerJoin),
        ("NATURAL LEFT JOIN", JoinType.LeftJoin),
        ("NATURAL FULL JOIN", JoinType.OuterJoin),
    ],
    ids=["inner", "left", "full"],
)
def test_expand_natural_joins_joins_on_the_shared_columns(natural: str, expected_type: JoinType) -> None:
    query = select(f"SELECT * FROM r {natural} s")

    expanded = transform.expand_natural_joins(query, schema=SCHEMA)

    [join] = from_items(expanded)
    assert isinstance(join, JoinTableSource)
    assert join.join_type == expected_type
    assert join.join_condition == pred("r.id = s.id")


def test_expand_natural_joins_turns_a_natural_right_join_into_a_right_join() -> None:
    query = select("SELECT * FROM r NATURAL RIGHT JOIN s")

    expanded = transform.expand_natural_joins(query, schema=SCHEMA)

    [rewritten] = from_items(expanded)
    assert isinstance(rewritten, JoinTableSource)
    assert rewritten.join_type == JoinType.RightJoin
    assert rewritten.join_condition == pred("r.id = s.id")


def test_expand_natural_joins_leaves_regular_joins_alone() -> None:
    query = select("SELECT * FROM r JOIN s ON r.a = s.b")

    assert transform.expand_natural_joins(query, schema=SCHEMA) == query


def test_expand_natural_joins_returns_a_query_without_from_clause_unchanged() -> None:
    query = select("SELECT 1")

    assert transform.expand_natural_joins(query, schema=SCHEMA) is query


def test_expand_natural_joins_expands_both_sides_of_a_set_query() -> None:
    query = parse("SELECT * FROM r NATURAL JOIN s UNION SELECT * FROM s NATURAL JOIN t")

    expanded = transform.expand_natural_joins(query, schema=SCHEMA)

    assert expanded == parse("SELECT * FROM r JOIN s ON r.id = s.id UNION SELECT * FROM s JOIN t ON s.id = t.id")


def test_expand_natural_joins_expands_a_chain_of_natural_joins() -> None:
    query = select("SELECT * FROM r NATURAL JOIN s NATURAL JOIN t")

    expanded = transform.expand_natural_joins(query, schema=SCHEMA)

    assert expanded.tables() == {R, S, T}
    assert join_pairs(expanded) == {
        frozenset({col("id", R), col("id", S)}),
        frozenset({col("id", S), col("id", T)}),
        frozenset({col("id", R), col("id", T)}),
    }


# -- normalize_query ------------------------------------------------------------------------------------------


def test_normalize_query_applies_all_normalizations() -> None:
    query = select("SELECT * FROM r JOIN s ON r.a = s.b JOIN t ON s.b = t.c WHERE r.id >= 1 AND r.id <= 5")

    normalized = transform.normalize_query(query, schema=SCHEMA)

    assert normalized.has_simple_from()
    assert normalized.select_clause == select("SELECT r.a, r.id, s.b, s.id, t.id, t.c FROM r, s, t").select_clause
    assert set(normalized.filters()) == {pred("r.id BETWEEN 1 AND 5")}
    assert join_pairs(normalized) == {
        frozenset({col("a", R), col("b", S)}),
        frozenset({col("b", S), col("c", T)}),
        frozenset({col("a", R), col("c", T)}),
    }
    assert all(not isinstance(child, AndPredicate) for child in conjuncts(normalized))


def test_normalize_query_normalizes_both_sides_of_a_set_query() -> None:
    query = parse("SELECT * FROM r UNION SELECT * FROM s")

    normalized = transform.normalize_query(query, schema=SCHEMA)

    assert normalized == parse("SELECT r.a, r.id FROM r UNION SELECT s.b, s.id FROM s")


def test_normalize_query_also_normalizes_the_ctes_of_a_select_statement() -> None:
    query = select("WITH c AS (SELECT * FROM r JOIN s ON r.a = s.b) SELECT c.a FROM c")

    normalized = transform.normalize_query(query, schema=SCHEMA)

    assert normalized.cte_clause is not None
    assert len(normalized.cte_clause.queries) == 1
    normalized_cte = normalized.cte_clause.queries[0].query
    assert normalized_cte.has_simple_from()
    assert normalized_cte.select_clause == select("SELECT r.a, r.id, s.b, s.id FROM r, s").select_clause
    assert join_pairs(normalized_cte) == {frozenset({col("a", R), col("b", S)})}


def test_normalize_query_rejects_outer_joins() -> None:
    query = select("SELECT * FROM r NATURAL LEFT JOIN s")

    with pytest.raises(ValueError, match="outer joins"):
        transform.normalize_query(query, schema=SCHEMA)


# -- public exposure ------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "flatten_and_predicate",
        "explicit_to_implicit",
        "extract_query_fragment",
        "extract_subquery",
        "move_into_subquery",
        "add_ec_predicates",
        "infer_between_predicates",
        "as_star_query",
        "as_count_star_query",
        "drop_hints",
        "as_explain",
        "as_explain_analyze",
        "replace_expressions",
        "rename_table",
        "merge_tables",
        "expand_select_star",
        "expand_natural_joins",
        "normalize_query",
    ],
)
def test_transformations_are_reachable_through_the_public_package(name: str) -> None:
    assert getattr(pb.transform, name) is getattr(transform, name)
