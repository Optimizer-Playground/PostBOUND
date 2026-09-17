"""Tests for the `AbstractPredicate` hierarchy and `PredicateTree` in `postbound.qal`.

Predicates are obtained by parsing real SQL, the same rationale as `test_qal_expressions.py`: the parser is
the normal way to build these objects, so there is no "external undocumented shape" risk to avoid here.

`SimpleFilter`/`SimpleJoin` (`can_wrap`/`wrap`) are also tested directly at the qal level, in addition to
the indirect coverage they get through `validation.SPJCheck` -- this is their actual home.
"""

from __future__ import annotations

import pytest

from postbound import parser
from postbound._core import ColumnReference, TableReference
from postbound.qal import (
    AndPredicate,
    BetweenPredicate,
    BinaryOperator,
    BinaryPredicate,
    CompoundOperator,
    InPredicate,
    NotPredicate,
    OrPredicate,
    SimpleFilter,
    SimpleJoin,
    StaticValueExpression,
    UnaryPredicate,
    as_predicate,
    determine_join_equivalence_classes,
    generate_predicates_for_equivalence_classes,
)

R = TableReference("r")
S = TableReference("s")
T = TableReference("t")


def parse(sql: str):
    return parser.parse_query(sql, bind_columns=False)


def where(sql: str):
    """Parses `sql` and returns the WHERE clause's root predicate."""
    return parse(sql).where_clause.root


def static_value(expr) -> object:
    """Narrows `expr` to `StaticValueExpression` and returns its value, failing loudly otherwise."""
    assert isinstance(expr, StaticValueExpression)
    return expr.value


# -- BinaryPredicate ------------------------------------------------------------------------------------


def test_binary_predicate_captures_operator_and_operands() -> None:
    pred = where("SELECT * FROM r WHERE r.a = 42")

    assert isinstance(pred, BinaryPredicate)
    assert pred.operator == BinaryOperator.Equal
    assert pred.lhs.column == ColumnReference("a", R)
    assert pred.rhs.value == 42


@pytest.mark.parametrize(
    ("sql", "is_join"),
    [
        ("SELECT * FROM r, s WHERE r.a = s.b", True),
        ("SELECT * FROM r, s WHERE r.a < s.b", True),
        ("SELECT * FROM r, s WHERE r.a = 42", False),
    ],
)
def test_binary_predicate_join_vs_filter_classification(sql: str, is_join: bool) -> None:
    pred = where(sql)

    assert pred.is_join() is is_join
    assert pred.is_filter() is not is_join


def test_binary_predicate_udf_around_join_columns_is_still_a_join() -> None:
    """`some_udf(R.a, S.b) = 42` still spans both tables, so it counts as a join."""
    pred = where("SELECT * FROM r, s WHERE some_udf(r.a, s.b) = 42")

    assert pred.is_join() is True


def test_binary_predicate_udf_around_single_column_is_a_filter() -> None:
    pred = where("SELECT * FROM r, s WHERE some_udf(r.a) = 42")

    assert pred.is_filter() is True


def test_binary_predicate_cast_on_one_side_is_still_a_join() -> None:
    pred = where("SELECT * FROM r, s WHERE r.a = s.b::integer")

    assert pred.is_join() is True


def test_binary_predicate_with_independent_subquery_is_a_filter() -> None:
    pred = where("SELECT * FROM r WHERE r.a = (SELECT min(t.c) FROM t)")

    assert pred.is_filter() is True


def test_binary_predicate_with_dependent_subquery_is_still_a_filter() -> None:
    """A subquery correlated back to the outer query is a filter, not a join -- the outer reference does
    not turn this into a join predicate between `r` and `s`.
    """
    pred = where("SELECT * FROM r WHERE r.a = (SELECT min(t.c) FROM t WHERE t.c = r.b)")

    assert pred.is_filter() is True


def test_binary_predicate_columns_union_both_sides() -> None:
    pred = where("SELECT * FROM r, s WHERE r.a = s.b")

    assert pred.columns() == {ColumnReference("a", R), ColumnReference("b", S)}


# -- BetweenPredicate ------------------------------------------------------------------------------------


def test_between_predicate_captures_column_and_bounds() -> None:
    pred = where("SELECT * FROM r WHERE r.a BETWEEN 24 AND 42")

    assert isinstance(pred, BetweenPredicate)
    assert pred.column.column == ColumnReference("a", R)
    assert pred.lower.value == 24
    assert pred.upper.value == 42
    assert pred.is_filter() is True


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM r, s WHERE r.a BETWEEN 24 AND s.b",
        "SELECT * FROM r, s WHERE r.a BETWEEN s.b AND 42",
        "SELECT * FROM r, s WHERE r.a BETWEEN s.b AND s.b + 42",
    ],
)
def test_between_predicate_is_a_join_if_either_bound_references_another_table(sql: str) -> None:
    pred = where(sql)

    assert pred.is_join() is True


def test_between_predicate_with_subquery_bound_is_a_filter() -> None:
    pred = where("SELECT * FROM r WHERE r.a BETWEEN 24 AND (SELECT min(t.c) FROM t)")

    assert pred.is_filter() is True


# -- InPredicate -----------------------------------------------------------------------------------------


def test_in_predicate_captures_column_and_values() -> None:
    pred = where("SELECT * FROM r WHERE r.a IN (1, 2, 3)")

    assert isinstance(pred, InPredicate)
    assert pred.column.column == ColumnReference("a", R)
    assert [static_value(v) for v in pred.values] == [1, 2, 3]
    assert pred.is_filter() is True


def test_in_predicate_with_a_column_in_the_values_list_is_a_join() -> None:
    pred = where("SELECT * FROM r, s WHERE r.a IN (1, s.b, 3)")

    assert pred.is_join() is True


def test_in_predicate_with_independent_subquery_is_a_filter() -> None:
    pred = where("SELECT * FROM r, s WHERE r.a IN (SELECT t.b FROM t WHERE t.c = 42)")

    assert pred.is_filter() is True
    assert pred.is_subquery_predicate() is True


def test_in_predicate_with_dependent_subquery_is_still_a_filter() -> None:
    pred = where("SELECT * FROM r, s WHERE r.a IN (SELECT t.b FROM t WHERE t.c = t.d)")

    assert pred.is_filter() is True


# -- UnaryPredicate --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("sql", "checker"),
    [
        ("SELECT * FROM r WHERE r.a IS NULL", "is_null_test"),
        ("SELECT * FROM r WHERE r.a IS NOT NULL", "is_null_test"),
    ],
)
def test_unary_predicate_null_tests(sql: str, checker: str) -> None:
    pred = where(sql)

    assert isinstance(pred, UnaryPredicate)
    assert getattr(pred, checker)() is True
    assert pred.is_filter() is True


def test_unary_predicate_exists_for_a_dependent_subquery() -> None:
    pred = where("SELECT * FROM r WHERE EXISTS (SELECT * FROM s WHERE r.a = s.b)")

    assert pred.is_exists() is True
    assert pred.is_filter() is True


def test_unary_predicate_not_exists_for_a_dependent_subquery() -> None:
    """`NOT EXISTS (...)` parses as a `NotPredicate` wrapping the `UnaryPredicate` EXISTS test."""
    pred = where("SELECT * FROM r WHERE NOT EXISTS (SELECT * FROM s WHERE r.a = s.b)")

    assert isinstance(pred, NotPredicate)
    assert pred.child.is_exists() is True
    assert pred.is_filter() is True


def test_unary_predicate_single_column_udf_is_a_filter() -> None:
    pred = where("SELECT * FROM r, s WHERE my_udf(r.a)")

    assert pred.is_filter() is True


def test_unary_predicate_two_column_udf_is_a_join() -> None:
    pred = where("SELECT * FROM r, s WHERE my_udf(r.a, s.b)")

    assert pred.is_join() is True


# -- Compound predicates (And/Or/Not) --------------------------------------------------------------------


def test_and_predicate_flattens_multiple_conjuncts() -> None:
    pred = where("SELECT * FROM r WHERE r.a = 1 AND r.b = 2 AND r.c = 3")

    assert isinstance(pred, AndPredicate)
    assert pred.operation == CompoundOperator.And
    assert len(pred.children) == 3


def test_and_of_udf_predicates_produces_no_joins_only_filters() -> None:
    """Every conjunct here is a single-table UDF or literal-membership filter -- none of them span two
    tables, so the whole conjunction contains zero joins.
    """
    query = parse(
        "SELECT * FROM r WHERE r.a = r.b AND my_udf_pred1(r.a) AND my_udf_pred2(r.a, r.c) AND r.a IN (24, r.b, 42)"
    )

    assert query.joins() == set()
    assert len(query.filters()) == 4


def test_or_predicate_takes_precedence_in_parsing() -> None:
    """`a OR b AND c` parses as `a OR (b AND c)` -- the OR is the root, matching normal operator precedence."""
    query = parse(
        "SELECT * FROM t JOIN mi ON t.id = mi.movie_id "
        "WHERE t.production_year < 2021 OR t.season_nr > 10 AND mi.info LIKE '%cartoon%'"
    )

    assert isinstance(query.where_clause.root, OrPredicate)


def test_not_predicate_wraps_a_single_child() -> None:
    pred = where("SELECT * FROM r WHERE NOT r.a = 1")

    assert isinstance(pred, NotPredicate)
    assert pred.child == as_predicate(ColumnReference("a", R), "=", 1)


def test_compound_predicate_is_join_if_any_child_is_a_join() -> None:
    pred = where("SELECT * FROM r, s WHERE r.a = 1 AND r.b = s.c")

    assert pred.is_compound() is True
    assert pred.is_join() is True


def test_compound_predicate_base_predicates_recurses_into_children() -> None:
    pred = where("SELECT * FROM r WHERE (r.a = 1 AND r.b = 2) OR r.c = 3")

    bases = set(pred.base_predicates())
    assert len(bases) == 3
    assert all(isinstance(base, BinaryPredicate) for base in bases)


# -- as_predicate -----------------------------------------------------------------------------------------


def test_as_predicate_builds_a_binary_predicate() -> None:
    col = ColumnReference("a", R)

    pred = as_predicate(col, "=", 42)

    assert isinstance(pred, BinaryPredicate)
    assert pred.operator == BinaryOperator.Equal


def test_as_predicate_normalizes_alternate_operator_spellings() -> None:
    col = ColumnReference("a", R)

    assert as_predicate(col, "!=", 1).operator == BinaryOperator.NotEqual
    assert as_predicate(col, "==", 1).operator == BinaryOperator.Equal


def test_as_predicate_between_accepts_two_positional_arguments() -> None:
    col = ColumnReference("a", R)

    pred = as_predicate(col, "between", 1, 10)

    assert isinstance(pred, BetweenPredicate)
    assert isinstance(pred.lower, StaticValueExpression) and isinstance(pred.upper, StaticValueExpression)
    assert pred.lower.value == 1 and pred.upper.value == 10


def test_as_predicate_between_accepts_a_single_tuple_argument() -> None:
    col = ColumnReference("a", R)

    pred = as_predicate(col, "between", (1, 10))

    assert isinstance(pred, BetweenPredicate)
    assert isinstance(pred.lower, StaticValueExpression) and isinstance(pred.upper, StaticValueExpression)
    assert pred.lower.value == 1 and pred.upper.value == 10


def test_as_predicate_in_accepts_varargs() -> None:
    col = ColumnReference("a", R)

    pred = as_predicate(col, "in", 1, 2, 3)

    assert isinstance(pred, InPredicate)
    assert [static_value(v) for v in pred.values] == [1, 2, 3]


def test_as_predicate_in_flattens_a_nested_iterable() -> None:
    col = ColumnReference("a", R)

    pred = as_predicate(col, "in", [1, 2, 3])

    assert isinstance(pred, InPredicate)
    assert [static_value(v) for v in pred.values] == [1, 2, 3]


def test_as_predicate_unary_operator_takes_no_arguments() -> None:
    col = ColumnReference("a", R)

    pred = as_predicate(col, "is null")

    assert isinstance(pred, UnaryPredicate)
    assert pred.is_null_test() is True


def test_as_predicate_unary_operator_rejects_arguments() -> None:
    col = ColumnReference("a", R)

    with pytest.raises(ValueError, match="does not accept any arguments"):
        as_predicate(col, "is null", 1)


def test_as_predicate_binary_rejects_wrong_argument_count() -> None:
    col = ColumnReference("a", R)

    with pytest.raises(ValueError, match="Too many arguments"):
        as_predicate(col, "=", 1, 2)


def test_as_predicate_rejects_an_unknown_operator() -> None:
    col = ColumnReference("a", R)

    with pytest.raises(ValueError, match="Unknown operator"):
        as_predicate(col, "does-not-exist")


# -- SimpleFilter / SimpleJoin ------------------------------------------------------------------------------


def test_simple_filter_wrap_exposes_column_operation_and_value() -> None:
    pred = where("SELECT * FROM r WHERE r.a = 42")

    wrapped = SimpleFilter.wrap(pred)

    assert wrapped.column == ColumnReference("a", R)
    assert wrapped.operation == BinaryOperator.Equal
    assert wrapped.value == 42


def test_simple_filter_can_wrap_a_plain_equality() -> None:
    pred = where("SELECT * FROM r WHERE r.a = 42")

    assert SimpleFilter.can_wrap(pred) is True


def test_simple_filter_cannot_wrap_a_function_call_predicate() -> None:
    """A filter with a function call on either side (e.g. `upper(r.a) = 'X'`) cannot be represented as a
    `SimpleFilter` -- there is no single column being compared directly against a plain value.
    """
    pred = where("SELECT * FROM r WHERE upper(r.a) = 'X'")

    assert SimpleFilter.can_wrap(pred) is False


def test_simple_join_wrap_exposes_both_sides() -> None:
    pred = where("SELECT * FROM r, s WHERE r.a = s.b")

    wrapped = SimpleJoin.wrap(pred)

    assert {wrapped.lhs, wrapped.rhs} == {ColumnReference("a", R), ColumnReference("b", S)}


def test_simple_join_partner_of_is_overloaded_by_table_or_column() -> None:
    """`partner_of` mirrors back the same kind of thing it was given: a table for a table, a column for a
    column.
    """
    pred = where("SELECT * FROM r, s WHERE r.a = s.b")
    wrapped = SimpleJoin.wrap(pred)

    assert wrapped.partner_of(R) == S
    assert wrapped.partner_of(S) == R
    assert wrapped.partner_of(ColumnReference("a", R)) == ColumnReference("b", S)
    assert wrapped.partner_of(ColumnReference("b", S)) == ColumnReference("a", R)


def test_simple_join_partner_of_an_unrelated_table_is_none() -> None:
    pred = where("SELECT * FROM r, s WHERE r.a = s.b")
    wrapped = SimpleJoin.wrap(pred)

    assert wrapped.partner_of(T) is None


def test_simple_join_cannot_wrap_a_non_equi_join() -> None:
    pred = where("SELECT * FROM r, s WHERE r.a < s.b")

    assert SimpleJoin.can_wrap(pred) is False


# -- PredicateTree ----------------------------------------------------------------------------------------


def test_predicate_tree_separates_filters_and_joins() -> None:
    preds = parse("SELECT * FROM r, s, t WHERE r.a = s.b AND s.c = t.d AND r.e = 1").predicates()

    assert len(preds.filters()) == 1
    assert len(preds.joins()) == 2


def test_predicate_tree_is_none_without_a_where_clause() -> None:
    assert parse("SELECT * FROM r").predicates() is None


def test_predicate_tree_join_graph_has_one_node_per_joined_table() -> None:
    preds = parse("SELECT * FROM r, s, t WHERE r.a = s.b AND s.c = t.d").predicates()

    graph = preds.join_graph()

    assert set(graph.nodes()) == {R, S, T}
    # nx.Graph is undirected, so an edge's tuple order is not part of its contract -- normalize before comparing.
    assert {frozenset(edge) for edge in graph.edges()} == {frozenset((R, S)), frozenset((S, T))}


def test_predicate_tree_filters_for_a_specific_table() -> None:
    preds = parse("SELECT * FROM r, s WHERE r.a = s.b AND r.c = 1 AND s.d = 2").predicates()

    filter_for_r = preds.filters_for(R)
    assert filter_for_r is not None
    assert filter_for_r.columns() == {ColumnReference("c", R)}


def test_predicate_tree_filters_for_an_unfiltered_table_is_none() -> None:
    preds = parse("SELECT * FROM r, s WHERE r.a = s.b").predicates()

    assert preds.filters_for(R) is None


def test_predicate_tree_joins_for_a_table_returns_all_its_joins() -> None:
    preds = parse("SELECT * FROM r, s, t WHERE r.a = s.b AND r.c = t.d").predicates()

    joins_for_r = preds.joins_for(R)

    assert len(joins_for_r) == 2


def test_predicate_tree_joins_between_two_tables() -> None:
    preds = parse("SELECT * FROM r, s, t WHERE r.a = s.b AND s.c = t.d").predicates()

    assert preds.joins_between(R, S) is not None
    assert preds.joins_between(R, T) is None


def test_predicate_tree_joins_tables_is_a_plain_boolean_check() -> None:
    preds = parse("SELECT * FROM r, s, t WHERE r.a = s.b").predicates()

    assert preds.joins_tables(R, S) is True
    assert preds.joins_tables(R, T) is False


def test_predicate_tree_all_simple_true_for_plain_equi_joins_and_filters() -> None:
    preds = parse("SELECT * FROM r, s WHERE r.a = s.b AND r.c = 1").predicates()

    assert preds.all_simple() is True


def test_predicate_tree_all_simple_false_for_a_non_equi_join() -> None:
    preds = parse("SELECT * FROM r, s WHERE r.a < s.b").predicates()

    assert preds.all_simple() is False


def test_predicate_tree_simplify_returns_simple_filter_and_join_views() -> None:
    preds = parse("SELECT * FROM r, s WHERE r.a = s.b AND r.c = 1").predicates()

    simplified = preds.simplify()

    assert len(simplified) == 2
    assert any(isinstance(s, SimpleFilter) for s in simplified)
    assert any(isinstance(s, SimpleJoin) for s in simplified)


def test_predicate_tree_merge_with_combines_two_trees_conjunctively() -> None:
    left = parse("SELECT * FROM r WHERE r.a = 1").predicates()
    right = parse("SELECT * FROM s WHERE s.b = 2").predicates()

    merged = left.merge_with(right)

    assert len(merged.filters()) == 2


def test_predicate_tree_is_iterable_over_its_predicates() -> None:
    preds = parse("SELECT * FROM r, s WHERE r.a = s.b AND r.c = 1").predicates()

    assert len(list(preds)) == 2


def test_predicate_tree_bool_is_true_whenever_it_exists() -> None:
    """Even a `PredicateTree` with a single trivial filter is truthy; only the absence of one (`predicates()
    returning None`) should be treated as "no predicates".
    """
    preds = parse("SELECT * FROM r WHERE r.a = 1").predicates()

    assert bool(preds) is True


# -- equivalence classes -----------------------------------------------------------------------------------


def test_determine_join_equivalence_classes_merges_transitively_equal_columns() -> None:
    col_a = ColumnReference("a", R)
    col_b = ColumnReference("b", S)
    col_c = ColumnReference("c", T)
    p_ab = as_predicate(col_a, "=", col_b)
    p_bc = as_predicate(col_b, "=", col_c)

    classes = determine_join_equivalence_classes([p_ab, p_bc])

    assert classes == {frozenset({col_a, col_b, col_c})}


def test_determine_join_equivalence_classes_keeps_unrelated_joins_separate() -> None:
    col_a, col_b = ColumnReference("a", R), ColumnReference("b", S)
    col_c, col_d = ColumnReference("c", T), ColumnReference("d", TableReference("u"))
    p_ab = as_predicate(col_a, "=", col_b)
    p_cd = as_predicate(col_c, "=", col_d)

    classes = determine_join_equivalence_classes([p_ab, p_cd])

    assert classes == {frozenset({col_a, col_b}), frozenset({col_c, col_d})}


def test_determine_join_equivalence_classes_discards_non_equi_joins() -> None:
    col_a, col_b = ColumnReference("a", R), ColumnReference("b", S)
    non_equi = as_predicate(col_a, "<", col_b)

    classes = determine_join_equivalence_classes([non_equi])

    assert classes == set()


def test_generate_predicates_for_equivalence_classes_produces_all_pairs() -> None:
    col_a, col_b, col_c = ColumnReference("a", R), ColumnReference("b", S), ColumnReference("c", T)

    predicates = generate_predicates_for_equivalence_classes({frozenset({col_a, col_b, col_c})})

    pairs = {frozenset(p.columns()) for p in predicates}
    assert pairs == {
        frozenset({col_a, col_b}),
        frozenset({col_a, col_c}),
        frozenset({col_b, col_c}),
    }
