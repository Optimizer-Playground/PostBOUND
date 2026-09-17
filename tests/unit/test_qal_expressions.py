"""Tests for the `SqlExpression` hierarchy in `postbound.qal`.

Expressions are obtained by parsing real SQL through the actual parser wherever that reaches the class under
test -- unlike EXPLAIN JSON, there is no "external, undocumented shape" risk here: qal is PostBOUND's own
model, and the parser is the normal, correct way to build one. `QuantifierExpression` is the one exception:
the parser currently fails on quantified comparisons (`> ALL (...)`) for unrelated reasons in `parser.py`,
which is out of scope here, so it is built directly through its own documented factory methods instead.

Column binding is disabled throughout (`bind_columns=False`): these tests are about expression *shape*, not
about schema-dependent binding (that belongs with the parser's own tests), and disabling it keeps the module
independent of any registered database.
"""

from __future__ import annotations

import pytest

from postbound import parser
from postbound._core import ColumnReference, TableReference
from postbound.qal import (
    ArrayAccessExpression,
    ArrayExpression,
    CastExpression,
    ColumnExpression,
    ExpressionCollector,
    FunctionExpression,
    MathExpression,
    MathOperator,
    QuantifierExpression,
    QuantifierOperator,
    StarExpression,
    StaticValueExpression,
    SubqueryExpression,
    WindowExpression,
    as_expression,
    as_func_expr,
    as_math_expr,
)

R = TableReference("r")


def parse(sql: str):
    return parser.parse_query(sql, bind_columns=False)


def target_expr(sql: str):
    """Parses `sql` and returns the single SELECT-list expression."""
    query = parse(sql)
    return query.select_clause.targets[0].expression


# -- StaticValueExpression ------------------------------------------------------------------------------


def test_static_value_expression_holds_a_plain_value() -> None:
    expr = StaticValueExpression(42)

    assert expr.value == 42
    assert expr.columns() == set()
    assert list(expr.itercolumns()) == []
    assert list(expr.iterchildren()) == []


def test_static_value_expressions_with_equal_values_are_equal() -> None:
    assert StaticValueExpression(42) == StaticValueExpression(42)
    assert hash(StaticValueExpression(42)) == hash(StaticValueExpression(42))
    assert StaticValueExpression(42) != StaticValueExpression(43)


def test_static_value_expression_is_immutable() -> None:
    expr = StaticValueExpression(42)

    with pytest.raises(AttributeError):
        expr.value = 43  # type: ignore - the point of this test is that this raises at runtime


# -- StarExpression --------------------------------------------------------------------------------------


def test_star_expression_has_no_columns() -> None:
    expr = StarExpression()

    assert expr.columns() == set()
    assert expr.tables() == set()


def test_select_star_is_not_counted_as_a_column() -> None:
    query = parse("SELECT * FROM r")

    assert query.columns() == set()


# -- ColumnExpression --------------------------------------------------------------------------------------


def test_column_expression_wraps_a_column_reference() -> None:
    expr = target_expr("SELECT r.a FROM r")

    assert isinstance(expr, ColumnExpression)
    assert expr.column == ColumnReference("a", R)
    assert expr.columns() == {ColumnReference("a", R)}
    assert list(expr.itercolumns()) == [ColumnReference("a", R)]


# -- MathExpression ----------------------------------------------------------------------------------------


def test_math_expression_captures_operator_and_operands() -> None:
    expr = target_expr("SELECT r.a + r.b * 2 FROM r")

    assert isinstance(expr, MathExpression)
    assert expr.operator == MathOperator.Add
    assert isinstance(expr.lhs, ColumnExpression)
    assert isinstance(expr.rhs, MathExpression)
    assert expr.rhs.operator == MathOperator.Multiply


def test_math_expression_columns_union_both_operands() -> None:
    expr = target_expr("SELECT r.a + r.b FROM r")

    assert expr.columns() == {ColumnReference("a", R), ColumnReference("b", R)}


def test_as_math_expr_builds_an_equivalent_expression() -> None:
    """Docstring example from `as_math_expr`."""
    built = as_math_expr(ColumnReference("a"), "+", ColumnReference("b"))

    assert built.operator == MathOperator.Add
    assert built == MathExpression(
        MathOperator.Add, ColumnExpression(ColumnReference("a")), ColumnExpression(ColumnReference("b"))
    )


def test_as_math_expr_accepts_plain_values_on_either_side() -> None:
    built = as_math_expr(ColumnReference("a"), MathOperator.Subtract, 42)

    assert isinstance(built.lhs, ColumnExpression)
    assert isinstance(built.rhs, StaticValueExpression)
    assert built.rhs.value == 42


# -- FunctionExpression ------------------------------------------------------------------------------------


def test_function_expression_captures_name_and_arguments() -> None:
    expr = target_expr("SELECT count(distinct r.a) FROM r")

    assert isinstance(expr, FunctionExpression)
    assert expr.function == "COUNT"
    assert expr.distinct is True
    assert expr.arguments == (ColumnExpression(ColumnReference("a", R)),)
    assert expr.is_aggregate() is True


def test_function_expression_non_aggregate() -> None:
    expr = target_expr("SELECT upper(r.a) FROM r")

    assert expr.function == "UPPER"
    assert expr.is_aggregate() is False


def test_as_func_expr_builds_a_function_expression() -> None:
    """Docstring example from `as_func_expr`."""
    built = as_func_expr("sum", ColumnReference("a"))

    assert built == FunctionExpression("sum", [ColumnExpression(ColumnReference("a"))])


def test_as_func_expr_treats_a_mapping_argument_as_keyword_arguments() -> None:
    """Docstring example from `as_func_expr` -- lets keywords that clash with Python syntax (e.g. `from`)
    be passed through anyway. Keyword names are normalized to uppercase, matching the SQL keywords they
    represent (`SUBSTRING(x FROM 1 FOR 3)`) and the same normalization `FunctionExpression` applies to the
    function name itself.
    """
    built = as_func_expr("substring", ColumnReference("a"), {"from": 1, "for": 3})

    assert built.arguments == (ColumnExpression(ColumnReference("a")),)
    assert set(built.keyword_args) == {"FROM", "FOR"}
    assert built.keyword_args["FROM"] == StaticValueExpression(1)


# -- ArrayExpression / ArrayAccessExpression --------------------------------------------------------------


def test_array_expression_holds_its_elements() -> None:
    expr = target_expr("SELECT ARRAY[1, 2, 3] FROM r")

    assert isinstance(expr, ArrayExpression)
    assert expr.elements == (StaticValueExpression(1), StaticValueExpression(2), StaticValueExpression(3))


def test_array_access_expression_indexes_into_an_array_valued_column() -> None:
    expr = target_expr("SELECT r.d[1] FROM r")

    assert isinstance(expr, ArrayAccessExpression)
    assert expr.array == ColumnExpression(ColumnReference("d", R))
    assert expr.index == StaticValueExpression(1)


# -- CastExpression -----------------------------------------------------------------------------------------


def test_cast_expression_captures_target_type() -> None:
    expr = target_expr("SELECT r.a::text FROM r")

    assert isinstance(expr, CastExpression)
    assert expr.casted_expression == ColumnExpression(ColumnReference("a", R))
    assert expr.target_type == "text"


def test_cast_expression_columns_delegate_to_the_casted_expression() -> None:
    expr = target_expr("SELECT r.a::integer FROM r")

    assert expr.columns() == {ColumnReference("a", R)}


# -- SubqueryExpression -------------------------------------------------------------------------------------


def test_subquery_expression_wraps_the_nested_query() -> None:
    expr = target_expr("SELECT (SELECT min(s.x) AS m FROM s) FROM r")

    assert isinstance(expr, SubqueryExpression)
    assert TableReference("s") in expr.query.tables()


def test_subquery_expression_columns_come_from_the_nested_query() -> None:
    expr = target_expr("SELECT (SELECT s.x FROM s WHERE s.y = r.z) FROM r")

    assert ColumnReference("z", R) in expr.columns()


# -- WindowExpression ---------------------------------------------------------------------------------------


def test_window_expression_captures_partitioning_and_ordering() -> None:
    expr = target_expr("SELECT row_number() OVER (PARTITION BY r.a ORDER BY r.b) FROM r")

    assert isinstance(expr, WindowExpression)
    assert isinstance(expr.window_function, FunctionExpression)
    assert expr.window_function.function == "ROW_NUMBER"
    assert len(expr.partitioning) == 1
    assert len(expr.ordering) == 1


def test_window_expression_without_partition_or_order() -> None:
    """`partitioning` is a plain sequence and normalizes to an empty tuple; `ordering` is a single `OrderBy`
    clause object (or *None*), not a sequence -- the two default differently by design.
    """
    expr = target_expr("SELECT row_number() OVER () FROM r")

    assert expr.partitioning == ()
    assert expr.ordering is None


# -- CaseExpression -----------------------------------------------------------------------------------------


def test_case_expression_captures_branches_and_else() -> None:
    expr = target_expr("SELECT CASE WHEN r.a > 1 THEN 'x' WHEN r.a > 2 THEN 'y' ELSE 'z' END FROM r")

    assert len(expr.cases) == 2
    _first_condition, first_result = expr.cases[0]
    assert first_result == StaticValueExpression("x")
    assert expr.else_expression == StaticValueExpression("z")


def test_case_expression_without_else_defaults_to_none() -> None:
    expr = target_expr("SELECT CASE WHEN r.a > 1 THEN 'x' END FROM r")

    assert expr.else_expression is None


# -- QuantifierExpression ----------------------------------------------------------------------------------
#
# Built directly rather than through the parser -- see the module docstring.


def test_quantifier_expression_any_factory() -> None:
    subquery = parse("SELECT s.b FROM s")
    expr = QuantifierExpression.any(subquery)

    assert expr.quantifier == QuantifierOperator.Any
    assert isinstance(expr.expression, SubqueryExpression)


def test_quantifier_expression_all_factory() -> None:
    subquery = parse("SELECT s.b FROM s")
    expr = QuantifierExpression.all(subquery)

    assert expr.quantifier == QuantifierOperator.All


def test_quantifier_expression_delegates_tables_and_columns_to_its_expression() -> None:
    subquery = parse("SELECT s.b FROM s WHERE s.c = 1")
    expr = QuantifierExpression.any(subquery)

    assert expr.tables() == subquery.tables()


# -- as_expression -----------------------------------------------------------------------------------------


def test_as_expression_passes_through_an_existing_expression() -> None:
    expr = StaticValueExpression(1)

    assert as_expression(expr) is expr


def test_as_expression_wraps_a_column_reference() -> None:
    result = as_expression(ColumnReference("a"))

    assert result == ColumnExpression(ColumnReference("a"))


def test_as_expression_wraps_a_select_statement_as_a_subquery() -> None:
    query = parse("SELECT s.b FROM s")

    result = as_expression(query)

    assert isinstance(result, SubqueryExpression)
    assert result.query is query


def test_as_expression_star_string_becomes_star_expression_by_default() -> None:
    assert as_expression("*") == StarExpression()


def test_as_expression_star_string_can_be_treated_as_a_literal() -> None:
    result = as_expression("*", allow_star=False)

    assert isinstance(result, StaticValueExpression)
    assert result.value == "*"


def test_as_expression_wraps_a_plain_value_as_a_static_value() -> None:
    assert as_expression("a") == StaticValueExpression("a")
    assert as_expression(42) == StaticValueExpression(42)


# -- ExpressionCollector ------------------------------------------------------------------------------------


def test_expression_collector_finds_all_matching_columns() -> None:
    expr = target_expr("SELECT r.a + r.b * r.c FROM r")

    collector = ExpressionCollector(lambda e: isinstance(e, ColumnExpression))
    found = expr.accept_visitor(collector)

    assert found == {
        ColumnExpression(ColumnReference("a", R)),
        ColumnExpression(ColumnReference("b", R)),
        ColumnExpression(ColumnReference("c", R)),
    }


def test_expression_collector_stops_at_first_match_by_default() -> None:
    """A `MathExpression` matching the predicate should not also report its nested column children."""
    expr = target_expr("SELECT r.a + r.b FROM r")

    collector = ExpressionCollector(lambda e: isinstance(e, MathExpression))
    found = expr.accept_visitor(collector)

    assert found == {expr}


def test_expression_collector_continues_after_match_when_requested() -> None:
    expr = target_expr("SELECT r.a + r.b FROM r")

    collector = ExpressionCollector(lambda e: isinstance(e, MathExpression), continue_after_match=True)
    found = expr.accept_visitor(collector)

    assert found == {expr}  # the collector's own predicate never matches the ColumnExpression children


def test_expression_collector_finds_no_matches() -> None:
    expr = target_expr("SELECT r.a FROM r")

    collector = ExpressionCollector(lambda e: isinstance(e, FunctionExpression))
    found = expr.accept_visitor(collector)

    assert found == set()
