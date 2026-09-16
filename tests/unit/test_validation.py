"""Tests for `postbound.validation` -- the `OptimizationPreCheck` hierarchy.

Every `check_supported_query` here is a pure `SqlQuery -> PreCheckResult` function: no database, no I/O.
`postbound/validation.py` had zero coverage before this file despite being 745 lines that gate which
queries an optimization strategy is even allowed to run on.

All test queries use explicitly qualified columns (``r.a``, not just ``a``), so the parser can bind them
without a schema or a database connection -- see `postbound/parser.py`'s `auto_bind_columns` docs.
"""

from __future__ import annotations

import pytest

from postbound import parser, validation
from postbound._core import JoinOperator, ScanOperator
from postbound._hints import HintType
from postbound.qal import SqlQuery
from tests.doubles import FakeHintService

# -- fixtures -----------------------------------------------------------------------------------------


def parse(sql: str) -> SqlQuery:
    return parser.parse_query(sql)


IMPLICIT_JOIN = parse("SELECT * FROM r, s WHERE r.a = s.b")
EXPLICIT_JOIN = parse("SELECT * FROM r JOIN s ON r.a = s.b")
LEFT_OUTER_JOIN = parse("SELECT * FROM r LEFT JOIN s ON r.a = s.b")
CROSS_PRODUCT = parse("SELECT * FROM r, s")
NON_EQUI_JOIN = parse("SELECT * FROM r, s WHERE r.a < s.b")
SUBQUERY_IN_WHERE = parse("SELECT * FROM r WHERE r.a IN (SELECT s.b FROM s)")
DEPENDENT_SUBQUERY = parse("SELECT * FROM r WHERE EXISTS (SELECT * FROM s WHERE r.a = s.b)")
INDEPENDENT_SUBQUERY = parse("SELECT * FROM r WHERE r.a = (SELECT MIN(s.b) FROM s)")
UNION_QUERY = parse("SELECT r.a FROM r UNION SELECT s.b FROM s")


# -- PreCheckResult -------------------------------------------------------------------------------------


def test_with_all_passed_has_no_failure() -> None:
    result = validation.PreCheckResult.with_all_passed()

    assert result.passed is True
    assert result.failure_reason == ""


def test_with_failure_carries_the_reason() -> None:
    result = validation.PreCheckResult.with_failure("bad query")

    assert result.passed is False
    assert result.failure_reason == "bad query"


def test_merge_of_all_passing_results_passes() -> None:
    merged = validation.PreCheckResult.merge(
        [validation.PreCheckResult.with_all_passed(), validation.PreCheckResult.with_all_passed()]
    )

    assert merged.passed is True


def test_merge_collects_every_failure_reason() -> None:
    merged = validation.PreCheckResult.merge(
        [
            validation.PreCheckResult.with_all_passed(),
            validation.PreCheckResult.with_failure("reason a"),
            validation.PreCheckResult.with_failure(["reason b", "reason c"]),
        ]
    )

    assert merged.passed is False
    assert merged.failure_reason == ["reason a", "reason b", "reason c"]


def test_merge_of_empty_iterable_passes() -> None:
    assert validation.PreCheckResult.merge([]).passed is True


def test_ensure_all_passed_is_a_noop_when_passed() -> None:
    validation.PreCheckResult.with_all_passed().ensure_all_passed()  # must not raise


def test_ensure_all_passed_without_context_raises_state_error() -> None:
    from postbound.util import StateError

    with pytest.raises(StateError):
        validation.PreCheckResult.with_failure("bad").ensure_all_passed()


def test_ensure_all_passed_with_query_context_raises_unsupported_query_error() -> None:
    with pytest.raises(validation.UnsupportedQueryError) as exc_info:
        validation.PreCheckResult.with_failure("bad").ensure_all_passed(IMPLICIT_JOIN)

    assert exc_info.value.query is IMPLICIT_JOIN
    assert exc_info.value.features == "bad"


def test_ensure_all_passed_with_database_context_raises_unsupported_system_error(fake_db) -> None:
    with pytest.raises(validation.UnsupportedSystemError) as exc_info:
        validation.PreCheckResult.with_failure("bad").ensure_all_passed(fake_db)

    assert exc_info.value.db_system is fake_db


# -- OptimizationPreCheck base -------------------------------------------------------------------------


def test_base_check_passes_every_query_by_default() -> None:
    check = validation.OptimizationPreCheck("noop")

    assert check.check_supported_query(IMPLICIT_JOIN).passed is True


def test_base_check_passes_every_database_by_default(fake_db) -> None:
    check = validation.OptimizationPreCheck("noop")

    assert check.check_supported_database_system(fake_db).passed is True


def test_checks_with_the_same_name_are_equal() -> None:
    assert validation.OptimizationPreCheck("x") == validation.OptimizationPreCheck("x")
    assert hash(validation.OptimizationPreCheck("x")) == hash(validation.OptimizationPreCheck("x"))


def test_checks_with_different_names_are_not_equal() -> None:
    assert validation.OptimizationPreCheck("x") != validation.OptimizationPreCheck("y")


def test_contains_checks_self_equality() -> None:
    check = validation.OptimizationPreCheck("x")

    assert check in check
    assert validation.OptimizationPreCheck("y") not in check


def test_describe_reports_the_name() -> None:
    assert validation.OptimizationPreCheck("x").describe() == {"name": "x"}


# -- EmptyPreCheck ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("query", [IMPLICIT_JOIN, EXPLICIT_JOIN, UNION_QUERY])
def test_empty_check_always_passes(query: SqlQuery) -> None:
    assert validation.EmptyPreCheck().check_supported_query(query).passed is True


def test_empty_check_describe() -> None:
    assert validation.EmptyPreCheck().describe() == {"name": "no_check"}


# -- CompoundCheck ---------------------------------------------------------------------------------------


def test_compound_check_passes_when_every_child_passes() -> None:
    compound = validation.CompoundCheck([validation.ImplicitQueryPreCheck(), validation.SubqueryPreCheck()])

    result = compound.check_supported_query(IMPLICIT_JOIN)

    assert result.passed is True


def test_compound_check_fails_when_any_child_fails() -> None:
    compound = validation.CompoundCheck([validation.ImplicitQueryPreCheck(), validation.SubqueryPreCheck()])

    result = compound.check_supported_query(EXPLICIT_JOIN)

    assert result.passed is False
    assert "Query does not have a simple FROM clause" in result.failure_reason


def test_compound_check_collects_every_child_failure() -> None:
    compound = validation.CompoundCheck([validation.ImplicitQueryPreCheck(), validation.SubqueryPreCheck()])

    result = compound.check_supported_query(parse("SELECT * FROM r JOIN s ON r.a = (SELECT MIN(t.c) FROM t)"))

    assert result.passed is False
    assert len(result.failure_reason) == 2


def test_compound_check_flattens_nested_compound_checks() -> None:
    inner = validation.CompoundCheck([validation.ImplicitQueryPreCheck()])
    outer = validation.CompoundCheck([inner, validation.SubqueryPreCheck()])

    assert len(outer.checks) == 2
    assert all(not isinstance(check, validation.CompoundCheck) for check in outer.checks)


def test_compound_check_drops_empty_pre_checks() -> None:
    compound = validation.CompoundCheck([validation.EmptyPreCheck(), validation.ImplicitQueryPreCheck()])

    assert len(compound.checks) == 1


def test_compound_check_contains_looks_into_children() -> None:
    child = validation.ImplicitQueryPreCheck()
    compound = validation.CompoundCheck([child])

    assert child in compound
    assert validation.SubqueryPreCheck() not in compound


# -- merge_checks ----------------------------------------------------------------------------------------


def test_merge_checks_of_nothing_is_empty_check() -> None:
    assert isinstance(validation.merge_checks([]), validation.EmptyPreCheck)


def test_merge_checks_of_a_single_check_returns_it_unwrapped() -> None:
    check = validation.ImplicitQueryPreCheck()

    merged = validation.merge_checks([check])

    assert merged == check
    assert not isinstance(merged, validation.CompoundCheck)


def test_merge_checks_of_several_distinct_checks_builds_a_compound() -> None:
    merged = validation.merge_checks([validation.ImplicitQueryPreCheck(), validation.SubqueryPreCheck()])

    assert isinstance(merged, validation.CompoundCheck)
    assert len(merged.checks) == 2


def test_merge_checks_deduplicates_equal_checks() -> None:
    merged = validation.merge_checks([validation.ImplicitQueryPreCheck(), validation.ImplicitQueryPreCheck()])

    assert merged == validation.ImplicitQueryPreCheck()
    assert not isinstance(merged, validation.CompoundCheck)


def test_merge_checks_ignores_empty_checks() -> None:
    merged = validation.merge_checks([validation.EmptyPreCheck(), validation.EmptyPreCheck()])

    assert isinstance(merged, validation.EmptyPreCheck)


def test_merge_checks_accepts_extra_checks_as_varargs() -> None:
    merged = validation.merge_checks(validation.ImplicitQueryPreCheck(), validation.SubqueryPreCheck())

    assert isinstance(merged, validation.CompoundCheck)
    assert len(merged.checks) == 2


# -- ImplicitQueryPreCheck --------------------------------------------------------------------------------


def test_implicit_query_check_passes_comma_separated_from() -> None:
    assert validation.ImplicitQueryPreCheck().check_supported_query(IMPLICIT_JOIN).passed is True


def test_implicit_query_check_fails_explicit_join_syntax() -> None:
    result = validation.ImplicitQueryPreCheck().check_supported_query(EXPLICIT_JOIN)

    assert result.passed is False
    assert result.failure_reason == "Query does not have a simple FROM clause"


# -- CrossProductPreCheck ----------------------------------------------------------------------------------


def test_cross_product_check_passes_a_connected_join_graph() -> None:
    assert validation.CrossProductPreCheck().check_supported_query(IMPLICIT_JOIN).passed is True


def test_cross_product_check_fails_a_disconnected_join_graph() -> None:
    query = parse("SELECT * FROM r, s WHERE r.a = 1 AND s.b = 2")  # a WHERE clause with no join predicate

    result = validation.CrossProductPreCheck().check_supported_query(query)

    assert result.passed is False
    assert result.failure_reason == "Query contains cross products"


def test_cross_product_check_crashes_when_the_query_has_no_where_clause_at_all() -> None:
    """Documents a real bug, not the intended behaviour.

    `SqlQuery.join_graph()` returns an *empty* `nx.Graph()` (not one node per table) whenever the query has
    no predicates at all -- `SqlQuery.join_graph` -> `PredicateTree.join_graph`, but with `predicates() is
    None` it short-circuits to a bare `nx.Graph()`. `nx.is_connected` treats a graph with zero nodes as
    undefined ("the null graph") and raises `NetworkXPointlessConcept` rather than returning a bool.

    `CrossProductPreCheck` does not guard against this, so it crashes with an unhandled exception instead of
    reporting a `PreCheckResult` -- for `SELECT * FROM r, s` (the single most literal cross product possible)
    and even for the single-table `SELECT * FROM r` with no WHERE clause at all, which is not a cross product
    and should trivially pass.

    This test exists so that fixing it (most likely: treat 0 or 1 nodes as connected before calling
    `nx.is_connected`, the same way `nx.is_connected` itself special-cases a single node) is a deliberate,
    visible change -- see the `lookup_column` fix in commit 823efb5 for the established pattern.
    """
    import networkx as nx

    with pytest.raises(nx.NetworkXPointlessConcept):
        validation.CrossProductPreCheck().check_supported_query(CROSS_PRODUCT)

    single_table_no_predicates = parse("SELECT * FROM r")
    with pytest.raises(nx.NetworkXPointlessConcept):
        validation.CrossProductPreCheck().check_supported_query(single_table_no_predicates)


# -- VirtualTablesPreCheck ---------------------------------------------------------------------------------


def test_virtual_tables_check_passes_a_query_of_only_base_tables() -> None:
    assert validation.VirtualTablesPreCheck().check_supported_query(IMPLICIT_JOIN).passed is True


def test_virtual_tables_check_fails_a_subquery_table_source() -> None:
    query = parse("SELECT * FROM (SELECT * FROM r) AS sub")

    result = validation.VirtualTablesPreCheck().check_supported_query(query)

    assert result.passed is False
    assert result.failure_reason == "Query contains virtual tables"


# -- EquiJoinPreCheck --------------------------------------------------------------------------------------


def test_equi_join_check_passes_a_plain_equi_join() -> None:
    assert validation.EquiJoinPreCheck().check_supported_query(IMPLICIT_JOIN).passed is True


def test_equi_join_check_fails_a_non_equi_join() -> None:
    result = validation.EquiJoinPreCheck().check_supported_query(NON_EQUI_JOIN)

    assert result.passed is False
    assert result.failure_reason == "Query contains non-equi-joins"


def test_equi_join_check_rejects_nesting_by_default() -> None:
    """`r.a + 1 = s.b` is an equi-join, but with a nested expression on one side."""
    query = parse("SELECT * FROM r, s WHERE r.a + 1 = s.b")

    assert validation.EquiJoinPreCheck().check_supported_query(query).passed is False


def test_equi_join_check_allows_nesting_when_enabled() -> None:
    query = parse("SELECT * FROM r, s WHERE r.a + 1 = s.b")

    result = validation.EquiJoinPreCheck(allow_nesting=True).check_supported_query(query)

    assert result.passed is True


def test_equi_join_check_and_connected_joins_are_always_split_and_checked_individually() -> None:
    """`.joins()` un-nests AND-connected predicates before `EquiJoinPreCheck` ever sees them.

    So a query like `r.a = s.b AND r.c = s.d` never actually reaches `_perform_compound_predicate_check` as
    one `CompoundPredicate` -- it arrives as two independent `BinaryPredicate`s, each individually equi. This
    holds regardless of `allow_conjunctions`.
    """
    query = parse("SELECT * FROM r, s WHERE r.a = s.b AND r.c = s.d")

    assert validation.EquiJoinPreCheck(allow_conjunctions=False).check_supported_query(query).passed is True
    assert validation.EquiJoinPreCheck(allow_conjunctions=True).check_supported_query(query).passed is True


def test_equi_join_check_allow_conjunctions_never_actually_changes_the_result() -> None:
    """Documents a real bug, not the intended behaviour.

    `_perform_compound_predicate_check` only accepts a `CompoundPredicate` whose `operation` is
    `CompoundOperator.And`. But `.joins()` (see the test above) always un-nests ANDs before handing predicates
    to a `OptimizationPreCheck`, so the only kind of `CompoundPredicate` `EquiJoinPreCheck` can ever actually
    receive from a real query is an OR-connected one (`.joins()`'s own docstring: "OR predicates are included
    as a whole if they are a join"). An OR is rejected unconditionally, before `allow_conjunctions` is even
    consulted. So the flag can never change the outcome of `check_supported_query` for any query -- it is
    dead configuration.

    This test exists so that fixing it (most likely: also accept `CompoundOperator.Or` under
    `allow_conjunctions`, or drop the flag if AND-splitting means it is genuinely unneeded) is a deliberate,
    visible change -- see the `lookup_column` fix in commit 823efb5 for the established pattern.
    """
    query = parse("SELECT * FROM r, s WHERE r.a = s.b OR r.a = s.c")

    without_flag = validation.EquiJoinPreCheck(allow_conjunctions=False).check_supported_query(query)
    with_flag = validation.EquiJoinPreCheck(allow_conjunctions=True).check_supported_query(query)

    assert without_flag == with_flag
    assert without_flag.passed is False


def test_equi_join_check_instances_with_different_flags_are_not_equal() -> None:
    assert validation.EquiJoinPreCheck() != validation.EquiJoinPreCheck(allow_nesting=True)


# -- InnerJoinPreCheck -------------------------------------------------------------------------------------


def test_inner_join_check_passes_implicit_joins() -> None:
    """Implicit (comma-separated) FROM entries are plain table sources -- there is no join type to violate."""
    assert validation.InnerJoinPreCheck().check_supported_query(IMPLICIT_JOIN).passed is True


def test_inner_join_check_passes_explicit_inner_joins() -> None:
    """Explicit `JOIN` syntax is a `JoinTableSource` with `join_type == JoinOperator.Inner`, which is allowed."""
    assert validation.InnerJoinPreCheck().check_supported_query(EXPLICIT_JOIN).passed is True
    assert validation.InnerJoinPreCheck().check_supported_query(LEFT_OUTER_JOIN).passed is False


# -- SubqueryPreCheck --------------------------------------------------------------------------------------


def test_subquery_check_passes_a_query_without_subqueries() -> None:
    assert validation.SubqueryPreCheck().check_supported_query(IMPLICIT_JOIN).passed is True


def test_subquery_check_fails_a_query_with_a_subquery() -> None:
    result = validation.SubqueryPreCheck().check_supported_query(SUBQUERY_IN_WHERE)

    assert result.passed is False
    assert result.failure_reason == "Query contains subqueries"


# -- DependentSubqueryPreCheck -----------------------------------------------------------------------------


def test_dependent_subquery_check_passes_an_independent_subquery() -> None:
    assert validation.DependentSubqueryPreCheck().check_supported_query(INDEPENDENT_SUBQUERY).passed is True


def test_dependent_subquery_check_fails_a_correlated_subquery() -> None:
    result = validation.DependentSubqueryPreCheck().check_supported_query(DEPENDENT_SUBQUERY)

    assert result.passed is False
    assert result.failure_reason == "Query contains dependent subqueries"


def test_dependent_subquery_check_passes_a_query_without_subqueries() -> None:
    assert validation.DependentSubqueryPreCheck().check_supported_query(IMPLICIT_JOIN).passed is True


# -- SetOperationsPreCheck ---------------------------------------------------------------------------------


def test_set_operations_check_passes_a_plain_select() -> None:
    assert validation.SetOperationsPreCheck().check_supported_query(IMPLICIT_JOIN).passed is True


def test_set_operations_check_fails_a_union() -> None:
    result = validation.SetOperationsPreCheck().check_supported_query(UNION_QUERY)

    assert result.passed is False
    assert result.failure_reason == "SET_OPERATION"


# -- SupportedHintCheck ------------------------------------------------------------------------------------


def test_supported_hint_check_passes_when_every_hint_is_supported() -> None:
    from tests.doubles import FakeDatabase

    db = FakeDatabase(hint_service=FakeHintService(supported={JoinOperator.HashJoin, ScanOperator.SequentialScan}))
    check = validation.SupportedHintCheck([JoinOperator.HashJoin])

    assert check.check_supported_database_system(db).passed is True


def test_supported_hint_check_fails_and_lists_unsupported_hints() -> None:
    from tests.doubles import FakeDatabase

    db = FakeDatabase(hint_service=FakeHintService(supported={ScanOperator.SequentialScan}))
    check = validation.SupportedHintCheck([JoinOperator.HashJoin, HintType.Cardinality])

    result = check.check_supported_database_system(db)

    assert result.passed is False
    assert result.failure_reason == [JoinOperator.HashJoin, HintType.Cardinality]


def test_supported_hint_check_describe_lists_the_features() -> None:
    check = validation.SupportedHintCheck([JoinOperator.HashJoin])

    assert check.describe() == {"name": "database_operator_support", "features": [JoinOperator.HashJoin]}


# -- CustomCheck -------------------------------------------------------------------------------------------


def test_custom_check_with_no_callbacks_passes_everything(fake_db) -> None:
    check = validation.CustomCheck()

    assert check.check_supported_query(IMPLICIT_JOIN).passed is True
    assert check.check_supported_database_system(fake_db).passed is True


def test_custom_check_delegates_to_the_query_callback() -> None:
    check = validation.CustomCheck(query_check=lambda q: validation.PreCheckResult.with_failure("nope"))

    result = check.check_supported_query(IMPLICIT_JOIN)

    assert result.passed is False
    assert result.failure_reason == "nope"


def test_custom_check_delegates_to_the_db_callback(fake_db) -> None:
    seen = []
    check = validation.CustomCheck(
        db_check=lambda db: (seen.append(db), validation.PreCheckResult.with_all_passed())[1]
    )

    check.check_supported_database_system(fake_db)

    assert seen == [fake_db]


# -- SPJCheck ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM r, s WHERE r.a = s.b",
        "SELECT COUNT(*) FROM r, s WHERE r.a = s.b",
        "SELECT * FROM r",
    ],
)
def test_spj_check_passes_simple_spj_queries(sql: str) -> None:
    result = validation.SPJCheck().check_supported_query(parse(sql))

    assert result.passed is True


def test_spj_check_fails_a_non_select_query() -> None:
    """A UNION is a `SetQuery`, not a `SelectStatement` -- `is_select_query()` is false for it."""
    result = validation.SPJCheck().check_supported_query(UNION_QUERY)

    assert result.passed is False
    assert result.failure_reason == "Query is not a SELECT query"


def test_spj_check_rejects_an_explain_query_for_its_extra_clause_before_reaching_the_select_check() -> None:
    """An EXPLAIN query is still a `SelectStatement` (`is_explain()` is an orthogonal flag), so it passes the
    "is this a SELECT" check -- but is then caught by the unsupported-clauses check instead, since `Explain`
    is not in `[Select, From, Where]`. Different message, same overall rejection.
    """
    explain_query = parse("EXPLAIN SELECT * FROM r")

    result = validation.SPJCheck().check_supported_query(explain_query)

    assert result.passed is False
    assert "Explain" in result.failure_reason


def test_spj_check_fails_a_complex_select_clause() -> None:
    result = validation.SPJCheck().check_supported_query(parse("SELECT r.a FROM r"))

    assert result.passed is False
    assert result.failure_reason == "SELECT clause has complex contents"


def test_spj_check_fails_unsupported_clauses() -> None:
    result = validation.SPJCheck().check_supported_query(parse("SELECT * FROM r, s WHERE r.a = s.b ORDER BY r.a"))

    assert result.passed is False
    assert "unsupported clauses" in result.failure_reason


def test_spj_check_fails_a_subquery_table_source() -> None:
    query = parse("SELECT * FROM (SELECT * FROM r) AS sub")

    result = validation.SPJCheck().check_supported_query(query)

    assert result.passed is False
    assert result.failure_reason == "FROM clause has complex contents"


def test_spj_check_fails_a_cross_product() -> None:
    query = parse("SELECT * FROM r, s WHERE r.a = 1 AND s.b = 2")  # a WHERE clause with no join predicate

    result = validation.SPJCheck().check_supported_query(query)

    assert result.passed is False
    assert result.failure_reason == "Query contains cross products"


def test_spj_check_passes_an_unfiltered_cross_product() -> None:
    """Documents a real bug, not the intended behaviour.

    The check's own docstring requires "the query only performs inner equi joins and no cross products", but
    the implementation returns `with_all_passed()` as soon as `query.predicates()` is *None* -- which is
    exactly the case for a query with no WHERE clause at all. `SELECT * FROM r, s` is the most literal
    possible cross product, yet it passes `SPJCheck` unchanged.

    This test exists so that fixing it (most likely: check `nx.is_connected` over `query.tables()` rather than
    short-circuiting on an absent predicate tree -- mind the null-graph crash documented in
    `test_cross_product_check_crashes_when_the_query_has_no_where_clause_at_all` if reusing that approach) is
    a deliberate, visible change -- see the `lookup_column` fix in commit 823efb5 for the established pattern.
    """
    result = validation.SPJCheck().check_supported_query(CROSS_PRODUCT)

    assert result.passed is True  # should be False


def test_spj_check_fails_a_non_equi_join() -> None:
    """`all_simple()` rejects a non-equi join as "complex" before the dedicated non-equi-join check ever
    runs -- `SimpleJoin` can only represent equi-joins by construction, so the later, more specific
    "Query contains non-equi joins" branch is unreachable in practice. Different message, same rejection.
    """
    result = validation.SPJCheck().check_supported_query(NON_EQUI_JOIN)

    assert result.passed is False
    assert result.failure_reason == "Query has complex predicates"


def test_spj_check_crashes_on_a_predicate_with_a_function_call() -> None:
    """Documents a real bug, not the intended behaviour -- and not local to `validation.py` or `SPJCheck`.

    `SimpleFilter.can_wrap` is documented to return a `bool` reporting whether a predicate can be represented
    simply. Its implementation is ``_attempt_filter_unwrap(predicate) is not None``, but
    `_attempt_filter_unwrap` delegates to `_unwrap_expression`, which raises a bare `ValueError` (rather than
    returning `None`) for any expression it does not recognise -- a function call among them. Nothing between
    `_unwrap_expression` and `can_wrap` catches it, so `PredicateTree.all_simple()` -- which `SPJCheck`'s own
    docstring says exists to reject exactly this case ("4. all predicates are simple (no function calls,
    etc.)") -- crashes instead of returning `False` for any query with a function call in a filter predicate.

    This test exists so that fixing it (most likely: `_attempt_filter_unwrap` catching the `ValueError` from
    `_unwrap_expression`, or `_unwrap_expression` itself returning `None` on the unrecognised branch instead
    of raising) is a deliberate, visible change -- see the `lookup_column` fix in commit 823efb5 for the
    established pattern.
    """
    query = parse("SELECT * FROM r, s WHERE r.a = s.b AND UPPER(r.c) = 'X'")

    with pytest.raises(ValueError, match="Cannot unwrap expression"):
        validation.SPJCheck().check_supported_query(query)
