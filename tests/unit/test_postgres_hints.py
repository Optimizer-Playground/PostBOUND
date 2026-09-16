"""Tests for Postgres hint generation, exercised as pure functions of qal/hint objects.

Both the pg_lab and pg_hint_plan dialects turn out to need no live connection at all: the pg_lab family
(`_generate_pglab_hints`, `_generate_pglab_plan`, `_extract_plan_join_order`, `_walk_join_order`,
`_expand_pglab_hints`) is already a pure function of a `JoinTree`/`PhysicalOperatorAssignment`/`QueryPlan`.
`_generate_pghintplan_hints` needs exactly one piece of live state -- `pg_instance.config["geqo_threshold"]`
-- so a tiny stub exposing that one lookup is enough to make it fully offline-testable too.
"""

from __future__ import annotations

import itertools
import math

import pytest

from postbound import parser
from postbound._core import (
    Cardinality,
    IntermediateOperator,
    JoinOperator,
    ScanOperator,
    TableReference,
)
from postbound._hints import (
    JoinOperatorAssignment,
    JoinTree,
    PhysicalOperatorAssignment,
    PlanParameterization,
    ScanOperatorAssignment,
)
from postbound._qep import QueryPlan
from postbound.db import HintWarning
from postbound.postgres._pg import (
    _expand_pglab_hints,
    _extract_plan_join_order,
    _generate_pghintplan_hints,
    _generate_pglab_hints,
    _generate_pglab_plan,
    _walk_join_order,
)

R = TableReference("r")
S = TableReference("s")
T = TableReference("t")


class StubPgInstance:
    """The one piece of live state `_generate_pghintplan_hints` needs: `pg_instance.config[key]`.

    Real code reads this via `PostgresConfigInterface.__getitem__`, which issues `SHOW <key>;`. This stub
    skips the connection entirely -- the function only ever looks up ``"geqo_threshold"``.
    """

    def __init__(self, geqo_threshold: int = 12) -> None:
        self.config = {"geqo_threshold": str(geqo_threshold)}


def assert_any_ordering(text: str, template: str, *tables: TableReference) -> None:
    """Asserts `template` (with a single `{}` placeholder for the space-joined identifiers) appears in `text`
    for *some* ordering of `tables`.

    Several hint fragments are built from a `frozenset[TableReference]` (join partners, intermediate-operator
    targets, cardinality keys). Iteration order over a frozenset is not part of its contract, so a test must
    not assume one -- only that the tables appear together in *some* order.
    """
    variants = [template.format(" ".join(t.identifier() for t in perm)) for perm in itertools.permutations(tables)]
    assert any(variant in text for variant in variants), f"none of {variants!r} found in:\n{text}"


def query_with_tables(*tables: TableReference) -> object:
    """A minimal cross-join query over the given tables, with column binding disabled.

    Only `query.tables()` is used by the code under test, so the query never references any columns and does
    not need a schema.
    """
    from_clause = ", ".join(table.full_name for table in tables)
    return parser.parse_query(f"SELECT * FROM {from_clause}", bind_columns=False)


# -- _walk_join_order / _extract_plan_join_order ----------------------------------------------------------


def test_walk_join_order_of_a_single_scan() -> None:
    tree = JoinTree.create_scan(R)

    assert _walk_join_order(tree) == "r"


def test_walk_join_order_nests_parenthesized_joins() -> None:
    tree = JoinTree.create_scan(R).join_with(S).join_with(T)

    assert _walk_join_order(tree) == "((r s) t)"


def test_extract_plan_join_order_of_a_single_scan() -> None:
    plan = QueryPlan("Seq Scan", operator=ScanOperator.SequentialScan, base_table=R)

    assert _extract_plan_join_order(plan) == "r"


def test_extract_plan_join_order_of_a_join() -> None:
    plan = QueryPlan(
        "Nested Loop",
        operator=JoinOperator.NestedLoopJoin,
        children=[
            QueryPlan("Seq Scan", operator=ScanOperator.SequentialScan, base_table=R),
            QueryPlan("Seq Scan", operator=ScanOperator.SequentialScan, base_table=S),
        ],
    )

    assert _extract_plan_join_order(plan) == "(r s)"


def test_extract_plan_join_order_skips_transparent_intermediate_nodes() -> None:
    """A node with a single `input_node` (Materialize, Sort, ...) is not part of the join order."""
    scan = QueryPlan("Seq Scan", operator=ScanOperator.SequentialScan, base_table=R)
    materialize = QueryPlan("Materialize", operator=IntermediateOperator.Materialize, children=[scan])

    assert _extract_plan_join_order(materialize) == "r"


# -- _generate_pglab_hints ---------------------------------------------------------------------------------


def test_pglab_hints_always_start_in_anchored_plan_mode() -> None:
    hint = _generate_pglab_hints(None, None, None)

    assert "Config(plan_mode=anchored)" in hint.query_hints
    assert hint.query_hints.startswith("/*=pg_lab=")
    assert hint.query_hints.rstrip().endswith("*/")


def test_pglab_hints_omit_join_order_for_a_single_table() -> None:
    tree = JoinTree.create_scan(R)

    hint = _generate_pglab_hints(tree, None, None)

    assert "JoinOrder" not in hint.query_hints


def test_pglab_hints_include_join_order_for_multiple_tables() -> None:
    tree = JoinTree.create_scan(R).join_with(S)

    hint = _generate_pglab_hints(tree, None, None)

    assert "JoinOrder((r s))" in hint.query_hints


def test_pglab_hints_render_scan_operator() -> None:
    ops = PhysicalOperatorAssignment().set_scan_operator(ScanOperator.IndexScan, R)

    hint = _generate_pglab_hints(None, ops, None)

    assert "IdxScan(r)" in hint.query_hints


def test_pglab_hints_render_scan_operator_with_worker_count() -> None:
    ops = PhysicalOperatorAssignment().set_scan_operator(
        ScanOperatorAssignment(ScanOperator.SequentialScan, R, parallel_workers=4)
    )

    hint = _generate_pglab_hints(None, ops, None)

    assert "SeqScan(r (workers=4))" in hint.query_hints


def test_pglab_hints_render_join_operator() -> None:
    ops = PhysicalOperatorAssignment().set_join_operator(JoinOperator.HashJoin, [R, S])

    hint = _generate_pglab_hints(None, ops, None)

    assert_any_ordering(hint.query_hints, "HashJoin({})", R, S)


def test_pglab_hints_render_intermediate_operator() -> None:
    ops = PhysicalOperatorAssignment().set_intermediate_operator(IntermediateOperator.Memoize, [R, S])

    hint = _generate_pglab_hints(None, ops, None)

    assert_any_ordering(hint.query_hints, "Memo({})", R, S)


def test_pglab_hints_render_global_settings() -> None:
    ops = PhysicalOperatorAssignment().set(ScanOperator.SequentialScan, False)

    hint = _generate_pglab_hints(None, ops, None)

    assert "Set(enable_seqscan = 'off')" in hint.query_hints


def test_pglab_hints_render_cardinality() -> None:
    params = PlanParameterization().add_cardinality([R, S], 42)

    hint = _generate_pglab_hints(None, None, params)

    assert_any_ordering(hint.query_hints, "Card({} #42)", R, S)


def test_pglab_hints_skip_nan_cardinality() -> None:
    params = PlanParameterization().add_cardinality([R, S], Cardinality.unknown())

    hint = _generate_pglab_hints(None, None, params)

    assert "Card(" not in hint.query_hints


def test_pglab_hints_warn_and_skip_infinite_cardinality() -> None:
    params = PlanParameterization().add_cardinality([R, S], Cardinality.infinite())

    with pytest.warns(HintWarning, match="infinite cardinality"):
        hint = _generate_pglab_hints(None, None, params)

    assert "Card(" not in hint.query_hints


def test_pglab_hints_render_system_settings() -> None:
    params = PlanParameterization().set_system_settings("work_mem", "256MB")

    hint = _generate_pglab_hints(None, None, params)

    assert "Set(work_mem = '256MB')" in hint.query_hints


def test_pglab_hints_render_execution_mode() -> None:
    params = PlanParameterization()
    params.execution_mode = "sequential"

    hint = _generate_pglab_hints(None, None, params)

    assert "Config(exec_mode=sequential)" in hint.query_hints


def test_pglab_hints_warns_when_worker_params_have_no_matching_operator_assignment() -> None:
    """Worker counts can only be attached to nodes with a known operator; without any, they are ignored."""
    params = PlanParameterization().set_workers([R], 4)

    with pytest.warns(HintWarning, match="known operators"):
        hint = _generate_pglab_hints(None, None, params)

    assert "workers=4" not in hint.query_hints


def test_pglab_hints_worker_params_for_a_single_table_are_never_integrated() -> None:
    """Documents a real bug in `PhysicalOperatorAssignment.__contains__`, not the intended behaviour.

    For a singleton candidate, `__contains__` checks ``item in self.scan_operators`` where `item` is still
    the original (frozen)set -- e.g. ``frozenset({R}) in {R: ...}`` -- rather than unwrapping it to the single
    `TableReference` the dict is actually keyed by. So `frozenset({R}) in ops` is always *False* even when `R`
    has a scan operator assigned (and raises `TypeError` outright for a plain, unhashable list).

    `_generate_pglab_hints` relies on this membership check to decide whether a single-table worker-count hint
    (`plan_params.set_workers([R], n)`) can be integrated into an existing scan assignment. Because the check
    is always wrong for the singleton case, every such hint is treated as "dangling" and dropped with a
    warning -- even though the assignment obviously has an operator for that table.

    Multi-table joins are unaffected: for `len(items) > 1` the same method correctly checks
    ``items in self.join_operators``.

    This test exists so that fixing `__contains__` is a deliberate, visible change rather than a silent one --
    see the `lookup_column` fix in commit 823efb5 for the established pattern.
    """
    ops = PhysicalOperatorAssignment().set_scan_operator(ScanOperator.SequentialScan, R)
    params = PlanParameterization().set_workers([R], 4)

    # What should happen (and does not): the worker count gets attached to the existing scan assignment.
    with pytest.warns(HintWarning, match="known operators"):
        hint = _generate_pglab_hints(None, ops, params)
    assert "workers=4" not in hint.query_hints

    # What actually happens: __contains__ itself already disagrees with `in` on a bare TableReference.
    assert R in ops
    assert frozenset([R]) not in ops  # should be True, since R has a scan operator assigned


def test_pglab_hints_no_preparatory_statements() -> None:
    """pg_lab hints are entirely comment-based; unlike pg_hint_plan, no SET LOCAL is ever emitted."""
    params = PlanParameterization().set_system_settings("work_mem", "256MB")

    hint = _generate_pglab_hints(None, None, params)

    assert hint.preparatory_statements == ""


# -- _expand_pglab_hints -----------------------------------------------------------------------------------


def test_expand_pglab_hints_wraps_raw_lines_in_the_comment_block() -> None:
    hint = _expand_pglab_hints(["JoinOrder(r s)", "SeqScan(r)"])

    assert hint.query_hints.startswith("/*=pg_lab=")
    assert "  JoinOrder(r s)" in hint.query_hints
    assert "  SeqScan(r)" in hint.query_hints
    assert hint.preparatory_statements == ""


# -- _generate_pglab_plan (full-plan dialect) --------------------------------------------------------------


def test_pglab_plan_starts_with_full_plan_mode_and_join_order() -> None:
    plan = QueryPlan("Seq Scan", operator=ScanOperator.SequentialScan, base_table=R)

    lines = _generate_pglab_plan(plan)

    assert lines[0] == "Config(plan_mode=full)"
    assert lines[1] == "JoinOrder(r)"


def test_pglab_plan_hints_the_single_scan_operator() -> None:
    plan = QueryPlan(
        "Seq Scan", operator=ScanOperator.SequentialScan, base_table=R, estimated_cardinality=Cardinality(1000)
    )

    lines = _generate_pglab_plan(plan)

    assert any(line.strip() == "SeqScan(r)" for line in lines)


def test_pglab_plan_injects_cardinality_for_hinted_nodes() -> None:
    """The plan dialect always injects the estimated cardinality alongside the operator hint (see the long
    rationale comment in `_generate_pglab_plan`): index-scan choice and memoize eligibility are sensitive to
    it, so the hinted replay must reproduce the same estimate that produced the original plan.
    """
    plan = QueryPlan(
        "Seq Scan", operator=ScanOperator.SequentialScan, base_table=R, estimated_cardinality=Cardinality(1000)
    )

    lines = _generate_pglab_plan(plan)

    assert any(line.strip() == "Card(r #1000)" for line in lines)


def test_pglab_plan_does_not_inject_cardinality_for_intermediate_operators() -> None:
    scan = QueryPlan("Seq Scan", operator=ScanOperator.SequentialScan, base_table=R)
    sort = QueryPlan(
        "Sort", operator=IntermediateOperator.Sort, children=[scan], estimated_cardinality=Cardinality(1000)
    )

    lines = _generate_pglab_plan(sort)

    assert not any("Card(" in line and "Sort" not in line and line.strip().startswith("Card") for line in lines)


def test_pglab_plan_recurses_into_join_children() -> None:
    plan = QueryPlan(
        "Nested Loop",
        operator=JoinOperator.NestedLoopJoin,
        estimated_cardinality=Cardinality(10),
        children=[
            QueryPlan(
                "Seq Scan", operator=ScanOperator.SequentialScan, base_table=R, estimated_cardinality=Cardinality(100)
            ),
            QueryPlan(
                "Index Scan", operator=ScanOperator.IndexScan, base_table=S, estimated_cardinality=Cardinality(1)
            ),
        ],
    )

    lines = _generate_pglab_plan(plan)
    rendered = " ".join(line.strip() for line in lines)

    assert_any_ordering(rendered, "NestLoop({})", R, S)
    assert "SeqScan(r)" in rendered
    assert "IdxScan(s)" in rendered


def test_pglab_plan_hints_parallel_workers_on_the_gather_result() -> None:
    """A Gather sitting above a plain aggregate (upperrel) becomes a `Result(workers=...)` hint."""
    scan = QueryPlan("Seq Scan", operator=ScanOperator.SequentialScan, base_table=R)
    aggregate = QueryPlan("Aggregate", children=[scan])
    gather = QueryPlan("Gather", parallel_workers=4, children=[aggregate])

    lines = _generate_pglab_plan(gather)

    assert any("Result(workers=4)" in line for line in lines)


# -- _generate_pghintplan_hints ------------------------------------------------------------------------------


def test_pghintplan_hints_no_geqo_disable_below_threshold() -> None:
    query = query_with_tables(R, S)

    hint = _generate_pghintplan_hints(query, None, None, None, pg_instance=StubPgInstance(geqo_threshold=12))

    assert "Set(geqo off)" not in hint.query_hints


def test_pghintplan_hints_disables_geqo_above_threshold() -> None:
    """geqo only supports the DP optimizer, so once the table count exceeds the threshold it must be disabled."""
    query = query_with_tables(R, S, T)

    with pytest.warns(HintWarning, match="disabling GEQO"):
        hint = _generate_pghintplan_hints(query, None, None, None, pg_instance=StubPgInstance(geqo_threshold=2))

    assert "Set(geqo off)" in hint.query_hints


def test_pghintplan_hints_render_leading_join_order() -> None:
    query = query_with_tables(R, S)
    tree = JoinTree.create_scan(R).join_with(S)

    hint = _generate_pghintplan_hints(query, tree, None, None, pg_instance=StubPgInstance())

    assert "Leading((r s))" in hint.query_hints


def test_pghintplan_hints_omit_leading_for_single_table() -> None:
    query = query_with_tables(R)
    tree = JoinTree.create_scan(R)

    hint = _generate_pghintplan_hints(query, tree, None, None, pg_instance=StubPgInstance())

    assert "Leading" not in hint.query_hints


def test_pghintplan_hints_render_scan_operator() -> None:
    query = query_with_tables(R)
    ops = PhysicalOperatorAssignment().set_scan_operator(ScanOperator.SequentialScan, R)

    hint = _generate_pghintplan_hints(query, None, ops, None, pg_instance=StubPgInstance())

    assert "SeqScan(r)" in hint.query_hints


def test_pghintplan_hints_render_scan_parallel_workers() -> None:
    query = query_with_tables(R)
    ops = PhysicalOperatorAssignment().set_scan_operator(
        ScanOperatorAssignment(ScanOperator.SequentialScan, R, parallel_workers=3)
    )

    hint = _generate_pghintplan_hints(query, None, ops, None, pg_instance=StubPgInstance())

    assert "Parallel(r 3 hard)" in hint.query_hints


def test_pghintplan_hints_render_join_operator() -> None:
    query = query_with_tables(R, S)
    ops = PhysicalOperatorAssignment().set_join_operator(JoinOperator.HashJoin, [R, S])

    hint = _generate_pghintplan_hints(query, None, ops, None, pg_instance=StubPgInstance())

    assert_any_ordering(hint.query_hints, "HashJoin({})", R, S)


def test_pghintplan_hints_join_parallel_workers_warn_and_apply_to_base_tables() -> None:
    """pg_hint_plan cannot set parallel workers directly on a join, only on base tables."""
    query = query_with_tables(R, S)
    ops = PhysicalOperatorAssignment().set_join_operator(
        JoinOperatorAssignment(JoinOperator.HashJoin, [R, S], parallel_workers=2)
    )

    with pytest.warns(HintWarning, match="Cannot directly set parallel workers"):
        hint = _generate_pghintplan_hints(query, None, ops, None, pg_instance=StubPgInstance())

    assert "Parallel(r 2 hard)" in hint.query_hints
    assert "Parallel(s 2 hard)" in hint.query_hints


def test_pghintplan_hints_render_intermediate_operator() -> None:
    query = query_with_tables(R, S)
    ops = PhysicalOperatorAssignment().set_intermediate_operator(IntermediateOperator.Memoize, [R, S])

    hint = _generate_pghintplan_hints(query, None, ops, None, pg_instance=StubPgInstance())

    assert_any_ordering(hint.query_hints, "Memoize({})", R, S)


def test_pghintplan_hints_unsupported_intermediate_operator_warns_and_is_skipped() -> None:
    """Sort is not in `PGHintPlanOptimizerHints`, so pg_hint_plan cannot enforce it directly."""
    query = query_with_tables(R, S)
    ops = PhysicalOperatorAssignment().set_intermediate_operator(IntermediateOperator.Sort, [R, S])

    with pytest.warns(HintWarning, match="Cannot enforce operator"):
        hint = _generate_pghintplan_hints(query, None, ops, None, pg_instance=StubPgInstance())

    assert "Sort" not in hint.query_hints


def test_pghintplan_hints_render_global_settings() -> None:
    query = query_with_tables(R)
    ops = PhysicalOperatorAssignment().set(ScanOperator.IndexScan, False)

    hint = _generate_pghintplan_hints(query, None, ops, None, pg_instance=StubPgInstance())

    assert "Set(enable_indexscan False)" in hint.query_hints


def test_pghintplan_hints_render_cardinality() -> None:
    query = query_with_tables(R, S)
    params = PlanParameterization().add_cardinality([R, S], 99)

    hint = _generate_pghintplan_hints(query, None, None, params, pg_instance=StubPgInstance())

    assert_any_ordering(hint.query_hints, "Rows({} #99)", R, S)


def test_pghintplan_hints_render_worker_hints() -> None:
    query = query_with_tables(R, S)
    params = PlanParameterization().set_workers([R, S], 3)

    hint = _generate_pghintplan_hints(query, None, None, params, pg_instance=StubPgInstance())

    assert_any_ordering(hint.query_hints, "Parallel({} 3 hard)", R, S)


def test_pghintplan_hints_skip_worker_hint_for_a_single_worker() -> None:
    """A worker count of 1 means "sequential"; there is nothing to hint."""
    query = query_with_tables(R, S)
    params = PlanParameterization().set_workers([R, S], 1)

    hint = _generate_pghintplan_hints(query, None, None, params, pg_instance=StubPgInstance())

    assert "Parallel" not in hint.query_hints


def test_pghintplan_hints_system_settings_become_preparatory_statements() -> None:
    """Unlike pg_lab, pg_hint_plan cannot set arbitrary GUCs inline and needs a SET LOCAL beforehand."""
    query = query_with_tables(R)
    params = PlanParameterization().set_system_settings("work_mem", "256MB")

    hint = _generate_pghintplan_hints(query, None, None, params, pg_instance=StubPgInstance())

    assert hint.preparatory_statements == "SET LOCAL work_mem TO '256MB';"
    assert "work_mem" not in hint.query_hints


def test_pghintplan_hints_execution_mode_is_unsupported() -> None:
    query = query_with_tables(R)
    params = PlanParameterization()
    params.execution_mode = "parallel"

    with pytest.warns(HintWarning, match="does not support execution mode"):
        _generate_pghintplan_hints(query, None, None, params, pg_instance=StubPgInstance())


def test_pghintplan_hints_wrap_in_the_comment_block() -> None:
    query = query_with_tables(R)

    hint = _generate_pghintplan_hints(query, None, None, None, pg_instance=StubPgInstance())

    assert hint.query_hints.startswith("/*+")
    assert hint.query_hints.rstrip().endswith("*/")


# -- sanity: neither dialect ever emits NaN into the hint text -----------------------------------------------


def test_neither_dialect_ever_emits_nan_literally() -> None:
    """A regression against `math.nan` leaking into hint text through an unset parallel-worker default."""
    assert not math.isnan(0)  # guard against accidental no-op test

    ops = PhysicalOperatorAssignment().set_scan_operator(ScanOperator.SequentialScan, R)
    pglab_hint = _generate_pglab_hints(None, ops, None)
    assert "nan" not in pglab_hint.query_hints.lower()

    query = query_with_tables(R)
    pghintplan_hint = _generate_pghintplan_hints(query, None, ops, None, pg_instance=StubPgInstance())
    assert "nan" not in pghintplan_hint.query_hints.lower()
