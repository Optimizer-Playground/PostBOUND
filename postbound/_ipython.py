from __future__ import annotations

from IPython.lib import pretty

from ._hints import JoinTree, PhysicalOperatorAssignment, PlanParameterization
from ._qep import QueryPlan
from .qal import SqlQuery, format_quick


def pretty_print_query(query: SqlQuery, p: pretty.PrettyPrinter, cycle: bool) -> None:
    if cycle:
        p.text("SqlQuery(...)")
        return
    p.text(format_quick(query))


def pretty_print_plan(plan: QueryPlan, p: pretty.PrettyPrinter, cycle: bool) -> None:
    if cycle:
        p.text("QueryPlan(...)")
        return
    p.text(plan.explain())


def pretty_print_join_tree(join_tree: JoinTree, p: pretty.PrettyPrinter, cycle: bool) -> None:
    if cycle:
        p.text("JoinTree(...)")
        return
    p.text(join_tree.inspect())


def pretty_print_physical_ops(assignment: PhysicalOperatorAssignment, p: pretty.PrettyPrinter, cycle: bool) -> None:
    if cycle:
        p.text("PhysicalOperatorAssignment(...)")
        return
    p.text(assignment.inspect())


def pretty_print_plan_params(parameters: PlanParameterization, p: pretty.PrettyPrinter, cycle: bool) -> None:
    if cycle:
        p.text("PlanParameterization(...)")
        return
    p.text(parameters.inspect())


def setup_pretty_printers() -> None:
    pretty.for_type(SqlQuery, pretty_print_query)
    pretty.for_type(QueryPlan, pretty_print_plan)
    pretty.for_type(JoinTree, pretty_print_join_tree)
    pretty.for_type(PhysicalOperatorAssignment, pretty_print_physical_ops)
    pretty.for_type(PlanParameterization, pretty_print_plan_params)
