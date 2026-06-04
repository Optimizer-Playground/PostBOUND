"""Provides logic to generate pretty strings for query objects."""

from __future__ import annotations

import functools
import warnings
from collections.abc import Callable, Sequence
from typing import Literal, Optional

from .. import util
from .._core import quote
from ._qal import (
    AbstractPredicate,
    ArrayAccessExpression,
    ArrayExpression,
    BetweenPredicate,
    BinaryPredicate,
    CaseExpression,
    CastExpression,
    ColumnExpression,
    CommonTableExpression,
    CompoundOperator,
    CompoundPredicate,
    DirectTableSource,
    ExceptClause,
    ExplicitFromClause,
    From,
    FunctionExpression,
    FunctionTableSource,
    GroupBy,
    Hint,
    ImplicitFromClause,
    InPredicate,
    IntersectClause,
    JoinTableSource,
    Limit,
    MathExpression,
    OrderBy,
    OrderByExpression,
    QuantifierExpression,
    Select,
    SelectStatement,
    SetQuery,
    SqlExpression,
    SqlQuery,
    StarExpression,
    StaticValueExpression,
    SubqueryExpression,
    SubqueryTableSource,
    TableSource,
    UnaryPredicate,
    UnionClause,
    ValuesTableSource,
    ValuesWithQuery,
    Where,
    WindowExpression,
)

DefaultIndent = 2
"""The default amount of whitespace that is used to indent specific parts of the SQL query."""

SqlDialect = Literal["vanilla", "postgres"]
"""The different flavors of SQL syntax that are supported by the formatter."""


def _increase_indentation(
    content: str, indentation: int = DefaultIndent
) -> str:
    """Prefixes all lines in a string by a given amount of whitespace.

    This breaks the input string into separate lines and adds whitespace equal to the desired amount at the start of
    each line.

    Parameters
    ----------
    content : str
        The string that should be prefixed/whose indentation should be increased.
    indentation : Optional[int], optional
        The amount of whitespace that should be added, by default `FormatIndentDepth`

    Returns
    -------
    str
        The indented string
    """
    indent_prefix = "\n" + " " * indentation
    return " " * indentation + indent_prefix.join(content.split("\n"))


class FormattingSubqueryExpression(SubqueryExpression):
    """Wraps subquery expressions to ensure that they are also pretty-printed and aligned properly.

    This class acts as a decorator around the actual subquery. It can be used entirely as a replacement of the original
    query.

    Parameters
    ----------
    original_expression : SubqueryExpression
        The actual subquery.
    flavor : SqlDialect
        The SQL dialect to emit
    inline_hint_block : bool
        Whether potential hint blocks of the subquery should be printed as preceding blocks or as inline blocks (see
        `legacy_format_quick` for details)
    indentation : int
        The current amount of indentation that should be used for the subquery. While pretty-printing, additional
        indentation levels can be inserted for specific parts of the query.
    """

    def __init__(
        self,
        original_expression: SubqueryExpression,
        *,
        flavor: SqlDialect,
        inline_hint_block: bool,
        indentation: int,
    ) -> None:
        super().__init__(original_expression.query)
        self._flavor = flavor
        self._inline_hint_block = inline_hint_block
        self._indentation = indentation

    def __str__(self) -> str:
        formatted = (
            "("
            + format_quick(
                self.query,
                flavor=self._flavor,
                inline_hint_block=self._inline_hint_block,
                trailing_semicolon=False,
            )
            + ")"
        )
        prefix = " " * (self._indentation + 2)
        if "\n" not in formatted:
            return prefix + formatted

        lines = formatted.split("\n")
        indented_lines = [f"\n{prefix}{lines[0]}"] + [
            prefix + line for line in lines[1:]
        ]
        return "\n".join(indented_lines)


class FormattingCaseExpression(CaseExpression):
    def __init__(
        self, cases, *, simple_expr, else_expr, indentation: int
    ) -> None:
        super().__init__(cases, simple_expr=simple_expr, else_expr=else_expr)
        self._indentation = indentation

    def __str__(self) -> str:
        case_indentation = " " * (self._indentation + 2)
        preamble = (
            f"CASE {self.simple_expression}"
            if self.simple_expression
            else "CASE"
        )
        case_block_entries = [preamble]
        for case, value in self.cases:
            case_block_entries.append(
                f"{case_indentation}WHEN {case} THEN {value}"
            )
        if self.else_expression is not None:
            case_block_entries.append(
                f"{case_indentation}ELSE {self.else_expression}"
            )
        case_block_entries.append("END")
        return "\n".join(case_block_entries)


def _quick_format_cte(
    cte_clause: CommonTableExpression, *, flavor: SqlDialect
) -> list[str]:
    """Formatting logic for Common Table Expressions

    Parameters
    ----------
    cte_clause : clauses.CommonTableExpression
        The clause to format
    flavor : SqlDialect
        The SQL dialect to emit

    Returns
    -------
    list[str]
        The pretty-printed parts of the CTE, indented as necessary.
    """
    recursive_info = "RECURSIVE " if cte_clause.recursive else ""

    if len(cte_clause.queries) == 1:
        cte_query = cte_clause.queries[0]
        if isinstance(cte_query, ValuesWithQuery):
            cte_structure = ", ".join(
                quote(target.name) for target in cte_query.cols
            )
            cte_structure = f"({cte_structure})" if cte_structure else ""
            cte_header = (
                f"WITH {quote(cte_query.target_name)}{cte_structure} AS ("
            )
            cte_rows: list[str] = []
            for row in cte_query.rows:
                row_values = ", ".join(str(value) for value in row)
                cte_rows.append(f"({row_values})")
            cte_rows[0] = f"VALUES {cte_rows[0]}" if cte_rows else ""
            cte_content = ",\n".join(cte_rows)
        else:
            match cte_query.materialized:
                case None:
                    mat_info = ""
                case True:
                    mat_info = "MATERIALIZED "
                case False:
                    mat_info = "NOT MATERIALIZED "

            cte_header = f"WITH {recursive_info}{quote(cte_query.target_name)} AS {mat_info}("
            cte_content = format_quick(
                cte_query.query, flavor=flavor, trailing_semicolon=False
            )
        cte_content = _increase_indentation(cte_content)
        cte_footer = ")"
        return [cte_header, cte_content, cte_footer]

    first_cte, *remaining_ctes = cte_clause.queries
    first_content = _increase_indentation(
        format_quick(first_cte.query, flavor=flavor, trailing_semicolon=False)
    )
    mat_info = (
        ""
        if first_cte.materialized is None
        else (
            "MATERIALIZED " if first_cte.materialized else "NOT MATERIALIZED "
        )
    )
    formatted_parts: list[str] = [
        f"WITH {recursive_info} {quote(first_cte.target_name)} AS {mat_info}(",
        first_content,
    ]
    for next_cte in remaining_ctes:
        match next_cte.materialized:
            case None:
                mat_info = ""
            case True:
                mat_info = "MATERIALIZED "
            case False:
                mat_info = "NOT MATERIALIZED "

        current_header = f"), {quote(next_cte.target_name)} AS {mat_info}("
        cte_content = _increase_indentation(
            format_quick(
                next_cte.query, flavor=flavor, trailing_semicolon=False
            )
        )

        formatted_parts.append(current_header)
        formatted_parts.append(cte_content)

    formatted_parts.append(")")
    return formatted_parts


def _quick_format_select(
    select_clause: Select,
    *,
    flavor: SqlDialect,
    inlined_hint_block: Optional[Hint] = None,
) -> list[str]:
    """Quick and dirty formatting logic for *SELECT* clauses.

    Up to 3 targets are put on the same line, otherwise each target is put on a separate line.

    Parameters
    ----------
    select_clause : Select
        The clause to format
    flavor : SqlDialect
        The SQL dialect to emit
    inlined_hint_block : Optional[Hint], optional
        A hint block that should be inserted after the *SELECT* statement. Defaults to *None* which indicates that
        no block should be inserted that way

    Returns
    -------
    list[str]
        The pretty-printed parts of the clause, indented as necessary.
    """
    prefix = " " * DefaultIndent
    hint_text = f"{inlined_hint_block} " if inlined_hint_block else ""
    if select_clause.is_distinct():
        on_cols = ", ".join(str(col) for col in select_clause.distinct_on)
        select_prefix = (
            f"SELECT DISTINCT ON ({on_cols})" if on_cols else "SELECT DISTINCT"
        )
    else:
        on_cols = ""
        select_prefix = "SELECT"

    if len(select_clause.targets) >= 3 or on_cols:
        first_target, *remaining_targets = select_clause.targets
        formatted_targets = (
            [f"{hint_text}{prefix}{first_target}"]
            if on_cols
            else [f"{select_prefix} {hint_text}{first_target}"]
        )
        formatted_targets += [
            f"{prefix}{target}" for target in remaining_targets
        ]
        for i in range(len(formatted_targets) - 1):
            formatted_targets[i] += ","
        if on_cols:
            formatted_targets.insert(0, select_prefix)
        return formatted_targets
    else:
        targets_text = ", ".join(
            str(target) for target in select_clause.targets
        )
        return [f"{select_prefix} {hint_text}{targets_text}"]


def _quick_format_implicit_from(
    from_clause: ImplicitFromClause, *, flavor: SqlDialect
) -> list[str]:
    """Quick and dirty formatting logic for implicit *FROM* clauses.

    Up to 3 tables are put on the same line, otherwise each table is put on its own line.

    Parameters
    ----------
    from_clause : ImplicitFromClause
        The clause to format
    flavor : SqlDialect
        The SQL dialect to emit

    Returns
    -------
    list[str]
        The pretty-printed parts of the clause, indented as necessary.
    """
    tables = list(from_clause.itertables())
    if not tables:
        return []
    elif len(tables) > 3:
        first_table, *remaining_tables = tables
        formatted_tables = [f"FROM {first_table}"]
        formatted_tables += [
            ((" " * DefaultIndent) + str(tab)) for tab in remaining_tables
        ]
        for i in range(len(formatted_tables) - 1):
            formatted_tables[i] += ","
        return formatted_tables
    else:
        tables_str = ", ".join(str(tab) for tab in tables)
        return [f"FROM {tables_str}"]


def _quick_format_tablesource(
    table_source: TableSource, *, flavor: SqlDialect
) -> list[str]:
    """Quick and dirty formatting logic for table sources.

    Parameters
    ----------
    table_source : TableSource
        The table source to format
    flavor : SqlDialect
        The SQL dialect to emit

    Returns
    -------
    list[str]
        The pretty-printed parts of the table source, indented as necessary.
    """

    prefix = " " * DefaultIndent
    match table_source:
        case DirectTableSource() | ValuesTableSource() | FunctionTableSource():
            return [str(table_source)]

        case SubqueryTableSource():
            elems: list[str] = ["LATERAL ("] if table_source.lateral else ["("]
            subquery_elems = format_quick(
                table_source.query, flavor=flavor, trailing_semicolon=False
            ).split("\n")
            subquery_elems = [
                ((" " * DefaultIndent) + str(child)) for child in subquery_elems
            ]
            elems.extend(subquery_elems)
            elems.append(")")
            if table_source.target_name:
                elems[-1] += f" AS {quote(table_source.target_name)}"
            return elems

        case JoinTableSource():
            if isinstance(table_source.left, DirectTableSource) and isinstance(
                table_source.right, DirectTableSource
            ):
                # case R JOIN S ON ...
                elems = [
                    str(table_source.left),
                    f"{prefix}{table_source.join_type} {table_source.right}",
                ]
                if table_source.join_condition:
                    elems[-1] += f" ON {table_source.join_condition}"
                return elems

            if isinstance(table_source.left, JoinTableSource) and isinstance(
                table_source.right, DirectTableSource
            ):
                # case R JOIN S ON ... JOIN T ON ...
                elems = _quick_format_tablesource(
                    table_source.left, flavor=flavor
                )
                join_condition = (
                    f" ON {table_source.join_condition}"
                    if table_source.join_condition
                    else ""
                )
                elems.append(
                    f"{prefix}{table_source.join_type} {table_source.right}{join_condition}"
                )
                return elems

            if isinstance(table_source.left, DirectTableSource) and isinstance(
                table_source.right, JoinTableSource
            ):
                elems = [str(table_source.left)]
                right_children = _quick_format_tablesource(
                    table_source.right, flavor=flavor
                )
                right_children[0] = (
                    f"{table_source.join_type} ({right_children[0]}"
                )
                right_children[1:] = [
                    ((" " * DefaultIndent) + str(child))
                    for child in right_children[1:]
                ]
                elems += right_children
                elems.append(")")
                if table_source.join_condition:
                    elems[-1] += f" ON {table_source.join_condition}"
                return elems

            elems: list[str] = []
            elems += _quick_format_tablesource(table_source.left, flavor=flavor)
            elems.append(f"{table_source.join_type}")
            elems += _quick_format_tablesource(
                table_source.right, flavor=flavor
            )
            if table_source.join_condition:
                elems[-1] += f" ON {table_source.join_condition}"
            elems = [((" " * DefaultIndent) + str(child)) for child in elems]
            return elems

        case _:
            raise ValueError(
                "Unsupported table source type: " + str(table_source)
            )


def _quick_format_explicit_from(
    from_clause: ExplicitFromClause, *, flavor: SqlDialect
) -> list[str]:
    """Quick and dirty formatting logic for explicit *FROM* clauses.

    This function just puts each *JOIN ON* statement on a separate line.

    Parameters
    ----------
    from_clause : ExplicitFromClause
        The clause to format
    flavor : SqlDialect
        The SQL dialect to emit

    Returns
    -------
    list[str]
        The pretty-printed parts of the clause, indented as necessary.
    """
    items = _quick_format_tablesource(from_clause.root, flavor=flavor)
    items[0] = f"FROM {items[0]}"
    return items


def _quick_format_general_from(
    from_clause: From, *, flavor: SqlDialect
) -> list[str]:
    """Quick and dirty formatting logic for general *FROM* clauses.

    This function just puts each part of the *FROM* clause on a separate line.

    Parameters
    ----------
    from_clause : From
        The clause to format
    flavor : SqlDialect
        The SQL dialect to emit

    Returns
    -------
    list[str]
        The pretty-printed parts of the clause, indented as necessary.
    """
    elems: list[str] = ["FROM"]
    for table_source in from_clause.items:
        current_elems = _quick_format_tablesource(table_source, flavor=flavor)
        current_elems = [
            ((" " * DefaultIndent) + str(child)) for child in current_elems
        ]
        current_elems[-1] += ","
        elems += current_elems
    elems[-1] = elems[-1].removesuffix(",")
    return elems


def _quick_format_predicate(
    predicate: AbstractPredicate, *, flavor: SqlDialect
) -> list[str]:
    """Quick and dirty formatting logic for arbitrary (i.e. also compound) predicates.

    *AND* conditions are put on separate lines, everything else is put on one line.

    Parameters
    ----------
    predicate : AbstractPredicate
        The predicate to format
    flavor : SqlDialect
        The SQL dialect to emit

    Returns
    -------
    list[str]
        The pretty-printed parts of the predicate, indented as necessary.
    """
    if not isinstance(predicate, CompoundPredicate):
        return [str(predicate)]
    compound_pred: CompoundPredicate = predicate
    if compound_pred.operation == CompoundOperator.And:
        first_child, *remaining_children = compound_pred.children
        return [str(first_child)] + [
            "AND " + str(child) for child in remaining_children
        ]
    return [str(compound_pred)]


def _quick_format_where(
    where_clause: Where, *, flavor: SqlDialect
) -> list[str]:
    """Quick and dirty formatting logic for *WHERE* clauses.

    This function just puts each part of an *AND* condition on a separate line and leaves the parts of *OR*
    conditions, negations or base predicates on the same line.

    Parameters
    ----------
    where_clause : Where
        The clause to format
    flavor : SqlDialect
        The SQL dialect to emit

    Returns
    -------
    list[str]
        The pretty-printed parts of the clause, indented as necessary.
    """
    first_pred, *additional_preds = _quick_format_predicate(
        where_clause.predicate, flavor=flavor
    )
    return [f"WHERE {first_pred}"] + [
        ((" " * DefaultIndent) + str(pred)) for pred in additional_preds
    ]


def _quick_format_groupby(
    groupby_clause: GroupBy, *, flavor: SqlDialect
) -> list[str]:
    """Quick and dirty formatting logic for *GROUP BY* clauses.

    Parameters
    ----------
    groupby_clause : GroupBy
        The clause to format
    flavor : SqlDialect
        The SQL dialect to emit

    Returns
    -------
    list[str]
        The pretty-printed parts of the clause, indented as necessary.
    """
    distinct_text = "DISTINCT " if groupby_clause.distinct else ""
    if len(groupby_clause.group_columns) > 3:
        first_target, *remaining_targets = groupby_clause.group_columns
        formatted_targets = [f"GROUP BY {distinct_text}{first_target}"]
        formatted_targets += [
            ((" " * DefaultIndent) + str(target))
            for target in remaining_targets
        ]
        for i in range(len(formatted_targets) - 1):
            formatted_targets[i] += ","
        return formatted_targets
    else:
        targets_text = ", ".join(str(target) for target in groupby_clause)
        return [f"GROUP BY {distinct_text}{targets_text}"]


def _quick_format_limit(
    limit_clause: Limit, *, flavor: SqlDialect
) -> list[str]:
    """Quick and dirty formatting logic for *FETCH FIRST* / *LIMIT* clauses.

    This produces output that is equivalent to the SQL standard's syntax to denote limit clauses and splits the limit
    and offset parts onto separate lines.

    Parameters
    ----------
    limit_clause : Limit
        The clause to format
    flavor : SqlDialect
        The SQL dialect to emit

    Returns
    -------
    list[str]
        The pretty-printed parts of the clause, indented as necessary.
    """
    match flavor:
        case "vanilla":
            fetch_direction = limit_clause.fetch_direction.upper()
            if limit_clause.limit and limit_clause.offset:
                return [
                    f"FETCH {fetch_direction} {limit_clause.limit} ROWS ONLY",
                    f"OFFSET {limit_clause.offset} ROWS",
                ]
            elif limit_clause.limit:
                return [
                    f"FETCH {fetch_direction} {limit_clause.limit} ROWS ONLY"
                ]
            elif limit_clause.offset:
                return [f"OFFSET {limit_clause.offset} ROWS"]
            else:
                return []

        case "postgres" if limit_clause.fetch_direction in {"first", "next"}:
            if limit_clause.limit and limit_clause.offset:
                return [
                    f"LIMIT {limit_clause.limit}",
                    f"OFFSET {limit_clause.offset}",
                ]
            elif limit_clause.limit:
                return [f"LIMIT {limit_clause.limit}"]
            elif limit_clause.offset:
                return [f"OFFSET {limit_clause.offset}"]
            return []

        case "postgres" if limit_clause.fetch_direction in {"prior", "last"}:
            warnings.warn(
                "Postgres does not support FETCH PRIOR and FETCH LAST. Falling back to naive formatting"
            )
            return [str(limit_clause)]

        case _:
            warnings.warn(
                "Unknown SQL flavor for LIMIT clauses. Falling back to naive formatting"
            )
            return [str(limit_clause)]


def _expression_prettifier(
    expression: SqlExpression,
    *,
    flavor: SqlDialect,
    inline_hints: bool,
    indentation: int,
) -> SqlExpression:
    """Handler method for `transform.replace_expressions` to apply our custom formatting expressions.

    Parameters
    ----------
    expression : SqlExpression
        The expression to replace.
    flavor : SqlDialect
        The SQL dialect to emit
    inline_hints : bool
        Whether potential hint blocks should be inserted as part of the *SELECT* clause rather than before the
        actual query.
    indentation : int
        The amount of indentation to use for the subquery

    Returns
    -------
    SqlExpression
        A semantically equivalent version of the original expression that uses our custom formatting rules
    """
    target = type(expression)
    match expression:
        case StaticValueExpression() | ColumnExpression() | StarExpression():
            return expression
        case SubqueryExpression():
            return FormattingSubqueryExpression(
                expression,
                flavor=flavor,
                inline_hint_block=inline_hints,
                indentation=indentation,
            )
        case CaseExpression(cases, simple_expr, else_expr):
            replaced_cases: list[tuple[AbstractPredicate, SqlExpression]] = []
            for condition, result in cases:
                replaced_condition = _expression_prettifier(
                    condition,
                    flavor=flavor,
                    inline_hints=inline_hints,
                    indentation=indentation + 2,
                )
                replaced_result = _expression_prettifier(
                    result,
                    flavor=flavor,
                    inline_hints=inline_hints,
                    indentation=indentation + 2,
                )
                replaced_cases.append((replaced_condition, replaced_result))
            replaced_simple = (
                _expression_prettifier(
                    simple_expr,
                    flavor=flavor,
                    inline_hints=inline_hints,
                    indentation=indentation + 2,
                )
                if simple_expr
                else None
            )
            replaced_else = (
                _expression_prettifier(
                    else_expr,
                    flavor=flavor,
                    inline_hints=inline_hints,
                    indentation=indentation + 2,
                )
                if else_expr
                else None
            )
            return FormattingCaseExpression(
                replaced_cases,
                simple_expr=replaced_simple,
                else_expr=replaced_else,
                indentation=indentation,
            )
        case CastExpression(casted_expression, typ, params, array):
            replaced_cast = _expression_prettifier(
                casted_expression,
                flavor=flavor,
                inline_hints=inline_hints,
                indentation=indentation,
            )
            return (
                target(replaced_cast, typ, type_params=params)
                if flavor == "vanilla"
                else _PostgresCastExpression(
                    replaced_cast, typ, type_params=params, array_type=array
                )
            )
        case MathExpression(op, lhs, rhs):
            replaced_lhs = _expression_prettifier(
                lhs,
                flavor=flavor,
                inline_hints=inline_hints,
                indentation=indentation,
            )
            rhs = util.enlist(rhs) if rhs else []
            replaced_rhs = [
                _expression_prettifier(
                    expr,
                    flavor=flavor,
                    inline_hints=inline_hints,
                    indentation=indentation,
                )
                for expr in rhs
            ]
            replaced_rhs = util.simplify(replaced_rhs)
            return target(op, replaced_lhs, replaced_rhs)
        case ArrayAccessExpression(array, ind, lo, hi):
            # This has to be implemented before the FunctionExpression since that is a supertype of this
            replaced_array = _expression_prettifier(
                array,
                flavor=flavor,
                inline_hints=inline_hints,
                indentation=indentation,
            )
            replaced_ind = (
                _expression_prettifier(
                    ind,
                    flavor=flavor,
                    inline_hints=inline_hints,
                    indentation=indentation,
                )
                if ind is not None
                else None
            )
            replaced_hi = (
                _expression_prettifier(
                    hi,
                    flavor=flavor,
                    inline_hints=inline_hints,
                    indentation=indentation,
                )
                if hi is not None
                else None
            )
            replaced_lo = (
                _expression_prettifier(
                    lo,
                    flavor=flavor,
                    inline_hints=inline_hints,
                    indentation=indentation,
                )
                if lo is not None
                else None
            )
            return target(
                replaced_array,
                idx=replaced_ind,
                lower_idx=replaced_lo,
                upper_idx=replaced_hi,
            )
        case FunctionExpression(fn, args, kw_args, distinct, cond):
            replaced_args = [
                _expression_prettifier(
                    arg,
                    flavor=flavor,
                    inline_hints=inline_hints,
                    indentation=indentation,
                )
                for arg in args
            ]
            replaced_kwargs = {
                kw: _expression_prettifier(
                    arg,
                    flavor=flavor,
                    inline_hints=inline_hints,
                    indentation=indentation,
                )
                for kw, arg in kw_args.items()
            }
            replaced_cond = (
                _expression_prettifier(
                    cond,
                    flavor=flavor,
                    inline_hints=inline_hints,
                    indentation=indentation,
                )
                if cond
                else None
            )
            return FunctionExpression(
                fn,
                replaced_args,
                keyword_args=replaced_kwargs,
                distinct=distinct,
                filter_where=replaced_cond,
            )
        case WindowExpression(fn, parts, ordering, cond):
            replaced_fn = _expression_prettifier(
                fn,
                flavor=flavor,
                inline_hints=inline_hints,
                indentation=indentation,
            )
            replaced_parts = [
                _expression_prettifier(
                    part,
                    flavor=flavor,
                    inline_hints=inline_hints,
                    indentation=indentation,
                )
                for part in parts
            ]
            replaced_cond = (
                _expression_prettifier(
                    cond,
                    flavor=flavor,
                    inline_hints=inline_hints,
                    indentation=indentation,
                )
                if cond
                else None
            )

            replaced_order_exprs: list[OrderByExpression] = []
            for order in ordering or []:
                replaced_expr = _expression_prettifier(
                    order.column,
                    flavor=flavor,
                    inline_hints=inline_hints,
                    indentation=indentation,
                )
                replaced_order_exprs.append(
                    OrderByExpression(
                        replaced_expr, order.ascending, order.nulls_first
                    )
                )
            replaced_ordering = (
                OrderBy(replaced_order_exprs) if replaced_order_exprs else None
            )

            return target(
                replaced_fn,
                partitioning=replaced_parts,
                ordering=replaced_ordering,
                filter_condition=replaced_cond,
            )
        case ArrayExpression(elems):
            pretty_elems = [
                _expression_prettifier(
                    elem,
                    flavor=flavor,
                    inline_hints=inline_hints,
                    indentation=indentation,
                )
                for elem in elems
            ]
            return target(pretty_elems)
        case QuantifierExpression(child, quantifier):
            replaced_child = _expression_prettifier(
                child,
                flavor=flavor,
                inline_hints=inline_hints,
                indentation=indentation,
            )
            return target(replaced_child, quantifier=quantifier)
        case BinaryPredicate(op, lhs, rhs):
            replaced_lhs = _expression_prettifier(
                lhs,
                flavor=flavor,
                inline_hints=inline_hints,
                indentation=indentation,
            )
            replaced_rhs = _expression_prettifier(
                rhs,
                flavor=flavor,
                inline_hints=inline_hints,
                indentation=indentation,
            )
            return BinaryPredicate(op, replaced_lhs, replaced_rhs)
        case BetweenPredicate(col, lo, hi):
            replaced_col = _expression_prettifier(
                col,
                flavor=flavor,
                inline_hints=inline_hints,
                indentation=indentation,
            )
            replaced_lo = _expression_prettifier(
                lo,
                flavor=flavor,
                inline_hints=inline_hints,
                indentation=indentation,
            )
            replaced_hi = _expression_prettifier(
                hi,
                flavor=flavor,
                inline_hints=inline_hints,
                indentation=indentation,
            )
            return target(replaced_col, (replaced_lo, replaced_hi))
        case InPredicate(col, vals):
            replaced_col = _expression_prettifier(
                col,
                flavor=flavor,
                inline_hints=inline_hints,
                indentation=indentation,
            )
            replaced_vals = [
                _expression_prettifier(
                    val,
                    flavor=flavor,
                    inline_hints=inline_hints,
                    indentation=indentation,
                )
                for val in vals
            ]
            return target(replaced_col, replaced_vals)
        case UnaryPredicate(col, op):
            replaced_col = _expression_prettifier(
                col,
                flavor=flavor,
                inline_hints=inline_hints,
                indentation=indentation,
            )
            return UnaryPredicate(replaced_col, op)
        case CompoundPredicate(op, children) if op in {
            CompoundOperator.And,
            CompoundOperator.Or,
        }:
            replaced_children = [
                _expression_prettifier(
                    child,
                    flavor=flavor,
                    inline_hints=inline_hints,
                    indentation=indentation,
                )
                for child in children
            ]
            return target(op, replaced_children)
        case CompoundPredicate(op, child) if op == CompoundOperator.Not:
            replaced_child = _expression_prettifier(
                child,
                flavor=flavor,
                inline_hints=inline_hints,
                indentation=indentation,
            )
            return target(op, [replaced_child])
        case _:
            raise ValueError(
                f"Unsupported expression type {type(expression)}: {expression}"
            )


def _quick_format_set_query(
    query: SetQuery,
    *,
    flavor: SqlDialect,
    inline_hint_block: bool,
    trailing_semicolon: bool,
    custom_formatter: Optional[Callable[[SqlQuery], SqlQuery]],
) -> str:
    """Quick and dirty formatting logic for set queries.

    Parameters
    ----------
    query : SetQuery
        The query to format
    inline_hint_block : bool
        Whether potential hint blocks should be inserted as part of the *SELECT* clause rather than before the
    trailing_semicolon : bool
        Whether to append a semicolon to the formatted query
    custom_formatter : Optional[Callable[[SqlQuery], SqlQuery]]
        A post-processing formatting service to apply to the SQL query after all preparatory steps have been performed,
        but *before* the actual formatting is started. This can be used to inject custom clause or expression
        formatting rules that are necessary to adhere to specific SQL syntax deviations for a database system. Defaults
        to *None* which skips this step.

    Returns
    -------
    str
        The pretty-printed parts of the query, indented as necessary.
    """
    # while formatting the nested queries, we still need to use rstrip in addition to trailing_semicolon=False in order to
    # format nested set queries correctly
    query_parts: list[str] = []

    if query.hints:
        query_parts.append(str(query.hints))
    if query.explain:
        query_parts.append(str(query.explain))
    if query.cte_clause:
        query_parts.extend(_quick_format_cte(query.cte_clause, flavor=flavor))

    left_query = format_quick(
        query.left_query,
        flavor=flavor,
        inline_hint_block=inline_hint_block,
        trailing_semicolon=False,
        custom_formatter=custom_formatter,
    ).rstrip("; ")
    query_parts.append(f"({left_query})")

    prefix = " " * DefaultIndent
    query_parts.append(f"{prefix}{query.set_operation.value}")

    right_query = format_quick(
        query.right_query,
        flavor=flavor,
        inline_hint_block=inline_hint_block,
        trailing_semicolon=False,
        custom_formatter=custom_formatter,
    ).rstrip("; ")
    query_parts.append(f"({right_query})")

    if query.orderby_clause:
        query_parts.append(str(query.orderby_clause))
    if query.limit_clause:
        query_parts.append(str(query.limit_clause))

    suffix = ";" if trailing_semicolon else ""
    if suffix:
        query_parts[-1] += suffix

    return "\n".join(query_parts)


class _PostgresCastExpression(CastExpression):
    """A specialized cast expression to handle the custom syntax for ``CAST`` statements used by Postgres."""

    def __init__(
        self,
        expression: SqlExpression,
        target_type: str,
        *,
        type_params: Optional[Sequence[SqlExpression]] = None,
        array_type: bool = False,
    ) -> None:
        super().__init__(
            expression,
            target_type,
            type_params=type_params,
            array_type=array_type,
        )

    def __str__(self) -> str:
        if self.type_params:
            type_args = ", ".join(str(arg) for arg in self.type_params)
            type_str = f"{self.target_type}({type_args})"
        else:
            type_str = self.target_type
        if self.array_type:
            type_str = f"{type_str}[]"
        casted_str = (
            str(self.casted_expression)
            if isinstance(
                self.casted_expression,
                (ColumnExpression, StaticValueExpression),
            )
            else f"({self.casted_expression})"
        )
        return f"{casted_str}::{type_str}"


def format_quick(
    query: SelectStatement,
    *,
    flavor: SqlDialect = "vanilla",
    inline_hint_block: bool = False,
    trailing_semicolon: bool = True,
    custom_formatter: Optional[Callable[[SqlQuery], SqlQuery]] = None,
) -> str:
    """Applies a quick formatting heuristic to structure the given query.

    The query will be structured as follows:

    - all clauses start at a new line
    - long clauses with multiple parts (e.g. *SELECT* clause, *FROM* clause) are split along multiple intended
      lines
    - the predicate in the *WHERE* clause is split on multiple lines along the different parts of a conjunctive
      predicate

    All other clauses are written on a single line (e.g. *GROUP BY* clause).

    Parameters
    ----------
    query : SelectStatement
        The query to format
    inline_hint_block : bool, optional
        Whether to insert a potential hint block in the *SELECT* clause (i.e. *inline* it), or leave it as a
        block preceding the actual query. Defaults to *False* which indicates that the clause should be printed
        before the actual query.
    custom_formatter : Callable[[SqlQuery], SqlQuery], optional
        A post-processing formatting service to apply to the SQL query after all preparatory steps have been performed,
        but *before* the actual formatting is started. This can be used to inject custom clause or expression
        formatting rules that are necessary to adhere to specific SQL syntax deviations for a database system. Defaults
        to *None* which skips this step.

    Returns
    -------
    str
        A pretty string representation of the query.
    """
    from .. import transform

    if isinstance(query, SetQuery):
        return _quick_format_set_query(
            query,
            flavor=flavor,
            inline_hint_block=inline_hint_block,
            trailing_semicolon=trailing_semicolon,
            custom_formatter=custom_formatter,
        )

    pretty_query_parts = []
    inlined_hint_block = None
    expression_prettifier = functools.partial(
        _expression_prettifier,
        flavor=flavor,
        inline_hints=inline_hint_block,
        indentation=DefaultIndent,
    )
    query = transform.replace_expressions(query, expression_prettifier)

    # Note: we cannot replace set operation clauses here, since they don't really exist in the SqlQuery object
    # instead, we have to handle them in the main loop

    if custom_formatter is not None:
        query = custom_formatter(query)

    for clause in query.clauses():
        if inline_hint_block and isinstance(clause, Hint):
            inlined_hint_block = clause
            continue

        match clause:
            case CommonTableExpression():
                pretty_query_parts.extend(
                    _quick_format_cte(clause, flavor=flavor)
                )
            case Select():
                pretty_query_parts.extend(
                    _quick_format_select(
                        clause,
                        flavor=flavor,
                        inlined_hint_block=inlined_hint_block,
                    )
                )
            case ImplicitFromClause():
                pretty_query_parts.extend(
                    _quick_format_implicit_from(clause, flavor=flavor)
                )
            case ExplicitFromClause():
                pretty_query_parts.extend(
                    _quick_format_explicit_from(clause, flavor=flavor)
                )
            case From():
                pretty_query_parts.extend(
                    _quick_format_general_from(clause, flavor=flavor)
                )
            case Where():
                pretty_query_parts.extend(
                    _quick_format_where(clause, flavor=flavor)
                )
            case GroupBy():
                pretty_query_parts.extend(
                    _quick_format_groupby(clause, flavor=flavor)
                )
            case Limit():
                pretty_query_parts.extend(
                    _quick_format_limit(clause, flavor=flavor)
                )
            case UnionClause() | IntersectClause() | ExceptClause():
                raise RuntimeError(
                    "Set operations should not appear in this context"
                )
            case _:
                pretty_query_parts.append(str(clause))

    if trailing_semicolon:
        pretty_query_parts[-1] += ";"
    return "\n".join(pretty_query_parts)
