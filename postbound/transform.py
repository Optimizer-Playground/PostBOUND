"""This module provides tools to modify the contents of existing `SqlQuery` instances.

Since queries are designed as immutable data objects, these transformations operate by implementing new query instances.

The tools differ in their granularity, ranging from utilities that swap out individual expressions and predicates, to tools
that change the entire structure of the query.

Some important transformations include:

- `flatten_and_predicate`: Simplifies the predicate structure by moving all nested ``AND`` predicates to their parent ``AND``
- `extract_query_fragment`: Extracts parts of an original query based on a subset of its tables (i.e. induced join graph and
  filter predicates)
- `add_ec_predicates`: Expands a querie's **WHERE** clause to include all join predicates that are implied by other (equi-)
  joins
- `as_count_star_query` and `as_explain_analyze` change the query to be executed as **COUNT(*)** or **EXPLAIN ANALYZE**
  respectively

In addition to these frequently-used transformations, there are also lots of utilities that add, remove, or modify specific
parts of queries, such as individual clauses or expressions.
"""

from __future__ import annotations

import typing
import warnings
from collections.abc import Callable, Iterable, Sequence
from typing import Literal, Optional, overload

from . import util
from ._core import ColumnReference, TableReference
from .qal import (
    AbstractPredicate,
    ArrayAccessExpression,
    ArrayExpression,
    BaseClause,
    BaseProjection,
    BetweenPredicate,
    BinaryPredicate,
    CaseExpression,
    CastExpression,
    ClauseVisitor,
    ColumnExpression,
    CommonTableExpression,
    CompoundOperator,
    CompoundPredicate,
    DirectTableSource,
    ExceptClause,
    Explain,
    ExplicitFromClause,
    ExplicitSqlQuery,
    From,
    FunctionExpression,
    FunctionTableSource,
    GroupBy,
    Having,
    Hint,
    ImplicitFromClause,
    ImplicitSqlQuery,
    InPredicate,
    IntersectClause,
    JoinTableSource,
    JoinType,
    Limit,
    MathExpression,
    MixedSqlQuery,
    OrderBy,
    OrderByExpression,
    PredicateVisitor,
    QuantifierExpression,
    Select,
    SelectStatement,
    SetQuery,
    SqlExpression,
    SqlExpressionVisitor,
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
    WithQuery,
    build_query,
    determine_join_equivalence_classes,
    generate_predicates_for_equivalence_classes,
)

# TODO: at a later point in time, the entire query traversal/modification logic could be refactored to use unified
# access instead of implementing the same pattern matching and traversal logic all over again

QueryType = typing.TypeVar("QueryType", bound=SqlQuery)
"""The concrete class of a query.

This generic type is used for transformations that do not change the type of a query and operate on all the different query
types.
"""

SelectQueryType = typing.TypeVar("SelectQueryType", bound=SelectStatement)
"""The concrete class of a select query.

This generic type is used for transformations that do not change the type of a query and operate on all the different select
query types.
"""

ClauseType = typing.TypeVar("ClauseType", bound=BaseClause)
"""The concrete class of a clause.

This generic type is used for transformations that do not change the type of a clause and operate on all the different clause
types.
"""

PredicateType = typing.TypeVar("PredicateType", bound=AbstractPredicate)
"""The concrete type of a predicate.

This generic type is used for transformations that do not change the type of a predicate and operate on all the different
predicate types.
"""


def flatten_and_predicate(predicate: AbstractPredicate) -> AbstractPredicate:
    """Simplifies the predicate structure by moving all nested ``AND`` predicates to their parent ``AND`` predicate.

    For example, consider the predicate ``(R.a = S.b AND R.a = 42) AND S.b = 24``. This is transformed into the flattened
    equivalent conjunction ``R.a = S.b AND R.a = 42 AND S.b = 24``.

    This procedure continues in a recursive manner, until the first disjunction or negation is encountered. All predicates
    below that point are left as-is for the current branch of the predicate hierarchy.

    Parameters
    ----------
    predicate : AbstractPredicate
        The predicate to simplified

    Returns
    -------
    AbstractPredicate
        An equivalent version of the given `predicate`, with all conjunctions unnested
    """
    if not isinstance(predicate, CompoundPredicate):
        return predicate

    not_operation = predicate.operation == CompoundOperator.Not
    or_operation = predicate.operation == CompoundOperator.Or
    if not_operation or or_operation:
        return predicate

    flattened_children = set()
    for child in predicate.children:
        if (
            isinstance(child, CompoundPredicate)
            and child.operation == CompoundOperator.And
        ):
            flattened_child = flatten_and_predicate(child)
            if isinstance(flattened_child, CompoundPredicate):
                flattened_children |= set(flattened_child.children)
            else:
                flattened_children.add(flattened_child)
        else:
            flattened_children.add(child)

    if len(flattened_children) == 1:
        return util.simplify(flattened_children)

    return CompoundPredicate.create_and(flattened_children)


def explicit_to_implicit(source_query: ExplicitSqlQuery) -> ImplicitSqlQuery:
    """Transforms a query with an explicit ``FROM`` clause to a query with an implicit ``FROM`` clause.

    Currently, this process is only supported for explicit queries that do not contain subqueries in their ``FROM`` clause.

    Parameters
    ----------
    source_query : ExplicitSqlQuery
        The query that should be transformed

    Returns
    -------
    ImplicitSqlQuery
        An equivalent version of the given query, using an implicit ``FROM`` clause

    Raises
    ------
    ValueError
        If the `source_query` contains subquery table sources
    """
    additional_predicates: list[AbstractPredicate] = []
    complete_from_tables: list[TableReference] = []
    join_working_set: list[TableSource] = list(source_query.from_clause.items)

    while join_working_set:
        current_table_source = join_working_set.pop()

        match current_table_source:
            case DirectTableSource():
                complete_from_tables.append(current_table_source.table)
            case SubqueryTableSource():
                raise ValueError(
                    "Transforming subqueries to implicit table references is not supported yet"
                )
            case JoinTableSource() if (
                current_table_source.join_type == JoinType.InnerJoin
            ):
                join_working_set.append(current_table_source.left)
                join_working_set.append(current_table_source.right)
                additional_predicates.append(
                    current_table_source.join_condition
                )
            case _:
                raise ValueError(
                    "Unsupported table source type: "
                    + str(type(current_table_source))
                )

    final_from_clause = ImplicitFromClause.create_for(complete_from_tables)

    if source_query.where_clause:
        final_predicate = CompoundPredicate.create_and(
            [source_query.where_clause.predicate] + additional_predicates
        )
    else:
        final_predicate = CompoundPredicate.create_and(additional_predicates)

    final_predicate = flatten_and_predicate(final_predicate)
    final_where_clause = Where(final_predicate)

    return ImplicitSqlQuery(
        select_clause=source_query.select_clause,
        from_clause=final_from_clause,
        where_clause=final_where_clause,
        groupby_clause=source_query.groupby_clause,
        having_clause=source_query.having_clause,
        orderby_clause=source_query.orderby_clause,
        limit_clause=source_query.limit_clause,
        cte_clause=source_query.cte_clause,
    )


def _get_predicate_fragment(
    predicate: AbstractPredicate, referenced_tables: set[TableReference]
) -> AbstractPredicate | None:
    """Filters the predicate hierarchy to include only those base predicates that reference the given tables.

    The referenced tables operate as a superset - parts of the predicate are retained if the tables that they reference are a
    subset of the target tables.

    In the general case, the resulting predicate is no longer equivalent to the original predicate, since large portions of
    the predicate hierarchy are pruned (exactly those that do not touch the given tables). Simplifications will be applied to
    the predicate as necessary. For example, if only a single child predicate of a conjunction references the given tables, the
    conjunction is removed and the child predicate is inserted instead.

    Notice that no logical simplifications are applied. For example, if the resulting predicate fragment is the conjunction
    ``R.a < 42 AND R.a < 84``, this is not simplified into ``R.a < 42``.

    Parameters
    ----------
    predicate : AbstractPredicate
        The predicate to filter
    referenced_tables : set[TableReference]
        The superset of all allowed tables. Those parts of the `predicate` are pruned, whose tables are not a subset of the

    Returns
    -------
    AbstractPredicate | None
        The largest fragment of the original predicate that references only a subset of the `referenced_tables`. If the entire
        predicate was pruned, ``None`` is returned.

    Examples
    --------
    Consider the predicate ``R.a > 100 AND (S.b = 42 OR R.a = 42)``. The predicate fragment for table ``R`` would be
    ``R.a > 100 AND R.a = 42``.
    """
    if not isinstance(predicate, CompoundPredicate):
        return (
            predicate
            if predicate.tables().issubset(referenced_tables)
            else None
        )

    compound_predicate: CompoundPredicate = predicate
    child_fragments = [
        _get_predicate_fragment(child, referenced_tables)
        for child in compound_predicate.children
    ]
    child_fragments = [fragment for fragment in child_fragments if fragment]
    if not child_fragments:
        return None
    elif (
        len(child_fragments) == 1
        and compound_predicate.operation != CompoundOperator.Not
    ):
        return child_fragments[0]
    else:
        return CompoundPredicate(compound_predicate.operation, child_fragments)


@overload
def extract_query_fragment(
    source_query: ImplicitSqlQuery,
    referenced_tables: TableReference | Iterable[TableReference],
    *,
    projection: Literal["keep", "star", "*", "count_star"] = "keep",
) -> Optional[ImplicitSqlQuery]: ...


@overload
def extract_query_fragment(
    source_query: SetQuery,
    referenced_tables: TableReference | Iterable[TableReference],
    *,
    projection: Literal["keep", "star", "*", "count_star"] = "keep",
) -> Optional[SetQuery]: ...


@overload
def extract_query_fragment(
    source_query: SqlQuery,
    referenced_tables: TableReference | Iterable[TableReference],
    *,
    projection: Literal["keep", "star", "*", "count_star"] = "keep",
) -> Optional[SqlQuery]: ...


def extract_query_fragment(
    source_query,
    referenced_tables: TableReference | Iterable[TableReference],
    *,
    projection: Literal["keep", "star", "*", "count_star"] = "keep",
):
    """Filters a query to only include parts that reference specific tables.

    This builds a new query from the given query that contains exactly those parts of the original query's clauses that
    reference only the given tables or a subset of them.

    For example, consider the query ``SELECT * FROM R, S, T WHERE R.a = S.b AND S.c = T.d AND R.a = 42 ORDER BY S.b``
    the query fragment for tables ``R`` and ``S`` would look like this:
    ``SELECT * FROM R, S WHERE R.a = S.b AND R.a = 42 ORDER BY S.b``, whereas the query fragment for table ``S`` would
    look like ``SELECT * FROM S ORDER BY S.b``.

    Notice that this can break disjunctions: the fragment for table ``R`` of query
    ``SELECT * FROM R, S, WHERE R.a < 100 AND (R.a = 42 OR S.b = 42)`` is ``SELECT * FROM R WHERE R.a < 100 AND R.a = 42``.
    This also indicates that the fragment extraction does not perform any logical pruning of superflous predicates.

    Parameters
    ----------
    source_query : ImplicitSqlQuery | SetQuery
        The query that should be transformed. Note that the current implementation only supports implicit queries and set
        queries.
    referenced_tables : TableReference | Iterable[TableReference]
        The tables that should be extracted
    projection : Literal["keep", "star", "*", "count_star"], optional
        How the projection of the resulting query should look like. Defaults to *keep*, which retains the original
        projection/*SELECT* clause. Alternatively, the projection can be set to *star* or (literal) *\\** to use a *SELECT \\**
        projection, or to *count_star* to use a *SELECT COUNT(\\*)* projection.

    Returns
    -------
    Optional[ImplicitSqlQuery | SetQuery]
        A query that only consists of those parts of the `source_query`, that reference (a subset of) the `referenced_tables`.
        If there is no such subset, ``None`` is returned.

    Warnings
    --------
    The current implementation only works for `SetQuery` and `ImplicitSqlQuery` instances. If the `source_query` is not of one
    of these types (or contains subqueries that are not of these types), a `ValueError` is raised.

    See Also
    --------
    extract_subquery : A wrapper with sane defaults. Use this whenever possible.
    """
    if isinstance(source_query, ExplicitSqlQuery):
        source_query = explicit_to_implicit(source_query)

    referenced_tables: set[TableReference] = set(util.enlist(referenced_tables))
    if not referenced_tables.issubset(source_query.tables()):
        return None

    cte_fragment = (
        [
            with_query
            for with_query in source_query.cte_clause.queries
            if with_query.target_table in referenced_tables
        ]
        if source_query.cte_clause
        else []
    )
    cte_clause = (
        CommonTableExpression(
            cte_fragment, recursive=source_query.cte_clause.recursive
        )
        if cte_fragment
        else None
    )

    if source_query.orderby_clause:
        order_fragment = [
            order
            for order in source_query.orderby_clause.expressions
            if order.column.tables().issubset(referenced_tables)
        ]
        orderby_clause = OrderBy(order_fragment) if order_fragment else None
    else:
        orderby_clause = None

    if isinstance(source_query, SetQuery):
        left_query = extract_query_fragment(
            source_query.left_query, referenced_tables
        )
        right_query = extract_query_fragment(
            source_query.right_query, referenced_tables
        )
        return SetQuery(
            left_query,
            right_query,
            set_operation=source_query.set_operation,
            cte_clause=cte_clause,
            orderby_clause=orderby_clause,
            limit_clause=source_query.limit_clause,
            hints=source_query.hints,
            explain_clause=source_query.explain,
        )

    match projection:
        case "keep":
            select_fragment = []
            for target in source_query.select_clause:
                if target.tables() == referenced_tables or not target.columns():
                    select_fragment.append(target)

            if select_fragment:
                select_clause = Select(
                    select_fragment,
                    distinct=source_query.select_clause.is_distinct(),
                )
            else:
                select_clause = Select.star(
                    distinct=source_query.select_clause.is_distinct()
                )

        case "star" | "*":
            select_clause = Select.star()

        case "count_star":
            select_clause = Select.count_star()

        case _:
            raise ValueError("Invalid projection type: " + str(projection))

    if source_query.from_clause:
        from_clause = ImplicitFromClause(
            [
                DirectTableSource(tab)
                for tab in source_query.tables()
                if tab in referenced_tables
            ]
        )
    else:
        from_clause = None

    if source_query.where_clause:
        predicate_fragment = _get_predicate_fragment(
            source_query.where_clause.predicate, referenced_tables
        )
        where_clause = Where(predicate_fragment) if predicate_fragment else None
    else:
        where_clause = None

    if source_query.groupby_clause:
        group_column_fragment = [
            col
            for col in source_query.groupby_clause.group_columns
            if col.tables().issubset(referenced_tables)
        ]
        if group_column_fragment:
            groupby_clause = GroupBy(
                group_column_fragment, source_query.groupby_clause.distinct
            )
        else:
            groupby_clause = None
    else:
        groupby_clause = None

    if source_query.having_clause:
        having_fragment = _get_predicate_fragment(
            source_query.having_clause.condition, referenced_tables
        )
        having_clause = Having(having_fragment) if having_fragment else None
    else:
        having_clause = None

    return ImplicitSqlQuery(
        select_clause=select_clause,
        from_clause=from_clause,
        where_clause=where_clause,
        groupby_clause=groupby_clause,
        having_clause=having_clause,
        orderby_clause=orderby_clause,
        limit_clause=source_query.limit_clause,
        cte_clause=cte_clause,
    )


def extract_subquery(
    query: SqlQuery, intermediate: TableReference | Iterable[TableReference]
) -> SqlQuery:
    """Extracts a subquery from a given query based on a subset of its tables.

    The subquery consists of exactly those predicates (join and filter) of the original query that reference the given tables.
    This is mostly a convenience wrapper around `extract_query_fragment` with the following rules:

    - if the subquery extraction fails, we raise an error
    - the subquery will always be a *SELECT \\** query
    """
    subquery = extract_query_fragment(query, intermediate, projection="star")
    if subquery is None:
        raise ValueError("Could not extract subquery for the given tables")
    return subquery


def _default_subquery_name(tables: Iterable[TableReference]) -> str:
    """Constructs a valid SQL table name for a subquery consisting of specific tables.

    Parameters
    ----------
    tables : Iterable[TableReference]
        The tables that should be represented by a subquery.

    Returns
    -------
    str
        The target name of the subquery
    """
    return "_".join(table.identifier() for table in tables)


def expand_to_query(predicate: AbstractPredicate) -> ImplicitSqlQuery:
    """Provides a ``SELECT *`` query that computes the result set of a specific predicate.

    Parameters
    ----------
    predicate : AbstractPredicate
        The predicate to expand

    Returns
    -------
    ImplicitSqlQuery
        An SQL query of the form ``SELECT * FROM <predicate tables> WHERE <predicate>``.
    """
    select_clause = Select.star()
    from_clause = ImplicitFromClause.create_for(predicate.tables())
    where_clause = Where(predicate)
    return build_query([select_clause, from_clause, where_clause])


def move_into_subquery(
    query: SqlQuery, tables: Iterable[TableReference], subquery_name: str = ""
) -> SqlQuery:
    """Transforms a specific query by moving some of its tables into a subquery.

    This transformation renames all usages of columns that are now produced by the subquery to references to the virtual
    subquery table instead.

    Notice that this transformation currently only really works for "good natured" queries, i.e. mostly implicit SPJ+ queries.
    Notably, the transformation is only supported for queries that do not already contain subqueries, since moving tables
    between subqueries is a quite tricky process. Likewise, the renaming is only applied at a table-level. If the tables export
    columns of the same name, these are not renamed and the transformation fails as well. If in doubt, you should definitely
    check the output of this method for your complicated queries to prevent bad surprises!

    Parameters
    ----------
    query : SqlQuery
        The query to transform
    tables : Iterable[TableReference]
        The tables that should be placed into a subquery
    subquery_name : str, optional
        The target name of the virtual subquery table. If empty, a default name (consisting of all the subquery tables) is
        generated

    Returns
    -------
    SqlQuery
        The transformed query

    Raises
    ------
    ValueError
        If the `query` does not contain a ``FROM`` clause.
    ValueError
        If the query contains virtual tables
    ValueError
        If `tables` contains less than 2 entries. In this case, using a subquery is completely pointless.
    ValueError
        If the tables that should become part of the subquery both provide columns of the same name, and these columns are used
        in the rest of the query. This level of renaming is currently not accounted for.
    """
    if not query.from_clause:
        raise ValueError(
            "Cannot create a subquery for a query without a FROM clause"
        )
    if any(table.virtual for table in query.tables()):
        raise ValueError(
            "Cannot move into subquery for queries with virtual tables"
        )

    tables = set(tables)

    # deleted CTE check: this was not necessary because a CTE produces a virtual table. This already fails the previous test
    if len(tables) < 2:
        raise ValueError("At least two tables required")

    predicates = query.predicates()
    all_referenced_columns = util.set_union(
        clause.columns()
        for clause in query.clauses()
        if not isinstance(clause, From)
    )
    columns_from_subquery_tables = {
        column for column in all_referenced_columns if column.table in tables
    }
    if len({column.name for column in columns_from_subquery_tables}) < len(
        columns_from_subquery_tables
    ):
        raise ValueError(
            "Cannot create subquery: subquery tables export columns of the same name"
        )

    subquery_name = (
        subquery_name if subquery_name else _default_subquery_name(tables)
    )
    subquery_table = TableReference.create_virtual(subquery_name)
    renamed_columns = {
        column: ColumnReference(column.name, subquery_table)
        for column in columns_from_subquery_tables
    }

    subquery_predicates: list[AbstractPredicate] = []
    for table in tables:
        filter_predicate = predicates.filters_for(table)
        if not filter_predicate:
            continue
        subquery_predicates.append(filter_predicate)
    join_predicates = predicates.joins_between(tables, tables)
    if join_predicates:
        subquery_predicates.append(join_predicates)

    subquery_select = Select.create_for(columns_from_subquery_tables)
    subquery_from = ImplicitFromClause.create_for(tables)
    subquery_where = (
        Where(CompoundPredicate.create_and(subquery_predicates))
        if subquery_predicates
        else None
    )
    subquery_clauses = [subquery_select, subquery_from, subquery_where]
    subquery = build_query(clause for clause in subquery_clauses if clause)
    subquery_table_source = SubqueryTableSource(subquery, subquery_name)

    updated_from_sources = [
        table_source
        for table_source in query.from_clause.items
        if not table_source.tables() < tables
    ]
    update_from_clause = ImplicitFromClause.create_for(
        updated_from_sources + [subquery_table_source]
    )
    updated_predicate = (
        query.where_clause.predicate if query.where_clause else None
    )
    for predicate in subquery_predicates:
        updated_predicate = remove_predicate(updated_predicate, predicate)
    updated_where_clause = (
        Where(updated_predicate) if updated_predicate else None
    )

    updated_query = drop_clause(query, [From, Where])
    updated_query = add_clause(
        updated_query, [update_from_clause, updated_where_clause]
    )

    updated_other_clauses: list[BaseClause] = []
    for clause in updated_query.clauses():
        if isinstance(clause, From):
            continue
        renamed_clause = rename_columns_in_clause(clause, renamed_columns)
        updated_other_clauses.append(renamed_clause)
    final_query = replace_clause(updated_query, updated_other_clauses)
    return final_query


def add_ec_predicates(query: ImplicitSqlQuery) -> ImplicitSqlQuery:
    """Expands the join predicates of a query to include all predicates that are implied by the join equivalence classes.

    Parameters
    ----------
    query : ImplicitSqlQuery
        The query to analyze

    Returns
    -------
    ImplicitSqlQuery
        An equivalent query that explicitly contains all predicates from join equivalence classes.

    See Also
    --------
    determine_join_equivalence_classes
    generate_predicates_for_equivalence_classes
    """
    if not query.where_clause:
        return query
    predicates = query.predicates()

    ec_classes = determine_join_equivalence_classes(predicates.joins())
    ec_predicates = generate_predicates_for_equivalence_classes(ec_classes)

    all_predicates = list(ec_predicates) + list(predicates.filters())
    updated_where_clause = Where(CompoundPredicate.create_and(all_predicates))

    return replace_clause(query, updated_where_clause)


def as_star_query(source_query: QueryType) -> QueryType:
    """Transforms a specific query to use a ``SELECT *`` projection instead.

    Notice that this can break certain queries where a renamed column from the ``SELECT`` clause is used in other parts of
    the query, such as ``ORDER BY`` clauses (e.g. ``SELECT SUM(foo) AS f FROM bar ORDER BY f``). We currently do not undo such
    a renaming.

    Parameters
    ----------
    source_query : QueryType
        The query to transform

    Returns
    -------
    QueryType
        A variant of the input query that uses a ``SELECT *`` projection.
    """
    select = Select.star()
    query_clauses = [
        clause
        for clause in source_query.clauses()
        if not isinstance(clause, Select)
    ]
    return build_query(query_clauses + [select])


def as_count_star_query(source_query: QueryType) -> QueryType:
    """Transforms a specific query to use a ``SELECT COUNT(*)`` projection instead.

    Notice that this can break certain queries where a renamed column from the ``SELECT`` clause is used in other parts of
    the query, such as ``ORDER BY`` clauses (e.g. ``SELECT SUM(foo) AS f FROM bar ORDER BY f``). We currently do not undo such
    a renaming.

    Parameters
    ----------
    source_query : QueryType
        The query to transform

    Returns
    -------
    QueryType
        A variant of the input query that uses a ``SELECT COUNT(*)`` projection.
    """
    select = Select.count_star()
    query_clauses = [
        clause
        for clause in source_query.clauses()
        if not isinstance(clause, Select)
    ]
    return build_query(query_clauses + [select])


def drop_hints(
    query: SelectQueryType, preparatory_statements_only: bool = False
) -> SelectQueryType:
    """Removes the hint clause from a specific query.

    Parameters
    ----------
    query : SelectQueryType
        The query to transform
    preparatory_statements_only : bool, optional
        Whether only the preparatory statements from the hint block should be removed. This would retain the actual hints.
        Defaults to ``False``, which removes the entire block, no matter its contents.

    Returns
    -------
    SelectQueryType
        The query without the hint block
    """
    new_hints = (
        Hint("", query.hints.query_hints)
        if preparatory_statements_only and query.hints
        else None
    )
    query_clauses = [
        clause for clause in query.clauses() if not isinstance(clause, Hint)
    ]
    return build_query(query_clauses + [new_hints])


def as_explain(
    query: SelectQueryType, explain: Explain = Explain.plan()
) -> SelectQueryType:
    """Transforms a specific query into an ``EXPLAIN`` query.

    Parameters
    ----------
    query : SelectQueryType
        The query to transform
    explain : Explain, optional
        The ``EXPLAIN`` block to use. Defaults to a standard ``Explain.plan()`` block.

    Returns
    -------
    SelectQueryType
        The transformed query
    """
    query_clauses = [
        clause for clause in query.clauses() if not isinstance(clause, Explain)
    ]
    return build_query(query_clauses + [explain])


def as_explain_analyze(query: SelectQueryType) -> SelectQueryType:
    """Transforms a specific query into an ``EXPLAIN ANALYZE`` query.

    Parameters
    ----------
    query : SelectQueryType
        The query to transform

    Returns
    -------
    SelectQueryType
        The transformed query. It uses an ``EXPLAIN ANALYZE`` block with the default output format. If this is not desired,
        the `as_explain` transformation has to be used and the target ``EXPLAIN`` block has to be given explicitly.
    """
    return as_explain(query, Explain.explain_analyze())


def remove_predicate(
    predicate: Optional[AbstractPredicate],
    predicate_to_remove: AbstractPredicate,
) -> Optional[AbstractPredicate]:
    """Drops a specific predicate from the predicate hierarchy.

    If necessary, the hierarchy will be simplified. For example, if the `predicate_to_remove` is one of two childs of a
    conjunction, the removal would leave a conjunction of just a single predicate. In this case, the conjunction can be dropped
    altogether, leaving just the other child predicate. The same also applies to disjunctions and negations.

    Parameters
    ----------
    predicate : Optional[AbstractPredicate]
        The predicate hierarchy from which should removed. If this is ``None``, no removal is attempted.
    predicate_to_remove : AbstractPredicate
        The predicate that should be removed.

    Returns
    -------
    Optional[AbstractPredicate]
        The resulting (simplified) predicate hierarchy. Will be ``None`` if there are no meaningful predicates left after
        removal, or if the `predicate` equals the `predicate_to_remove`.
    """
    if not predicate or predicate == predicate_to_remove:
        return None
    if not isinstance(predicate, CompoundPredicate):
        return predicate

    if predicate.operation == CompoundOperator.Not:
        updated_child = remove_predicate(
            predicate.children, predicate_to_remove
        )
        return (
            CompoundPredicate.create_not(updated_child)
            if updated_child
            else None
        )

    updated_children = [
        remove_predicate(child_pred, predicate_to_remove)
        for child_pred in predicate.children
    ]
    updated_children = [
        child_pred for child_pred in updated_children if child_pred
    ]
    if not updated_children:
        return None
    elif len(updated_children) == 1:
        return updated_children[0]
    else:
        return CompoundPredicate(predicate.operation, updated_children)


def add_clause(
    query: SelectQueryType, clauses_to_add: BaseClause | Iterable[BaseClause]
) -> SelectQueryType:
    """Creates a new SQL query, potentailly with additional clauses.

    No validation is performed. Conflicts are resolved according to the rules of `build_query`. This means that the query
    can potentially be switched from an implicit query to an explicit one and vice-versa.

    Parameters
    ----------
    query : SqlQuery
        The query to which the clause(s) should be added
    clauses_to_add : BaseClause | Iterable[BaseClause]
        The new clauses

    Returns
    -------
    SqlQuery
        A new clauses consisting of the old query's clauses and the `clauses_to_add`. Duplicate clauses are overwritten by
        the `clauses_to_add`.
    """
    clauses_to_add = util.enlist(clauses_to_add)
    new_clause_types = {type(clause) for clause in clauses_to_add}
    remaining_clauses = [
        clause
        for clause in query.clauses()
        if type(clause) not in new_clause_types
    ]
    return build_query(remaining_clauses + list(clauses_to_add))


ClauseDescription = typing.Union[
    typing.Type, BaseClause, Iterable[typing.Type | BaseClause]
]
"""Denotes different ways clauses to remove can be denoted.

See Also
--------
drop_clause
"""


def drop_clause[T: SelectStatement](
    query: T, clauses_to_drop: ClauseDescription
) -> T:
    """Removes specific clauses from a query.

    The clauses can be denoted in two different ways: either as the raw type of the clause, or as an instance of the same
    clause type as the one that should be removed. Notice that the instance of the clause does not need to be equal to the
    clause of the query. It just needs to be the same type of clause.

    This method does not perform any validation, other than the rules described in `build_query`.

    Parameters
    ----------
    query : SelectQueryType
        The query to remove clauses from
    clauses_to_drop : ClauseDescription
        The clause(s) to remove. This can be a single clause type or clause instance, or an iterable of clauses types,
        intermixed with clause instances. In either way clauses of the desired types are dropped from the query.

    Returns
    -------
    SelectQueryType
        A query without the specified clauses

    Examples
    --------

    The following two calls achieve exactly the same thing: getting rid of the ``LIMIT`` clause.

    .. code-block:: python

        drop_clause(query, Limit)
        drop_clause(query, query.limit_clause)
    """
    clauses_to_drop = set(util.enlist(clauses_to_drop))
    clauses_to_drop = {
        drop if isinstance(drop, typing.Type) else type(drop)
        for drop in clauses_to_drop
    }
    remaining_clauses = [
        clause
        for clause in query.clauses()
        if type(clause) not in clauses_to_drop
    ]
    return build_query(remaining_clauses)


def replace_clause(
    query: SelectQueryType, replacements: BaseClause | Iterable[BaseClause]
) -> SelectQueryType:
    """Creates a new SQL query with the replacements being used instead of the original clauses.

    Clauses are matched on a per-type basis (including subclasses, i.e. a replacement can be a subclass of an existing clause).
    Therefore, this function does not switch a query from implicit to explicit or vice-versa. Use a combination of
    `drop_clause` and `add_clause` for that. If a replacement is not present in the original query, it is simply ignored.

    No validation other than the rules of `build_query` is performed.

    Parameters
    ----------
    query : SelectQueryType
        The query to update
    replacements : BaseClause | Iterable[BaseClause]
        The new clause instances that should be used instead of the old ones.

    Returns
    -------
    SelectQueryType
        An updated query where the matching `replacements` clauses are used in place of the clause instances that were
        originally present in the query
    """
    available_replacements: set[BaseClause] = set(util.enlist(replacements))

    replaced_clauses: list[BaseClause] = []
    for current_clause in query.clauses():
        if not available_replacements:
            replaced_clauses.append(current_clause)
            continue

        final_clause = current_clause
        for replacement_clause in available_replacements:
            if isinstance(replacement_clause, type(current_clause)):
                final_clause = replacement_clause
                break
        replaced_clauses.append(final_clause)
        if final_clause in available_replacements:
            available_replacements.remove(final_clause)

    return build_query(replaced_clauses)


def _replace_expression_in_predicate(
    predicate: Optional[PredicateType],
    replacement: Callable[[SqlExpression], SqlExpression],
) -> Optional[PredicateType]:
    """Handler to update all expressions in a specific predicate.

    This method does not perform any sanity checks on the new predicate.


    Parameters
    ----------
    predicate : PredicateType
        The predicate to update. Can be ``None``, in which case no replacement is performed.
    replacement : Callable[[SqlExpression], SqlExpression]
        A function mapping each expression to a (potentially updated) expression

    Returns
    -------
    Optional[PredicateType]
        The updated predicate. Can be ``None``, if `predicate` already was.

    Raises
    ------
    ValueError
        If the predicate is of no known type. This indicates that this method is missing a handler for a specific predicate
        type that was added later on.
    """
    if not predicate:
        return None

    if isinstance(predicate, BinaryPredicate):
        renamed_first_arg = replacement(predicate.first_argument)
        renamed_second_arg = replacement(predicate.second_argument)
        return BinaryPredicate(
            predicate.operation, renamed_first_arg, renamed_second_arg
        )
    elif isinstance(predicate, BetweenPredicate):
        renamed_col = replacement(predicate.column)
        renamed_interval_start = replacement(predicate.interval_start)
        renamed_interval_end = replacement(predicate.interval_end)
        return BetweenPredicate(
            renamed_col, (renamed_interval_start, renamed_interval_end)
        )
    elif isinstance(predicate, InPredicate):
        renamed_col = replacement(predicate.column)
        renamed_vals = [replacement(val) for val in predicate.values]
        return InPredicate(renamed_col, renamed_vals)
    elif isinstance(predicate, UnaryPredicate):
        return UnaryPredicate(
            replacement(predicate.column), predicate.operation
        )
    elif isinstance(predicate, CompoundPredicate):
        if predicate.operation == CompoundOperator.Not:
            renamed_children = [
                _replace_expression_in_predicate(
                    predicate.children, replacement
                )
            ]
        else:
            renamed_children = [
                _replace_expression_in_predicate(child, replacement)
                for child in predicate.children
            ]
        return CompoundPredicate(predicate.operation, renamed_children)
    else:
        raise ValueError("Unknown predicate type: " + str(predicate))


def _replace_expression_in_table_source(
    table_source: Optional[TableSource],
    replacement: Callable[[SqlExpression], SqlExpression],
) -> Optional[TableSource]:
    """Handler to update all expressions in a table source.

    This method does not perform any sanity checks on the updated sources.

    Parameters
    ----------
    table_source : TableSource
        The source to update. Can be ``None``, in which case no replacement is performed.
    replacement : Callable[[SqlExpression], SqlExpression]
        A function mapping each expression to a (potentially updated) expression

    Returns
    -------
    Optional[TableSource]
        The updated table source. Can be ``None``, if `table_source` already was.

    Raises
    ------
    ValueError
        If the table source is of no known type. This indicates that this method is missing a handler for a specific source
        type that was added later on.
    """
    if table_source is None:
        return None

    match table_source:
        case DirectTableSource():
            # no expressions in a plain table reference, we are done here
            return table_source
        case SubqueryTableSource():
            replaced_subquery = replacement(table_source.expression)
            assert isinstance(replaced_subquery, SubqueryExpression)
            replaced_subquery = replace_expressions(
                replaced_subquery.query, replacement
            )
            return SubqueryTableSource(
                replaced_subquery,
                table_source.target_name,
                lateral=table_source.lateral,
            )
        case JoinTableSource():
            replaced_left = _replace_expression_in_table_source(
                table_source.left, replacement
            )
            replaced_right = _replace_expression_in_table_source(
                table_source.right, replacement
            )
            replaced_condition = _replace_expression_in_predicate(
                table_source.join_condition, replacement
            )
            return JoinTableSource(
                replaced_left,
                replaced_right,
                join_condition=replaced_condition,
                join_type=table_source.join_type,
            )
        case ValuesTableSource():
            replaced_values = [
                tuple([replacement(val) for val in row])
                for row in table_source.rows
            ]
            return ValuesTableSource(
                replaced_values,
                alias=table_source.table.identifier(),
                columns=table_source.cols,
            )
        case FunctionTableSource():
            replaced_function = replacement(table_source.function)
            return FunctionTableSource(
                replaced_function, alias=table_source.target_table
            )
        case _:
            raise TypeError("Unknown table source type: " + str(table_source))


def _replace_expressions_in_clause(
    clause: Optional[ClauseType],
    replacement: Callable[[SqlExpression], SqlExpression],
) -> Optional[ClauseType]:
    """Handler to update all expressions in a clause.

    This method does not perform any sanity checks on the updated clauses.

    Parameters
    ----------
    clause : ClauseType
        The clause to update. Can be ``None``, in which case no replacement is performed.
    replacement : Callable[[SqlExpression], SqlExpression]
        A function mapping each expression to a (potentially updated) expression

    Returns
    -------
    Optional[ClauseType]
        The updated clause. Can be ``None``, if `clause` already was.

    Raises
    ------
    ValueError
        If the clause is of no known type. This indicates that this method is missing a handler for a specific clause type that
        was added later on.
    """
    if not clause:
        return None

    if isinstance(clause, Hint) or isinstance(clause, Explain):
        return clause
    elif isinstance(clause, CommonTableExpression):
        replaced_queries: list[WithQuery] = []

        for cte in clause.queries:
            if isinstance(cte, ValuesWithQuery):
                replaced_values = [
                    tuple([replacement(val) for val in row]) for row in cte.rows
                ]
                replaced_cte = ValuesWithQuery(
                    replaced_values,
                    target_name=cte.target_table,
                    columns=cte.cols,
                    materialized=cte.materialized,
                )
                replaced_queries.append(replaced_cte)
                continue

            replaced_cte = WithQuery(
                replace_expressions(cte.query, replacement),
                cte.target_table,
                materialized=cte.materialized,
            )
            replaced_queries.append(replaced_cte)

        return CommonTableExpression(
            replaced_queries, recursive=clause.recursive
        )
    elif isinstance(clause, Select):
        replaced_targets = [
            BaseProjection(replacement(proj.expression), proj.target_name)
            for proj in clause.targets
        ]
        return Select(replaced_targets, distinct=clause.distinct_specifier())
    elif isinstance(clause, ImplicitFromClause):
        return clause
    elif isinstance(clause, ExplicitFromClause):
        replaced_joins = [
            _replace_expression_in_table_source(join, replacement)
            for join in clause.items
        ]
        return ExplicitFromClause(replaced_joins)
    elif isinstance(clause, From):
        replaced_contents = [
            _replace_expression_in_table_source(target, replacement)
            for target in clause.items
        ]
        return From(replaced_contents)
    elif isinstance(clause, Where):
        return Where(
            _replace_expression_in_predicate(clause.predicate, replacement)
        )
    elif isinstance(clause, GroupBy):
        replaced_cols = [replacement(col) for col in clause.group_columns]
        return GroupBy(replaced_cols, clause.distinct)
    elif isinstance(clause, Having):
        return Having(
            _replace_expression_in_predicate(clause.condition, replacement)
        )
    elif isinstance(clause, OrderBy):
        replaced_cols = [
            OrderByExpression(
                replacement(col.column), col.ascending, col.nulls_first
            )
            for col in clause.expressions
        ]
        return OrderBy(replaced_cols)
    elif isinstance(clause, Limit):
        return clause
    elif isinstance(clause, UnionClause):
        replaced_left = replace_expressions(clause.left_query, replacement)
        replaced_right = replace_expressions(clause.right_query, replacement)
        return UnionClause(
            replaced_left, replaced_right, union_all=clause.union_all
        )
    elif isinstance(clause, IntersectClause):
        replaced_left = replace_expressions(clause.left_query, replacement)
        replaced_right = replace_expressions(clause.right_query, replacement)
        return IntersectClause(replaced_left, replaced_right)
    elif isinstance(clause, ExceptClause):
        replaced_left = replace_expressions(clause.left_query, replacement)
        replaced_right = replace_expressions(clause.right_query, replacement)
        return ExceptClause(replaced_left, replaced_right)
    else:
        raise ValueError("Unknown clause: " + str(clause))


def replace_expressions(
    query: SelectQueryType,
    replacement: Callable[[SqlExpression], SqlExpression],
) -> SelectQueryType:
    """Updates all expressions in a query.

    The replacement handler can either produce entirely new expressions, or simply return the current expression instance if
    no update should be performed. Be very careful with this method since no sanity checks are performed, other than the rules
    of `build_query`.

    Parameters
    ----------
    query : SelectQueryType
        The query to update
    replacement : Callable[[SqlExpression], SqlExpression]
        A function mapping each of the current expressions in the `query` to potentially updated expressions.

    Returns
    -------
    SelectQueryType
        The updated query
    """
    replaced_clauses = [
        _replace_expressions_in_clause(clause, replacement)
        for clause in query.clauses()
    ]
    return build_query(replaced_clauses)


def _perform_predicate_replacement(
    current_predicate: AbstractPredicate,
    target_predicate: AbstractPredicate,
    new_predicate: AbstractPredicate,
) -> AbstractPredicate:
    """Handler to change specific predicates in a predicate hierarchy to other predicates.

    This does not perform any sanity checks on the updated predicate hierarchy, nor is the hierarchy simplified.

    Parameters
    ----------
    current_predicate : AbstractPredicate
        The predicate hierarchy in which the updates should occur
    target_predicate : AbstractPredicate
        The predicate that should be replaced
    new_predicate : AbstractPredicate
        The new predicate that should be used instead of the `target_predicate`

    Returns
    -------
    AbstractPredicate
        The updated predicate
    """
    if current_predicate == target_predicate:
        return new_predicate

    if isinstance(current_predicate, CompoundPredicate):
        if current_predicate.operation == CompoundOperator.Not:
            replaced_children = [
                _perform_predicate_replacement(
                    current_predicate.children, target_predicate, new_predicate
                )
            ]
        else:
            replaced_children = [
                _perform_predicate_replacement(
                    child_pred, target_predicate, new_predicate
                )
                for child_pred in current_predicate.children
            ]
        return CompoundPredicate(current_predicate.operation, replaced_children)
    else:
        return current_predicate


def replace_predicate(
    query: ImplicitSqlQuery,
    predicate_to_replace: AbstractPredicate,
    new_predicate: AbstractPredicate,
) -> ImplicitSqlQuery:
    """Rewrites a specific query to use a new predicate in place of an old one.

    In the current implementation this does only work for top-level predicates, i.e. subqueries and CTEs are not considered.
    Furthermore, only the ``WHERE`` clause and the ``HAVING`` clause are modified, since these should be the only ones that
    contain predicates.

    If the predicate to replace is not found, nothing happens. In the same vein, no sanity checks are performed on the updated
    query.

    Parameters
    ----------
    query : ImplicitSqlQuery
        The query update
    predicate_to_replace : AbstractPredicate
        The old predicate that should be dropped
    new_predicate : AbstractPredicate
        The predicate that should be used in place of `predicate_to_replace`. This can be an entirely different type of
        predicate, e.g. a conjunction of join conditions that replace a single join predicate.

    Returns
    -------
    ImplicitSqlQuery
        The updated query
    """
    # TODO: also allow replacement in explicit SQL queries
    # TODO: allow predicate replacement in subqueries / CTEs
    # TODO: allow replacement in set queries
    if not query.where_clause and not query.having_clause:
        return query

    if query.where_clause:
        replaced_predicate = _perform_predicate_replacement(
            query.where_clause.predicate, predicate_to_replace, new_predicate
        )
        replaced_where = Where(replaced_predicate)
    else:
        replaced_where = None

    if query.having_clause:
        replaced_predicate = _perform_predicate_replacement(
            query.having_clause.condition, predicate_to_replace, new_predicate
        )
        replaced_having = Having(replaced_predicate)
    else:
        replaced_having = None

    candidate_clauses = [replaced_where, replaced_having]
    return replace_clause(
        query, [clause for clause in candidate_clauses if clause]
    )


def rename_columns_in_query[T: SelectStatement](
    query: T, available_renamings: dict[ColumnReference, ColumnReference]
) -> T:
    """Replaces specific column references by new references for an entire query.

    Parameters
    ----------
    query : SelectQueryType
        The query to update
    available_renamings : dict[ColumnReference, ColumnReference]
        A dictionary mapping each of the old column values to the values that should be used instead.

    Returns
    -------
    SelectQueryType
        The updated query

    Raises
    ------
    TypeError
        If the query is of no known type. This indicates that this method is missing a handler for a specific query type that
        was added later on.
    """
    renamed_cte = rename_columns_in_clause(
        query.cte_clause, available_renamings
    )
    renamed_having = rename_columns_in_clause(
        query.having_clause, available_renamings
    )
    renamed_orderby = rename_columns_in_clause(
        query.orderby_clause, available_renamings
    )

    if isinstance(query, SetQuery):
        renamed_left = rename_columns_in_query(
            query.left_query, available_renamings
        )
        renamed_right = rename_columns_in_query(
            query.right_query, available_renamings
        )
        return SetQuery(
            renamed_left,
            renamed_right,
            query.set_operation,
            cte_clause=renamed_cte,
            orderby_clause=renamed_orderby,
            limit_clause=query.limit_clause,
            hints=query.hints,
            explain_clause=query.explain,
        )

    renamed_select = rename_columns_in_clause(
        query.select_clause, available_renamings
    )
    renamed_from = rename_columns_in_clause(
        query.from_clause, available_renamings
    )
    renamed_where = rename_columns_in_clause(
        query.where_clause, available_renamings
    )
    renamed_groupby = rename_columns_in_clause(
        query.groupby_clause, available_renamings
    )

    if isinstance(query, ImplicitSqlQuery):
        return ImplicitSqlQuery(
            select_clause=renamed_select,
            from_clause=renamed_from,
            where_clause=renamed_where,
            groupby_clause=renamed_groupby,
            having_clause=renamed_having,
            orderby_clause=renamed_orderby,
            limit_clause=query.limit_clause,
            cte_clause=renamed_cte,
            hints=query.hints,
            explain_clause=query.explain,
        )
    elif isinstance(query, ExplicitSqlQuery):
        return ExplicitSqlQuery(
            select_clause=renamed_select,
            from_clause=renamed_from,
            where_clause=renamed_where,
            groupby_clause=renamed_groupby,
            having_clause=renamed_having,
            orderby_clause=renamed_orderby,
            limit_clause=query.limit_clause,
            cte_clause=renamed_cte,
            hints=query.hints,
            explain_clause=query.explain,
        )
    elif isinstance(query, MixedSqlQuery):
        return MixedSqlQuery(
            select_clause=renamed_select,
            from_clause=renamed_from,
            where_clause=renamed_where,
            groupby_clause=renamed_groupby,
            having_clause=renamed_having,
            orderby_clause=renamed_orderby,
            limit_clause=query.limit_clause,
            cte_clause=renamed_cte,
            hints=query.hints,
            explain_clause=query.explain,
        )
    else:
        raise TypeError("Unknown query type: " + str(query))


def rename_columns_in_expression(
    expression: Optional[SqlExpression],
    available_renamings: dict[ColumnReference, ColumnReference],
) -> Optional[SqlExpression]:
    """Replaces references to specific columns in an expression.

    Parameters
    ----------
    expression : Optional[SqlExpression]
        The expression to update. If ``None``, no renaming is performed.
    available_renamings : dict[ColumnReference, ColumnReference]
        A dictionary mapping each of the old column values to the values that should be used instead.

    Returns
    -------
    Optional[SqlExpression]
        The updated expression. Can be ``None``, if `expression` already was.

    Raises
    ------
    ValueError
        If the expression is of no known type. This indicates that this method is missing a handler for a specific expressoin
        type that was added later on.
    """
    if expression is None:
        return None

    if isinstance(expression, StaticValueExpression) or isinstance(
        expression, StarExpression
    ):
        return expression
    elif isinstance(expression, ColumnExpression):
        return (
            ColumnExpression(available_renamings[expression.column])
            if expression.column in available_renamings
            else expression
        )
    elif isinstance(expression, AbstractPredicate):
        return rename_columns_in_predicate(expression, available_renamings)
    elif isinstance(expression, CastExpression):
        renamed_child = rename_columns_in_expression(
            expression.casted_expression, available_renamings
        )
        renamed_params = (
            [
                rename_columns_in_expression(param, available_renamings)
                for param in expression.type_params
            ]
            if expression.type_params
            else None
        )
        return CastExpression(
            renamed_child,
            expression.target_type,
            type_params=renamed_params,
            array_type=expression.array_type,
        )
    elif isinstance(expression, MathExpression):
        renamed_first_arg = rename_columns_in_expression(
            expression.first_arg, available_renamings
        )
        renamed_second_arg = rename_columns_in_expression(
            expression.second_arg, available_renamings
        )
        return MathExpression(
            expression.operator, renamed_first_arg, renamed_second_arg
        )
    elif isinstance(expression, ArrayAccessExpression):
        # NB: ArrayAccessExpression needs to be checked before FunctionExpression, since it is a subclass of it

        renamed_array = rename_columns_in_expression(
            expression.array, available_renamings
        )
        renamed_idx = (
            rename_columns_in_expression(expression.index, available_renamings)
            if expression.index
            else None
        )
        renamed_lower = (
            rename_columns_in_expression(
                expression.lower_index, available_renamings
            )
            if expression.lower_index
            else None
        )
        renamed_upper = (
            rename_columns_in_expression(
                expression.upper_index, available_renamings
            )
            if expression.upper_index
            else None
        )
        return ArrayAccessExpression(
            renamed_array,
            index=renamed_idx,
            lower_index=renamed_lower,
            upper_index=renamed_upper,
        )
    elif isinstance(expression, FunctionExpression):
        renamed_arguments = [
            rename_columns_in_expression(arg, available_renamings)
            for arg in expression.arguments
        ]
        renamed_kwargs = {
            kw: rename_columns_in_expression(arg, available_renamings)
            for kw, arg in expression.keyword_args.items()
        }
        renamed_filter = (
            rename_columns_in_predicate(
                expression.filter_where, available_renamings
            )
            if expression.filter_where
            else None
        )
        return FunctionExpression(
            expression.function,
            renamed_arguments,
            keyword_args=renamed_kwargs,
            distinct=expression.distinct,
            filter_where=renamed_filter,
        )
    elif isinstance(expression, SubqueryExpression):
        return SubqueryExpression(
            rename_columns_in_query(expression.query, available_renamings)
        )
    elif isinstance(expression, WindowExpression):
        renamed_function = rename_columns_in_expression(
            expression.window_function, available_renamings
        )
        renamed_partition = [
            rename_columns_in_expression(part, available_renamings)
            for part in expression.partitioning
        ]
        renamed_orderby = (
            rename_columns_in_clause(expression.ordering, available_renamings)
            if expression.ordering
            else None
        )
        renamed_filter = (
            rename_columns_in_predicate(
                expression.filter_condition, available_renamings
            )
            if expression.filter_condition
            else None
        )
        return WindowExpression(
            renamed_function,
            partitioning=renamed_partition,
            ordering=renamed_orderby,
            filter_condition=renamed_filter,
        )
    elif isinstance(expression, CaseExpression):
        renamed_cases = [
            (
                rename_columns_in_predicate(condition, available_renamings),
                rename_columns_in_expression(result, available_renamings),
            )
            for condition, result in expression.cases
        ]
        renamed_simple = (
            rename_columns_in_expression(
                expression.simple_expression, available_renamings
            )
            if expression.simple_expression
            else None
        )
        renamed_else = (
            rename_columns_in_expression(
                expression.else_expression, available_renamings
            )
            if expression.else_expression
            else None
        )
        return CaseExpression(
            renamed_cases, simple_expr=renamed_simple, else_expr=renamed_else
        )
    elif isinstance(expression, QuantifierExpression):
        renamed_child = rename_columns_in_expression(
            expression.expression, available_renamings
        )
        return QuantifierExpression(
            renamed_child, quantifier=expression.quantifier
        )
    elif isinstance(expression, ArrayExpression):
        renamed_elements = [
            rename_columns_in_expression(elem, available_renamings)
            for elem in expression.elements
        ]
        return ArrayExpression(renamed_elements)
    else:
        raise ValueError("Unknown expression type: " + str(expression))


def _rename_columns_in_expression(
    expression: Optional[SqlExpression],
    available_renamings: dict[ColumnReference, ColumnReference],
) -> Optional[SqlExpression]:
    """See `rename_columns_in_expression` for details.

    See Also
    --------
    rename_columns_in_expression
    """
    warnings.warn(
        "This method is deprecated and will be removed in the future. Use `rename_columns_in_expression` instead.",
        FutureWarning,
    )
    return rename_columns_in_expression(expression, available_renamings)


def rename_columns_in_predicate(
    predicate: Optional[AbstractPredicate],
    available_renamings: dict[ColumnReference, ColumnReference],
) -> Optional[AbstractPredicate]:
    """Replaces all references to specific columns in a predicate by new references.

    Parameters
    ----------
    predicate : Optional[AbstractPredicate]
        The predicate to update. Can be ``None``, in which case no update is performed.
    available_renamings : dict[ColumnReference, ColumnReference]
        A dictionary mapping each of the old column values to the values that should be used instead.

    Returns
    -------
    Optional[AbstractPredicate]
        The updated predicate. Can be ``None``, if `predicate` already was.

    Raises
    ------
    ValueError
        If the query is of no known type. This indicates that this method is missing a handler for a specific query type that
        was added later on.
    """
    if not predicate:
        return None

    if isinstance(predicate, BinaryPredicate):
        renamed_first_arg = rename_columns_in_expression(
            predicate.first_argument, available_renamings
        )
        renamed_second_arg = rename_columns_in_expression(
            predicate.second_argument, available_renamings
        )
        return BinaryPredicate(
            predicate.operation, renamed_first_arg, renamed_second_arg
        )
    elif isinstance(predicate, BetweenPredicate):
        renamed_col = rename_columns_in_expression(
            predicate.column, available_renamings
        )
        renamed_interval_start = rename_columns_in_expression(
            predicate.interval_start, available_renamings
        )
        renamed_interval_end = rename_columns_in_expression(
            predicate.interval_end, available_renamings
        )
        return BetweenPredicate(
            renamed_col, (renamed_interval_start, renamed_interval_end)
        )
    elif isinstance(predicate, InPredicate):
        renamed_col = rename_columns_in_expression(
            predicate.column, available_renamings
        )
        renamed_vals = [
            rename_columns_in_expression(val, available_renamings)
            for val in predicate.values
        ]
        return InPredicate(renamed_col, renamed_vals)
    elif isinstance(predicate, UnaryPredicate):
        return UnaryPredicate(
            rename_columns_in_expression(predicate.column, available_renamings),
            predicate.operation,
        )
    elif isinstance(predicate, CompoundPredicate):
        renamed_children = (
            [
                rename_columns_in_predicate(
                    predicate.children, available_renamings
                )
            ]
            if predicate.operation == CompoundOperator.Not
            else [
                rename_columns_in_predicate(child, available_renamings)
                for child in predicate.children
            ]
        )
        return CompoundPredicate(predicate.operation, renamed_children)
    else:
        raise ValueError("Unknown predicate type: " + str(predicate))


def _rename_columns_in_table_source(
    table_source: TableSource,
    available_renamings: dict[ColumnReference, ColumnReference],
) -> Optional[TableSource]:
    """Handler method to replace all references to specific columns by new columns.

    Parameters
    ----------
    table_source : TableSource
        The source that should be updated
    available_renamings : dict[ColumnReference, ColumnReference]
        A dictionary mapping each of the old column values to the values that should be used instead.

    Returns
    -------
    Optional[TableSource]
        The updated source. Can be ``None``, if `table_source` already was.

    Raises
    ------
    TypeError
        If the source is of no known type. This indicates that this method is missing a handler for a specific source type that
        was added later on.
    """
    if table_source is None:
        return None

    match table_source:
        case DirectTableSource():
            # no columns in a plain table reference, we are done here
            return table_source

        case SubqueryTableSource():
            renamed_subquery = rename_columns_in_query(
                table_source.query, available_renamings
            )
            return SubqueryTableSource(
                renamed_subquery,
                table_source.target_name,
                lateral=table_source.lateral,
            )

        case JoinTableSource():
            renamed_left = _rename_columns_in_table_source(
                table_source.left, available_renamings
            )
            renamed_right = _rename_columns_in_table_source(
                table_source.right, available_renamings
            )
            renamed_condition = rename_columns_in_predicate(
                table_source.join_condition, available_renamings
            )
            return JoinTableSource(
                renamed_left,
                renamed_right,
                join_condition=renamed_condition,
                join_type=table_source.join_type,
            )

        case ValuesTableSource():
            if not any(
                col.belongs_to(table_source.table)
                for col in available_renamings
            ):
                return table_source

            for current_col, target_col in available_renamings.items():
                if not current_col.belongs_to(table_source.table):
                    continue
                if current_col.table != target_col.table:
                    raise ValueError(
                        "Cannot rename columns in a VALUES table source to a different table"
                    )

                # if we found a column that should be renamed, we need to replace the whole column specification
                # this process might be repeated multiple times, if multiple appropriate renamings exist
                current_col_spec = table_source.cols
                new_col_spec = [
                    (col if col.name != current_col.name else target_col.name)
                    for col in current_col_spec
                ]
                table_source = ValuesTableSource(
                    table_source.rows,
                    alias=table_source.table.identifier(),
                    columns=new_col_spec,
                )
            return table_source

        case FunctionTableSource():
            renamed_function = rename_columns_in_expression(
                table_source.function, available_renamings
            )
            return FunctionTableSource(
                renamed_function, alias=table_source.target_table
            )

        case _:
            raise TypeError("Unknown table source type: " + str(table_source))


def rename_columns_in_clause(
    clause: Optional[ClauseType],
    available_renamings: dict[ColumnReference, ColumnReference],
) -> Optional[ClauseType]:
    """Replaces all references to specific columns in a clause by new columns.

    Parameters
    ----------
    clause : Optional[ClauseType]
        The clause to update. Can be ``None``, in which case no update is performed.
    available_renamings : dict[ColumnReference, ColumnReference]
        A dictionary mapping each of the old column values to the values that should be used instead.

    Returns
    -------
    Optional[ClauseType]
        The updated clause. Can be ``None``, if `clause` already was.

    Raises
    ------
    ValueError
        If the clause is of no known type. This indicates that this method is missing a handler for a specific clause type that
        was added later on.
    """
    if not clause:
        return None

    if isinstance(clause, Hint) or isinstance(clause, Explain):
        return clause
    if isinstance(clause, CommonTableExpression):
        renamed_ctes: list[WithQuery] = []

        for cte in clause.queries:
            if not isinstance(cte, ValuesWithQuery):
                new_query = rename_columns_in_query(
                    cte.query, available_renamings
                )
                renamed_ctes.append(
                    WithQuery(
                        new_query,
                        cte.target_table,
                        materialized=cte.materialized,
                    )
                )
                continue

            if not any(
                col.belongs_to(cte.target_table) for col in available_renamings
            ):
                continue

            renamed_cte = cte
            for current_col, target_col in available_renamings.items():
                if not current_col.belongs_to(cte.target_table):
                    continue
                if current_col.table != target_col.table:
                    raise ValueError(
                        "Cannot rename columns in a VALUES table source to a different table"
                    )

                # if we found a column that should be renamed, we need to replace the whole column specification
                # this process might be repeated multiple times, if multiple appropriate renamings exist
                current_col_spec = cte.cols
                new_col_spec = [
                    (col if col.name != current_col.name else target_col.name)
                    for col in current_col_spec
                ]
                renamed_cte = ValuesWithQuery(
                    cte.rows,
                    target_name=cte.target_table,
                    columns=new_col_spec,
                    materialized=cte.materialized,
                )

            renamed_ctes.append(renamed_cte)

        return CommonTableExpression(renamed_ctes, recursive=clause.recursive)
    if isinstance(clause, Select):
        renamed_targets = [
            BaseProjection(
                rename_columns_in_expression(
                    proj.expression, available_renamings
                ),
                proj.target_name,
            )
            for proj in clause.targets
        ]
        return Select(renamed_targets, distinct=clause.distinct_specifier())
    elif isinstance(clause, ImplicitFromClause):
        return clause
    elif isinstance(clause, ExplicitFromClause):
        renamed_joins = [
            _rename_columns_in_table_source(join, available_renamings)
            for join in clause.items
        ]
        return ExplicitFromClause(renamed_joins)
    elif isinstance(clause, From):
        renamed_sources = [
            _rename_columns_in_table_source(table_source, available_renamings)
            for table_source in clause.items
        ]
        return From(renamed_sources)
    elif isinstance(clause, Where):
        return Where(
            rename_columns_in_predicate(clause.predicate, available_renamings)
        )
    elif isinstance(clause, GroupBy):
        renamed_cols = [
            rename_columns_in_expression(col, available_renamings)
            for col in clause.group_columns
        ]
        return GroupBy(renamed_cols, clause.distinct)
    elif isinstance(clause, Having):
        return Having(
            rename_columns_in_predicate(clause.condition, available_renamings)
        )
    elif isinstance(clause, OrderBy):
        renamed_cols = [
            OrderByExpression(
                rename_columns_in_expression(col.column, available_renamings),
                col.ascending,
                col.nulls_first,
            )
            for col in clause.expressions
        ]
        return OrderBy(renamed_cols)
    elif isinstance(clause, Limit):
        return clause
    elif isinstance(clause, UnionClause):
        renamed_left = rename_columns_in_query(
            clause.left_query, available_renamings
        )
        renamed_right = rename_columns_in_query(
            clause.right_query, available_renamings
        )
        return UnionClause(
            renamed_left, renamed_right, union_all=clause.union_all
        )
    elif isinstance(clause, IntersectClause):
        renamed_left = rename_columns_in_query(
            clause.left_query, available_renamings
        )
        renamed_right = rename_columns_in_query(
            clause.right_query, available_renamings
        )
        return IntersectClause(renamed_left, renamed_right)
    elif isinstance(clause, ExceptClause):
        renamed_left = rename_columns_in_query(
            clause.left_query, available_renamings
        )
        renamed_right = rename_columns_in_query(
            clause.right_query, available_renamings
        )
        return ExceptClause(renamed_left, renamed_right)
    else:
        raise ValueError("Unknown clause: " + str(clause))


class _TableReferenceRenamer(
    ClauseVisitor[BaseClause],
    PredicateVisitor[AbstractPredicate],
    SqlExpressionVisitor[SqlExpression],
):
    """Visitor to replace all references to specific tables to refer to new tables instead.

    Parameters
    ----------
    renamings : Optional[dict[TableReference, TableReference]], optional
        Map from old table name to new name. If this is given, `source_table` and `target_table` are ignored.
    source_table : Optional[TableReference], optional
        Create a visitor for a single renaming operation. This parameter specifies the old table name that should be replaced.
        This parameter is ignored if `renamings` is given.
    target_table : Optional[TableReference], optional
        Create a visitor for a single renaming operation. This parameter specifies the new table name that should be used
        instead of `source_table`. This parameter is ignored if `renamings` is given.
    """

    def __init__(
        self,
        renamings: dict[TableReference, TableReference],
    ) -> None:
        self._renamings = renamings
        self._renaming_targets = set(
            tab.identifier() for tab in self._renamings.values()
        )

    def visit_hint_clause(self, clause: Hint, *args, **kwargs) -> Hint:
        return clause

    def visit_explain_clause(self, clause: Explain, *args, **kwargs) -> Explain:
        return clause

    def visit_cte_clause(
        self, clause: CommonTableExpression, *args, **kwargs
    ) -> CommonTableExpression:
        ctes: list[WithQuery] = []

        for cte in clause.queries:
            nested_renamings = cte.query.accept_visitor(self)
            nested_query = build_query(nested_renamings.values())
            target_table = self._rename_table(cte.target_table)

            ctes.append(
                WithQuery(
                    nested_query, target_table, materialized=cte.materialized
                )
            )

        return CommonTableExpression(ctes, recursive=clause.recursive)

    def visit_select_clause(self, clause, *args, **kwargs) -> Select:
        projections: list[BaseProjection] = []

        for proj in clause:
            renamed_expression = proj.expression.accept_visitor(self)
            projections.append(
                BaseProjection(renamed_expression, proj.target_name)
            )

        return Select(projections, distinct=clause.distinct_specifier())

    def visit_from_clause(self, clause: From, *args, **kwargs) -> From:
        used_identifieres = set(tab.identifier() for tab in clause.tables())
        intersect = used_identifieres & self._renaming_targets
        match clause:
            case ImplicitFromClause(tables):
                renamed_tables = [
                    self._rename_table_source(src) for src in tables
                ]
                if not intersect:
                    return ImplicitFromClause(renamed_tables)

                separate_tables = used_identifieres - intersect
                final_tables: list[DirectTableSource] = []
                for tab in renamed_tables:
                    if tab.identifier() in separate_tables:
                        final_tables.append(tab)
                        continue
                    if tab.identifier() not in intersect:
                        continue
                    final_tables.append(tab)
                    intersect.remove(tab.identifier())
                return ImplicitFromClause(final_tables)

            case ExplicitFromClause(join):
                if intersect:
                    raise ValueError(
                        "Cannot rename explicit JOINs: "
                        f"Renaming targets {intersect} is already present in the query. "
                        "This is currently not supported."
                    )
                renamed_join = self._rename_table_source(join)
                return ExplicitFromClause(renamed_join)

            case From(items):
                if intersect:
                    raise ValueError(
                        "Cannot rename explicit JOINs: "
                        f"Renaming targets {intersect} is already present in the query. "
                        "This is currently not supported."
                    )
                renamed_items = [
                    self._rename_table_source(item) for item in items
                ]
                return From(renamed_items)

            case _:
                raise ValueError("Unknown from clause type: " + str(clause))

    def visit_where_clause(self, clause: Where, *args, **kwargs) -> Where:
        renamed_predicate = clause.predicate.accept_visitor(self)
        return Where(renamed_predicate)

    def visit_groupby_clause(self, clause: GroupBy, *args, **kwargs) -> GroupBy:
        renamed_groupings = [
            grouping.accept_visitor(self) for grouping in clause.group_columns
        ]
        return GroupBy(renamed_groupings, clause.distinct)

    def visit_having_clause(self, clause: Having, *args, **kwargs) -> Having:
        renamed_predicate = clause.condition.accept_visitor(self)
        return Having(renamed_predicate)

    def visit_orderby_clause(self, clause: OrderBy, *args, **kwargs) -> OrderBy:
        renamed_orderings: list[OrderByExpression] = []

        for ordering in clause:
            renamed_expression = ordering.column.accept_visitor(self)
            renamed_orderings.append(
                OrderByExpression(
                    renamed_expression, ordering.ascending, ordering.nulls_first
                )
            )

        return renamed_orderings

    def visit_limit_clause(self, clause: Limit, *args, **kwargs) -> Limit:
        return clause

    def visit_union_clause(
        self, clause: UnionClause, *args, **kwargs
    ) -> UnionClause:
        renamed_lhs = clause.left_query.accept_visitor(self)
        renamed_rhs = clause.right_query.accept_visitor(self)
        return UnionClause(renamed_lhs, renamed_rhs, union_all=clause.union_all)

    def visit_except_clause(
        self, clause: ExceptClause, *args, **kwargs
    ) -> ExceptClause:
        renamed_lhs = clause.left_query.accept_visitor(self)
        renamed_rhs = clause.right_query.accept_visitor(self)
        return ExceptClause(renamed_lhs, renamed_rhs)

    def visit_intersect_clause(
        self, clause: IntersectClause, *args, **kwargs
    ) -> IntersectClause:
        renamed_lhs = clause.left_query.accept_visitor(self)
        renamed_rhs = clause.right_query.accept_visitor(self)
        return IntersectClause(renamed_lhs, renamed_rhs)

    def visit_binary_predicate(
        self, predicate: BinaryPredicate, *args, **kwargs
    ) -> BinaryPredicate:
        renamed_lhs = predicate.first_argument.accept_visitor(self)
        renamed_rhs = predicate.second_argument.accept_visitor(self)
        return BinaryPredicate(predicate.operation, renamed_lhs, renamed_rhs)

    def visit_between_predicate(
        self, predicate: BetweenPredicate, *args, **kwargs
    ) -> BetweenPredicate:
        renamed_col = predicate.column.accept_visitor(self)
        renamed_start = predicate.interval_start.accept_visitor(self)
        renamed_end = predicate.interval_end.accept_visitor(self)
        return BetweenPredicate(renamed_col, (renamed_start, renamed_end))

    def visit_in_predicate(
        self, predicate: InPredicate, *args, **kwargs
    ) -> InPredicate:
        renamed_col = predicate.column.accept_visitor(self)
        renamed_vals = [val.accept_visitor(self) for val in predicate.values]
        return InPredicate(renamed_col, renamed_vals)

    def visit_unary_predicate(
        self, predicate: UnaryPredicate, *args, **kwargs
    ) -> UnaryPredicate:
        renamed_col = predicate.column.accept_visitor(self)
        return UnaryPredicate(renamed_col, predicate.operation)

    def visit_not_predicate(
        self,
        predicate: CompoundPredicate,
        child_predicate: AbstractPredicate,
        *args,
        **kwargs,
    ) -> CompoundPredicate:
        renamed_child = child_predicate.accept_visitor(self)
        return CompoundPredicate(CompoundOperator.Not, [renamed_child])

    def visit_or_predicate(
        self,
        predicate: CompoundPredicate,
        components: Sequence[AbstractPredicate],
        *args,
        **kwargs,
    ) -> CompoundPredicate:
        renamed_children = [child.accept_visitor(self) for child in components]
        return CompoundPredicate(CompoundOperator.Or, renamed_children)

    def visit_and_predicate(
        self,
        predicate: CompoundPredicate,
        components: Sequence[AbstractPredicate],
        *args,
        **kwargs,
    ) -> CompoundPredicate:
        renamed_children = [child.accept_visitor(self) for child in components]
        return CompoundPredicate(CompoundOperator.And, renamed_children)

    def visit_static_value_expr(
        self, expr: StaticValueExpression, *args, **kwargs
    ) -> StaticValueExpression:
        return expr

    def visit_cast_expr(
        self, expr: CastExpression, *args, **kwargs
    ) -> CastExpression:
        renamed_child = expr.casted_expression.accept_visitor(self)
        renamed_params = (
            [param.accept_visitor(self) for param in expr.type_params]
            if expr.type_params
            else None
        )
        return CastExpression(
            renamed_child,
            expr.target_type,
            type_params=renamed_params,
            array_type=expr.array_type,
        )

    def visit_math_expr(
        self, expr: MathExpression, *args, **kwargs
    ) -> MathExpression:
        renamed_lhs = expr.first_arg.accept_visitor(self)

        if isinstance(expr.second_arg, SqlExpression):
            renamed_rhs = expr.second_arg.accept_visitor(self)
        elif expr.second_arg is not None:
            renamed_rhs = [
                nested_expr.accept_visitor(self)
                for nested_expr in expr.second_arg
            ]
        else:
            renamed_rhs = None

        return MathExpression(expr.operator, renamed_lhs, renamed_rhs)

    def visit_column_expr(
        self, expr: ColumnExpression, *args, **kwargs
    ) -> ColumnExpression:
        col = self._rename_column(expr.column)
        return ColumnExpression(col)

    def visit_function_expr(
        self, expr: FunctionExpression, *args, **kwargs
    ) -> FunctionExpression:
        renamed_args = [arg.accept_visitor(self) for arg in expr.arguments]
        renamed_kwargs = {
            kw: arg.accept_visitor(self)
            for kw, arg in expr.keyword_args.items()
        }
        renamed_filter = (
            expr.filter_where.accept_visitor(self)
            if expr.filter_where
            else None
        )
        return FunctionExpression(
            expr.function,
            renamed_args,
            keyword_args=renamed_kwargs,
            distinct=expr.distinct,
            filter_where=renamed_filter,
        )

    def visit_subquery_expr(
        self, expr: SubqueryExpression, *args, **kwargs
    ) -> SubqueryExpression:
        subquery_comps = expr.query.accept_visitor(self)
        renamed_subquery = build_query(subquery_comps.values())
        return SubqueryExpression(renamed_subquery)

    def visit_star_expr(
        self, expr: StarExpression, *args, **kwargs
    ) -> StarExpression:
        renamed_table = (
            self._rename_table(expr.from_table) if expr.from_table else None
        )
        return StarExpression(from_table=renamed_table)

    def visit_window_expr(
        self, expr: WindowExpression, *args, **kwargs
    ) -> WindowExpression:
        renamed_window_func = expr.window_function.accept_visitor(self)
        renamed_partition = [
            part.accept_visitor(self) for part in expr.partitioning
        ]
        renamed_ordering = (
            expr.ordering.accept_visitor(self) if expr.ordering else None
        )
        renamed_filter = (
            expr.filter_condition.accept_visitor(self)
            if expr.filter_condition
            else None
        )
        return WindowExpression(
            renamed_window_func,
            partitioning=renamed_partition,
            ordering=renamed_ordering,
            filter_condition=renamed_filter,
        )

    def visit_case_expr(
        self, expr: CaseExpression, *args, **kwargs
    ) -> CaseExpression:
        renamed_cases: list[tuple[AbstractPredicate, SqlExpression]] = []

        for condition, value in expr.cases:
            renamed_condition = condition.accept_visitor(self)
            renamed_value = value.accept_visitor(self)
            renamed_cases.append((renamed_condition, renamed_value))

        renamed_simple = (
            expr.simple_expression.accept_visitor(self)
            if expr.simple_expression
            else None
        )

        renamed_default = (
            expr.else_expression.accept_visitor(self)
            if expr.else_expression
            else None
        )
        return CaseExpression(
            renamed_cases, simple_expr=renamed_simple, else_expr=renamed_default
        )

    def visit_quantifier_expr(
        self, expr: QuantifierExpression, *args, **kwargs
    ) -> QuantifierExpression:
        renamed_child = expr.expression.accept_visitor(self)
        return QuantifierExpression(expr.quantifier, quantifier=renamed_child)

    def visit_array_expr(
        self, expr: ArrayExpression, *args, **kwargs
    ) -> ArrayExpression:
        renamed_elems = [elem.accept_visitor(self) for elem in expr.elements]
        return ArrayExpression(renamed_elems)

    def visit_array_access_expr(
        self, expr: ArrayAccessExpression, *args, **kwargs
    ) -> ArrayAccessExpression:
        renamed_array = expr.array.accept_visitor(self)
        renamed_idx = (
            expr.lower_index.accept_visitor(self) if expr.index else None
        )
        renamed_lower = (
            expr.lower_index.accept_visitor(self) if expr.lower_index else None
        )
        renamed_upper = (
            expr.upper_index.accept_visitor(self) if expr.upper_index else None
        )
        return ArrayAccessExpression(
            renamed_array,
            idx=renamed_idx,
            lower_index=renamed_lower,
            upper_index=renamed_upper,
        )

    def visit_predicate_expr(
        self, expr: AbstractPredicate, *args, **kwargs
    ) -> AbstractPredicate:
        return expr.accept_visitor(self)

    def _rename_table_source[T: TableSource](self, source: T) -> T:
        """Helper method to traverse and rename (the contents of) an arbitrary *FROM* item."""
        match source:
            case DirectTableSource(tab):
                renamed_table = self._rename_table(tab)
                return DirectTableSource(renamed_table)

            case SubqueryTableSource(subquery, target_name, lateral):
                nested_renamings = subquery.accept_visitor(self)
                nested_query = build_query(nested_renamings.values())
                target_table = self._rename_table(target_name)
                return SubqueryTableSource(
                    nested_query, target_table, lateral=lateral
                )

            case JoinTableSource(lhs, rhs, join_condition, join_type):
                renamed_lhs = self._rename_table_source(lhs)
                renamed_rhs = self._rename_table_source(rhs)
                renamed_condition = (
                    join_condition.accept_visitor(self)
                    if join_condition
                    else None
                )
                return JoinTableSource(
                    renamed_lhs,
                    renamed_rhs,
                    join_condition=renamed_condition,
                    join_type=join_type,
                )

            case ValuesTableSource(rows, alias, columns):
                renamed_alias = self._rename_table(alias)
                return ValuesTableSource(
                    rows, alias=renamed_alias, columns=columns
                )

            case FunctionTableSource(function, alias):
                renamed_function = function.accept_visitor(self)
                renamed_alias = self._rename_table(alias) if alias else ""
                return FunctionTableSource(
                    renamed_function, alias=renamed_alias
                )

            case _:
                raise ValueError("Unknown table source type: " + str(source))

    @overload
    def _rename_table(self, table: TableReference) -> TableReference: ...

    @overload
    def _rename_table(self, table: str) -> str: ...

    def _rename_table(
        self, table: str | TableReference
    ) -> str | TableReference:
        """Helper method to rename a specific table reference independent of its specific representation."""
        if isinstance(table, TableReference):
            return self._renamings.get(table, table)

        return next(
            (
                target_tab.identifier()
                for orig_tab, target_tab in self._renamings.items()
                if orig_tab.identifier() == table
            ),
            table,
        )

    def _rename_column(self, column: ColumnReference) -> ColumnReference:
        """Helper method to rename a specific column reference."""
        if not ColumnReference.assert_bound(column):
            return column

        target_table = self._renamings.get(column.table)
        return column.bind_to(target_table) if target_table else column


@overload
def rename_table[T: SelectStatement](
    source_query: T,
    renamings: dict[TableReference, TableReference],
    *,
    prefix_column_names: bool = False,
) -> T: ...


@overload
def rename_table[T: SelectStatement](
    source_query: T,
    from_table: TableReference,
    target_table: TableReference,
    *,
    prefix_column_names: bool = False,
) -> T: ...


def rename_table[T: SelectStatement](
    source_query: T,
    from_table: TableReference | dict[TableReference, TableReference],
    target_table: TableReference | None = None,
    *,
    prefix_column_names: bool = False,
) -> T:
    """Changes all references to a specific table to refer to another table instead.

    Parameters
    ----------
    source_query : SelectQueryType
        The query that should be updated
    from_table : TableReference
        The table that should be replaced
    target_table : TableReference
        The table that should be used instead
    prefix_column_names : bool, optional
        Whether a prefix should be added to column names. If this is ``True``, column references will be changed in two ways:

        1. if they belonged to the `from_table`, they will now belong to the `target_table` after the renaming
        2. The column names will be changed to include the identifier of the `from_table` as a prefix.

    Returns
    -------
    SelectQueryType
        The updated query
    """
    if isinstance(from_table, TableReference) and target_table is None:
        raise ValueError(
            "Renamings must be specified as a dict, or both from_table and target_table must be set!"
        )
    elif isinstance(from_table, TableReference):
        renamings = {from_table: target_table}
    else:
        renamings = from_table

    tabs_to_rename = set(tab for tab in renamings.keys())

    # Despite the convenient _TableReferenceRenamer, we still need to do a little bit of manual gathering/traversal to support
    # column prefixes.
    necessary_renamings: dict[ColumnReference, ColumnReference] = {}
    for column in filter(
        lambda c: c.table in tabs_to_rename, source_query.columns()
    ):
        new_column_name = (
            f"{column.table.identifier()}_{column.name}"
            if prefix_column_names
            else column.name
        )
        necessary_renamings[column] = ColumnReference(
            new_column_name, target_table
        )

    renamed_cols = rename_columns_in_query(source_query, necessary_renamings)

    tab_renamer = _TableReferenceRenamer(renamings)
    renamed_clauses = renamed_cols.accept_visitor(tab_renamer)

    return build_query(renamed_clauses.values())


def merge_tables[T: SelectStatement](
    query: T, tables: Iterable[TableReference], *, target: TableReference
) -> T:
    """Rewrites a query to replace all references to any of the given tables by references to a single target table.

    This is useful, for example, for mat view scenarios where queries should re-written after a materialized view has been
    created over a specific join.

    Examples
    --------
    >>> stats = pb.workloads.stats()
    >>> posts = pb.TableReference("posts", "p")
    >>> comments = pb.TableReference("comments", "c")
    >>> mat_view = pb.TableReference("post_comments", "pc")
    >>> pb.transform.merge_tables(stats["q-10"], [posts, comments], target=mat_view)
    """

    # The issue of merging tables is a bit more complicated than just calling rename_tables()
    # because we also need to update the FROM clause to not contain any duplicate entries after
    # the fact. Therefore, we perform the required steps manually here.

    tables = set(tables)
    cols_to_rename: dict[ColumnReference, ColumnReference] = {
        col: col.bind_to(target)
        for col in query.columns()
        if col.table in tables
    }
    merged = rename_columns_in_query(query, cols_to_rename)
    for tab in tables:
        merged = rename_table(merged, from_table=tab, target_table=target)
    return merged
