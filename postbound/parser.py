"""The parser constructs `SqlQuery` objects from query strings.

Other than the parsing itself, the process will also execute a basic column binding process. For example, consider
a query like *SELECT \\* FROM R WHERE R.a = 42*. In this case, the binding only affects the column reference *R.a*
and sets the table of that column to *R*. This binding based on column and table names is always performed.

If the table cannot be inferred based on the column name (e.g. for a query like *SELECT * FROM R, S WHERE a = 42*), a
second binding phase can be executed. This binding needs a working database connection and queries the database schema
to detect the correct tables for each column. Whether the second phase should also be executed by default can be
configured system-wide by setting the `auto_bind_columns` variable.

Notes
-----
Please beware that SQL parsing is a very challenging undertaking and there might be bugs in some lesser-used features.
If you encounter any issues, please report them on the GitHub issue tracker.
We test the parser based on some popular benchmarks, namely JOB and Stats to ensure that result sets from the raw SQL queries
match result sets from the parsed queries. However, we cannot guarantee that the parser will work for all SQL queries.

The parsing itself is based on the pglast project that implements a SQL -> JSON/dict conversion, based on the actual Postgres
query parser. Our parser implementation takes such a JSON representation as input and generates the more verbose structures of
the qal. There exists a Jupyter notebook called *PglastParsingTests* in the *tests* directory that shows the output emitted by
pglast for different SQL query features.

References
----------

.. pglast project: https://github.com/lelit/pglast Thanks a lot for maintaining this fantastic tool and the great support!
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Literal, Optional, overload

import pglast

from . import util
from ._core import ColumnReference, TableReference
from .db import DatabasePool, DatabaseSchema
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
    ColumnExpression,
    CommonTableExpression,
    CompoundOperator,
    CompoundPredicate,
    DirectTableSource,
    Explain,
    ExplicitFromClause,
    From,
    FunctionExpression,
    FunctionTableSource,
    GroupBy,
    Having,
    Hint,
    ImplicitFromClause,
    InPredicate,
    JoinTableSource,
    JoinType,
    Limit,
    LogicalOperator,
    MathExpression,
    MathOperator,
    OrderBy,
    OrderByExpression,
    QuantifierExpression,
    QuantifierOperator,
    Select,
    SelectStatement,
    SetOperator,
    SetQuery,
    SqlExpression,
    SqlOperator,
    SqlQuery,
    StarExpression,
    StaticValueExpression,
    SubqueryExpression,
    SubqueryTableSource,
    TableSource,
    UnaryPredicate,
    ValuesList,
    ValuesTableSource,
    ValuesWithQuery,
    Where,
    WindowExpression,
    WithQuery,
    build_query,
)

auto_bind_columns: bool = True
"""Indicates whether the parser should use the database catalog to obtain column bindings."""


class ParserWarning(UserWarning):
    """A warning that is raised during parsing, but does not prevent successful parsing."""

    pass


class ParserError(RuntimeError):
    """An error that is raised when parsing fails."""

    def __init__(self, msg: str) -> None:
        super().__init__(msg)


class SchemaCache:
    """A simple cache that stores the columns that belong to tables in our database schema.

    The cache only queries the actual catalog of the database system, if the requested table has not been cached, yet.

    Parameters
    ----------
    schema : Optional[DatabaseSchema]
        The schema to cache. If not provided, the cache cannot resolve column bindings.
    """

    def __init__(self, schema: Optional[DatabaseSchema] = None) -> None:
        self._schema = schema
        self._lookup_cache: dict[str, tuple[list[str], set[str]]] = {}

    def initialize_with(self, schema: Optional[DatabaseSchema]) -> None:
        """Sets the catalog if necessary"""
        if self._schema is not None and self._schema != schema:
            warnings.warn(
                "Parsing query for new schema. Dropping old schema cache."
            )
            self._schema = schema
            self._lookup_cache.clear()
        elif self._schema is not None:
            # same schema as before, do nothing
            return
        self._schema = schema

    def lookup_column(
        self, colname: str, candidate_tables: Iterable[TableReference]
    ) -> Optional[TableReference]:
        """Resolves the table that defines a specific column.

        If no catalog is available, this method will always return *None*.

        Returns
        -------
        Optional[TableReference]
            The table that defines the column. If there are multiple tables that could define the column, an arbitrary one
            is returned. If none of the candidates is the correct table, *None* is returned.
        """
        if not self._schema:
            return None

        for candidate in candidate_tables:
            if candidate.virtual:
                continue

            _, table_columns = self._inflate_cache(candidate)
            if colname in table_columns:
                return candidate

        return None

    def columns_of(self, table: str | TableReference) -> list[str]:
        """Provides the columns that belong to a specific table.

        If no catalog is available, this method will always return an empty list.
        """
        if not self._schema:
            return []

        cols, _ = self._inflate_cache(table)
        return cols

    def _inflate_cache(
        self, table: str | TableReference
    ) -> tuple[list[str], set[str]]:
        """Provides the columns that belong to a specific table, consulting the online catalog if necessary.

        This method assumes that there is indeed an online schema available. Calling this method without a schema will
        result in an arbitrary runtime error.

        Returns
        -------
        tuple[list[str], set[str]]
            The columns of the table in their defined order, as well as the same columns as a set.
        """
        table = table.full_name if isinstance(table, TableReference) else table
        cached_res = self._lookup_cache.get(table)
        if cached_res:
            return cached_res
        if not self._schema:
            return [], set()

        cols: list[str] = [col.name for col in self._schema.columns(table)]
        cols_set = set(cols)
        self._lookup_cache[table] = cols, cols_set
        return cols, cols_set


class QueryNamespace:
    """The query namespace acts as the central service to resolve column bindings in a query.

    It maintains a visibility map of all tables at a given point in the query and keeps track of the columns that form the
    result relation at the same points in time. This information is used to bind column references to the correct tables,
    including temporary virtual tables that alias existing physical columns.

    The namespace protocol works as follows:

    - While parsing a query, the table sources (CTEs and FROM entries) should be handled first. Each source should be
      registered in the namespace using the `register_table` method.
    - When a subquery or CTE is encoutered, the `open_nested` method has to be called to open a new local namespace and
      track the virtual table correctly.
    - Once all tables are registered, the parser can handle the *SELECT* clause. Afterwards, `determine_output_shape` has to
      called to compute all columns that are part of the result relation of the current namespace. This method takes care
      of resolving *SELECT \\** operations as necessary and requires that all input sources have already been registered and
      completely parsed, such that their output shapes are known.
    - While parsing the different clauses of the query, `lookup_column` and `resolve_table` can be used to determine the
      correct table references based on the sources that are currently available in the namespace.

    Each namespace can be connected to a parent namespace, which in turn can provide additional CTEs, physical tables or
    subqueries (if the current namespace is for a LATERAL subquery). This allows the current namespace to check whether some
    column is actually provided by an outer scope if the namespace does not provide the column itself.
    """

    _schema_cache: SchemaCache = SchemaCache()
    """The schema cache that is used to resolve column bindings. This cache is shared through the entire program lifetime.

    Changing the actual database schema while PostBOUND is running will result in undefined behavior.
    """

    @staticmethod
    def empty(
        schema: Optional[DatabaseSchema] = None, *, top_level: bool = False
    ) -> QueryNamespace:
        QueryNamespace._schema_cache.initialize_with(schema)
        return QueryNamespace(top_level=top_level)

    def __init__(
        self,
        *,
        parent: Optional[QueryNamespace] = None,
        top_level: bool = False,
    ) -> None:
        self._parent = parent
        self._top_level = top_level

        self._subquery_children: dict[str, QueryNamespace] = {}
        """Nested namespaces that are provided as part of subqueries. Entries map alias -> query."""

        self._setop_children: list[QueryNamespace] = []
        """Namespace of the queries that form a set operation in the current namespace."""

        self._current_ctx: list[TableReference] = []
        """The tables that are currently in scope, no matter their origin (CTEs or FROM clause).

        For the purpose of this dictionary, it does not matter where a table comes from (physical table, CTE, subquery, ...).
        The only thing that matters is that the table is part of the FROM clause. This is especially important to build the
        correct output shape of the namespace/relation if the SELECT clause contains * expressions.

        Therefore, there might be tables that are contained in the `_cte_sources`, but not here (if the CTE is only used to
        build other CTEs, but not part of the FROM clause itself).

        Notice that tables that are defined in an enclosing scope (e.g. outer query in a sequence of nested CTEs) are not
        contained in this context if they are not also part of this namespace's *FROM* clause. Instead, they are resolved
        through the API on the parent namespace.

        The ordering is important to resolve output columns correctly to the first match as Postgres does.

        An optimized container to check whether a table is part of the current context is available via `_table_sources`
        """

        self._cte_sources: dict[str, QueryNamespace] = {}
        """Namespaces that are induced by CTEs. Entries map alias -> CTE."""

        self._table_sources: dict[str, TableReference] = {}
        """The tables that are part of the FROM clause of the query. Entries map alias -> table.

        Tables can be contained in this dictionary multiple times: once for each relevant identifier. If a table has both an
        alias as well as a full name, both keys will be present.
        """

        self._output_shape: list[str] = []
        """The column names that are part of the result set produced by the queries in this namespace.

        These are really just the column names, not full references. This is because it makes the access mechanism more
        transparent (just use the name, duh) and prevents accidental issues when a column from an inner query is re-used in an
        outer query and is bound to both virtual tables. Comparing the references would indicate that these are different
        columns (which arguably they are, just not for our purposes).
        """

        self._produced_columns: set[str] = set()
        """Column names that are produced by the SELECT clause of this namespace.

        This includes all aliases that are defined in this select clause. These columns cannot be bound to any table in the
        current context, but have to be bound to the aliased table for subqueries, etc.
        """

        self._column_cache: dict[str, TableReference] = {}
        """A cache to resolve common columns in the current context more quickly."""

    def determine_output_shape(
        self, select_clause: Optional[Select | Iterable[ColumnReference | str]]
    ) -> None:
        """Determines the columns that form the result relation of this namespace.

        The result is only stored internally to allow parent namespaces to resolve column references correctly.

        This method should only be called after all table sources from the current namespace are already registered in order to
        ensure that star expressions can be resolved correctly.
        """
        self._output_shape = []
        if self._setop_children:
            # We use Postgre's rules here: the output relation of a set operation contains exactly those columns that are
            # contained in the LHS relation
            self._output_shape = list(self._setop_children[0]._output_shape)
            return

        assert select_clause is not None, (
            "SELECT clause expected for non-setop query"
        )
        for projection in select_clause:
            if isinstance(projection, (str, ColumnReference)):
                # this is for temporary tables (CTEs or VALUES) that define their schema
                colname = (
                    projection.name
                    if isinstance(projection, ColumnReference)
                    else projection
                )
                self._output_shape.append(colname)
                self._produced_columns.add(colname)
                continue

            # Must be BaseProjection of a SELECT clause
            if projection.target_name:
                # If we have an alias on the projection, we need to use this alias to reference to the column in the future
                self._output_shape.append(projection.target_name)
                self._produced_columns.add(projection.target_name)
                continue

            match projection.expression:
                case ColumnExpression(col):
                    self._output_shape.append(col.name)

                case StarExpression(from_table):
                    ctx = {from_table} if from_table else self._current_ctx
                    for table in ctx:
                        if not table.virtual:
                            self._output_shape.extend(
                                self._schema_cache.columns_of(table)
                            )
                            continue

                        if table.alias:
                            defining_nsp = self._lookup_namespace(table.alias)
                        if not defining_nsp and table.full_name:
                            # if we try to look up an aliased CTE, we need to use the full name instead
                            defining_nsp = self._lookup_namespace(
                                table.full_name
                            )
                        if not defining_nsp:
                            raise ParserError(
                                f"No namespace found for table '{table}'"
                            )
                        self._output_shape.extend(defining_nsp._output_shape)

                case _ if self._top_level:
                    pass

                case _ if not self._top_level:
                    # Do nothing, this is an expression that cannot be referenced later on!
                    # Or at least, it is highly system-dependent how the expression would be named.
                    # For example, Postgres uses the function names for function calls without alias
                    warnings.warn(
                        f"Found complex expression without an alias: '{projection.expression}'. "
                        "Such expressions may currently not be referenced later on.",
                        category=ParserWarning,
                        stacklevel=2,
                    )

    def register_table(self, table: TableReference) -> None:
        """Adds a "physical" table to the current namespace.

        In truth, the table does not need to be physical, it can also be a CTE that was defined in an outer namespace and
        is scanned here. "Physical" in this context means that the current namespace does not define the table itself.
        """
        self._invalidate_column_cache()
        self._current_ctx.append(table)
        if table.alias:
            self._table_sources[table.alias] = table
        if table.full_name:
            self._table_sources[table.full_name] = table

    def provides_column(self, name: str) -> bool:
        """Checks, whether the current namespace has a specific column in its output relation."""
        return name in self._output_shape

    def lookup_column(self, key: str) -> Optional[TableReference]:
        """Searches for the table that provies a specific column.

        This table can be either virtual, i.e. a subquery or CTE (possibly from an outer namespace), or an actual physical
        table from the current database.

        If no table is found , *None* is returned.
        """
        if key in self._produced_columns:
            # This is a column that we just "created" in our select clause. It cannot (yet) belong to any table.
            return None

        cached_table = self._column_cache.get(key)
        if cached_table:
            return cached_table

        matching_table: Optional[TableReference] = None
        for table in self._current_ctx:
            # later tables overwrite unqualified columns of earlier tables
            physical_table = (
                self._schema_cache.lookup_column(key, [table])
                if not table.virtual
                else None
            )
            if physical_table:
                matching_table = table
                break

            subquery_nsp = self._subquery_children.get(table.identifier())
            if subquery_nsp and subquery_nsp.provides_column(key):
                matching_table = table
                break

            cte_nsp = self._cte_sources.get(table.identifier())
            if cte_nsp and cte_nsp.provides_column(key):
                matching_table = table
                break

            parent_nsp = self._lookup_namespace(table.identifier())
            if parent_nsp and parent_nsp.provides_column(key):
                matching_table = table
                break

        if not matching_table:
            return None

        self._column_cache[key] = matching_table
        return matching_table

    def resolve_table(self, key: str) -> Optional[TableReference]:
        """Searches for the table that is referenced by a specific key.

        The table can be either provided by this namespace (as a physical table in the *FROM* clause, or defined through a
        subquery/CTE), or by an outer namspace.
        """
        sourced_table = self._table_sources.get(key)
        if sourced_table:
            return sourced_table

        if key in self._cte_sources:
            return TableReference.create_virtual(key)

        return self._parent.resolve_table(key) if self._parent else None

    def open_nested(
        self,
        *,
        alias: str = "",
        source: Literal["cte", "subquery", "setop", "values", "temporary"],
    ) -> QueryNamespace:
        """Creates a new local namespace for a nested query.

        Depending on the type of nested query, the namespace will be registered in different ways and used for different
        purposes (see parameters below).

        Parameters
        ----------
        alias : str, optional
            The name of the namespace. This is only relevant for CTEs and subqueries in the FROM clause.
        source : Literal["cte", "subquery", "setop", "values", "temporary"]
            The type of nested query. This value is used to determine the use of the subquery namespace as follows:

            - "cte": The namespace is a CTE that is part of the query.
            - "subquery": The namespace is a subquery in the FROM clause.
            - "setop": The namespace is part of a set operation. No alias is required, but the namespace might be used to
              determine the output shape of the current namespace
            - "values": The namespace is a temporary table that is part of a VALUES clause.
            - "temporary": The namespace is a temporary table that is part of a subquery which is not used in the FROM clause,
              e.g. as a filter condition.

        """
        if source != "temporary":
            self._invalidate_column_cache()

        child = QueryNamespace(parent=self)

        match source:
            case "cte":
                self._cte_sources[alias] = child
            case "subquery" | "values":
                table = TableReference.create_virtual(alias)
                self._subquery_children[alias] = child
                self._current_ctx.append(table)
                self._table_sources[alias] = table
            case "setop":
                self._setop_children.append(child)
            case _:
                # ignore other sources
                pass

        return child

    def _lookup_namespace(self, table_key: str) -> Optional[QueryNamespace]:
        """Searches for the (parent) namespace that provides a specific table."""
        cte_nsp = self._cte_sources.get(table_key)
        if cte_nsp:
            return cte_nsp

        subquery_nsp = self._subquery_children.get(table_key)
        if subquery_nsp:
            return subquery_nsp

        return (
            self._parent._lookup_namespace(table_key) if self._parent else None
        )

    def _invalidate_column_cache(self) -> None:
        """Clears all currently cached columns in case there is fear of a change in the column bindings."""
        self._column_cache.clear()


def _pglast_is_actual_colref(pglast_data: dict) -> bool:
    """Checks, whether a apparent column reference is actually a column reference and not a star expression in disguise.

    pglast represents both column references such as *R.a* or *a* as well as star expressions like *R.\\** as ``ColumnRef``
    dictionaries, hence we need to make sure we are actually parsing the right thing. This method takes care of distinguishing
    the two cases.

    Parameters
    ----------
    pglast_data : dict
        JSON encoding of the potential column

    Returns
    -------
    bool
        *True* if this is an actual column reference, *False* if this is a star expression.
    """
    fields: list[dict] = pglast_data["fields"]
    if len(fields) == 1:
        return "A_Star" not in fields[0]
    if len(fields) == 2:
        would_be_col: str = fields[1]
        return "A_Star" not in would_be_col

    would_be_col: str = fields[0]["String"]["sval"]
    return not would_be_col.endswith("*")


def _pglast_create_bound_colref(
    tab: str, col: str, *, namespace: QueryNamespace
) -> ColumnReference:
    """Creates a new reference to a column with known binding info.

    Parameters
    ----------
    tab : str
        The table to which to bind
    col : str
        The column to bind
    namespace : QueryNamespace
        The tables and columns that are available in the current query.

    Returns
    -------
    ColumnReference
        The new column reference
    """
    owning_table = namespace.resolve_table(tab)
    if not owning_table:
        raise ParserError("Table not found: " + tab)
    parsed_column = ColumnReference(col, owning_table)
    return parsed_column


def _pglast_parse_colref(
    pglast_data: dict, *, namespace: QueryNamespace
) -> ColumnReference:
    """Handler method to parse column references in the query.

    The column will be bound to its table if possible. This binding process uses the following rules:

    - if the columns has already been resolved as part of an earlier parsing step in the same namespace, this column is re-used
    - if the column is specified in qualified syntax (i.e. *table.column*), the table is directly inferred
    - if the column is not qualified, but a `schema` is given, this schema is used together with the candidates from the
      current namespace to lookup the owning table
    - otherwise, the column is left unbound :(

    Parameters
    ----------
    pglast_data : dict
        JSON enconding of the column
    namespace : QueryNamespace
        The tables and columns that are available in the current query.

    Returns
    -------
    ColumnReference
        The parsed column reference.
    """
    fields: list[dict] = pglast_data["fields"]
    if len(fields) > 2:
        raise ParserError(
            "Unknown column reference format: " + str(pglast_data)
        )

    if len(fields) == 2:
        tab, col = fields
        return _pglast_create_bound_colref(
            tab["String"]["sval"], col["String"]["sval"], namespace=namespace
        )

    # at this point, we must have a single column parameter. It could be unbounded, or - if quoted - bounded
    col: str = fields[0]["String"]["sval"]
    owning_table = namespace.lookup_column(col)
    return ColumnReference(col, owning_table)


def _pglast_parse_star(
    pglast_data: dict, *, namespace: QueryNamespace
) -> StarExpression:
    """Handler method to parse star expressions that are potentially bounded to a specific table, e.g. *R.\\**.

    Parameters
    ----------
    pglast_data : dict
        JSON enconding of the star expression
    namespace : QueryNamespace
        The tables and columns that are available in the current query.

    Returns
    -------
    StarExpression
        The parsed star expression.
    """
    fields = pglast_data["fields"]
    if len(fields) == 1 and "A_Star" in fields[0]:
        return StarExpression()

    if len(fields) == 2:
        tab = fields[0]["String"]["sval"]
        return StarExpression(from_table=namespace.resolve_table(tab))

    raise ParserError("Unknown star reference format: " + str(pglast_data))


def _pglast_parse_const(pglast_data: dict) -> StaticValueExpression:
    """Handler method to parse constant values in the query.

    Parameters
    ----------
    pglast_data : dict
        JSON enconding of the value. This data is extracted from the pglast data structure.

    Returns
    -------
    StaticValueExpression
        The parsed constant value.
    """
    pglast_data.pop("location", None)
    valtype = util.dicts.key(pglast_data)
    match valtype:
        case "isnull":
            return StaticValueExpression.null()
        case "ival":
            val = (
                pglast_data["ival"]["ival"]
                if "ival" in pglast_data["ival"]
                else 0
            )
            return StaticValueExpression(val)
        case "fval":
            val = pglast_data["fval"]["fval"]
            return StaticValueExpression(float(val))
        case "sval":
            return StaticValueExpression(pglast_data["sval"]["sval"])
        case "boolval":
            val = pglast_data["boolval"].get("boolval", False)
            return StaticValueExpression(val)
        case _:
            raise ParserError("Unknown constant type: " + str(pglast_data))


_PglastOperatorMap: dict[str, SqlOperator] = {
    "=": LogicalOperator.Equal,
    "<": LogicalOperator.Less,
    "<=": LogicalOperator.LessEqual,
    ">": LogicalOperator.Greater,
    ">=": LogicalOperator.GreaterEqual,
    "<>": LogicalOperator.NotEqual,
    "!=": LogicalOperator.NotEqual,
    "AND_EXPR": CompoundOperator.And,
    "OR_EXPR": CompoundOperator.Or,
    "NOT_EXPR": CompoundOperator.Not,
    "+": MathOperator.Add,
    "-": MathOperator.Subtract,
    "*": MathOperator.Multiply,
    "/": MathOperator.Divide,
    "%": MathOperator.Modulo,
    "||": MathOperator.Concatenate,
    "@>": LogicalOperator.Contains,
    "<@": LogicalOperator.ContainedBy,
    "&&": LogicalOperator.Overlaps,
    "AEXPR_DISTINCT": LogicalOperator.DistinctFrom,
    "AEXPR_NOT_DISTINCT": LogicalOperator.NotDistinctFrom,
}
"""Map from the internal representation of Postgres operators to our standardized QAL operators."""


def _pglast_parse_operator(pglast_data: list[dict]) -> SqlOperator:
    """Handler method to parse operators into our query representation.

    Parameters
    ----------
    pglast_data : list[dict]
        JSON enconding of the operator. This data is extracted from the pglast data structure.

    Returns
    -------
    SqlOperator
        The parsed operator.
    """
    if len(pglast_data) != 1:
        raise ParserError("Unknown operator format: " + str(pglast_data))
    operator = pglast_data[0]
    if "String" not in operator or "sval" not in operator["String"]:
        raise ParserError("Unknown operator format: " + str(pglast_data))
    sval = operator["String"]["sval"]
    if sval not in _PglastOperatorMap:
        raise ParserError("Operator not yet in target map: " + sval)
    return _PglastOperatorMap[sval]


_PglastTypeMap: dict[str, str] = {
    "bpchar": "char",
    "serial8": "bigserial",
    "int4": "integer",
    "int2": "smallint",
    "int8": "bigint",
    "float4": "real",
    "float8": "double precision",
    "boolean": "bool",
}
"""Map from the internal representation of Postgres types to the SQL standard types."""


def _pglast_parse_type(pglast_data: dict) -> str:
    """Handler method to parse type information from explicit type casts

    Parameters
    ----------
    pglast_data : dict
        JSON encoding of the type information.

    Returns
    -------
    str
        The actual type
    """
    if "names" not in pglast_data:
        raise ParserError("Unknown type format: " + str(pglast_data))
    names = pglast_data["names"]
    if len(names) > 2:
        raise ParserError("Unknown type format: " + str(pglast_data))
    raw_type = names[-1]["String"]["sval"]

    # for user-defined types we use get with the same type as argument
    return _PglastTypeMap.get(raw_type, raw_type)


def _pglast_parse_case(
    pglast_data: dict, *, namespace: QueryNamespace, query_txt: str
) -> CaseExpression:
    """Handler method to parse *CASE* expressions in a query.

    Parameters
    ----------
    pglast_data : dict
        JSON encoding of the *CASE* expression data. This data is extracted from the pglast data structure.
    namespace : QueryNamespace
        The tables and columns that are available in the current query.
    query_txt : str
        The raw query text that was passed to the parser. This is used to extract information that the PG parser does not
        consider, such as hint blocks.

    Returns
    -------
    CaseExpression
        The parsed *CASE* expression.
    """
    cases: list[tuple[AbstractPredicate, SqlExpression]] = []
    for arg in pglast_data["args"]:
        current_case = _pglast_parse_predicate(
            arg["CaseWhen"]["expr"], namespace=namespace, query_txt=query_txt
        )
        current_result = _pglast_parse_expression(
            arg["CaseWhen"]["result"], namespace=namespace, query_txt=query_txt
        )
        cases.append((current_case, current_result))

    if "arg" in pglast_data:
        simple_expr = _pglast_parse_expression(
            pglast_data["arg"], namespace=namespace, query_txt=query_txt
        )
    else:
        simple_expr = None

    if "defresult" in pglast_data:
        default_result = _pglast_parse_expression(
            pglast_data["defresult"], namespace=namespace, query_txt=query_txt
        )
    else:
        default_result = None

    return CaseExpression(
        cases, simple_expr=simple_expr, else_expr=default_result
    )


def _pglast_parse_expression(
    pglast_data: dict, *, namespace: QueryNamespace, query_txt: str
) -> SqlExpression:
    """Handler method to parse arbitrary expressions in the query.

    For some more complex expressions, this method will delegate to tailored parsing methods.

    Parameters
    ----------
    pglast_data : dict
        JSON encoding of the expression data. This data is extracted from the pglast data structure.
    namespace: QueryNamespace
        The tables and columns that are available in the current query.
    query_txt : str
        The raw query text that was passed to the parser. This is used to extract information that the PG parser does not
        consider, such as hint blocks.

    Returns
    -------
    SqlExpression
        The parsed expression.
    """
    pglast_data.pop("location", None)
    expression_key = util.dicts.key(pglast_data)

    # When parsing the actual expression, we need to be aware that many expressions can actually be predicates, just not
    # within the WHERE or HAVING clause. For example, "SELECT a IS NOT NULL FROM foo" is a perfectly valid query.
    # Therefore, we handle a lot of expression cases by passing the input data back to our predicate parser and let it do the
    # heavy lifting.

    match expression_key:
        case "ColumnRef" if _pglast_is_actual_colref(pglast_data["ColumnRef"]):
            column = _pglast_parse_colref(
                pglast_data["ColumnRef"], namespace=namespace
            )
            return ColumnExpression(column)

        case "ColumnRef" if not _pglast_is_actual_colref(
            pglast_data["ColumnRef"]
        ):
            return _pglast_parse_star(
                pglast_data["ColumnRef"], namespace=namespace
            )

        case "A_Const":
            return _pglast_parse_const(pglast_data["A_Const"])

        case "A_Expr" if pglast_data["A_Expr"]["kind"] == "AEXPR_OP":
            expression = pglast_data["A_Expr"]
            operation = _pglast_parse_operator(expression["name"])
            right = _pglast_parse_expression(
                expression["rexpr"], namespace=namespace, query_txt=query_txt
            )

            if "lexpr" not in expression and operation in MathOperator:
                return MathExpression(operation, right)
            elif "lexpr" not in expression:
                raise ParserError("Unknown operator format: " + str(expression))

            left = _pglast_parse_expression(
                expression["lexpr"], namespace=namespace, query_txt=query_txt
            )

            if operation in LogicalOperator:
                return BinaryPredicate(operation, left, right)

            if isinstance(operation, CompoundOperator):
                raise ParserError(f"Unexpected compound in query '{query_txt}'")

            return MathExpression(operation, left, right)

        case "A_Expr" if pglast_data["A_Expr"]["kind"] == "AEXPR_LIKE":
            expression = pglast_data["A_Expr"]
            operator = (
                LogicalOperator.Like
                if expression["name"][0]["String"]["sval"] == "~~"
                else LogicalOperator.NotLike
            )
            left = _pglast_parse_expression(
                expression["lexpr"], namespace=namespace, query_txt=query_txt
            )
            right = _pglast_parse_expression(
                expression["rexpr"], namespace=namespace, query_txt=query_txt
            )
            return BinaryPredicate(operator, left, right)

        case "A_Expr" if pglast_data["A_Expr"]["kind"] == "AEXPR_ILIKE":
            expression = pglast_data["A_Expr"]
            operator = (
                LogicalOperator.ILike
                if expression["name"][0]["String"]["sval"] == "~~*"
                else LogicalOperator.NotILike
            )
            left = _pglast_parse_expression(
                expression["lexpr"], namespace=namespace, query_txt=query_txt
            )
            right = _pglast_parse_expression(
                expression["rexpr"], namespace=namespace, query_txt=query_txt
            )
            return BinaryPredicate(operator, left, right)

        case "A_Expr" if pglast_data["A_Expr"]["kind"] == "AEXPR_BETWEEN":
            expression = pglast_data["A_Expr"]
            left = _pglast_parse_expression(
                expression["lexpr"], namespace=namespace, query_txt=query_txt
            )
            raw_interval = expression["rexpr"]["List"]["items"]
            if len(raw_interval) != 2:
                raise ParserError(
                    "Invalid BETWEEN interval: " + str(raw_interval)
                )
            lower = _pglast_parse_expression(
                raw_interval[0], namespace=namespace, query_txt=query_txt
            )
            upper = _pglast_parse_expression(
                raw_interval[1], namespace=namespace, query_txt=query_txt
            )
            return BetweenPredicate(left, (lower, upper))

        case "A_Expr" if pglast_data["A_Expr"]["kind"] == "AEXPR_IN":
            expression = pglast_data["A_Expr"]
            left = _pglast_parse_expression(
                expression["lexpr"], namespace=namespace, query_txt=query_txt
            )
            raw_values = expression["rexpr"]["List"]["items"]
            values = [
                _pglast_parse_expression(
                    value, namespace=namespace, query_txt=query_txt
                )
                for value in raw_values
            ]
            predicate = InPredicate(left, values)
            operator = expression["name"][0]["String"]["sval"]
            if operator == "=":
                return predicate
            elif operator == "<>":
                return CompoundPredicate.create_not(predicate)
            else:
                raise ParserError("Invalid IN operator: " + operator)

        case "A_Expr" if pglast_data["A_Expr"]["kind"] in [
            "AEXPR_DISTINCT",
            "AEXPR_NOT_DISTINCT",
        ]:
            expression = pglast_data["A_Expr"]
            operator = _PglastOperatorMap[expression["kind"]]
            left = _pglast_parse_expression(
                expression["lexpr"], namespace=namespace, query_txt=query_txt
            )
            right = _pglast_parse_expression(
                expression["rexpr"], namespace=namespace, query_txt=query_txt
            )
            return BinaryPredicate(operator, left, right)

        case "A_Expr" if pglast_data["A_Expr"]["kind"] in [
            "AEXPR_OP_ANY",
            "AEXPR_OP_ALL",
        ]:
            expression = pglast_data["A_Expr"]
            operation = _pglast_parse_operator(expression["name"])
            left = _pglast_parse_expression(
                expression["lexpr"], namespace=namespace, query_txt=query_txt
            )
            right = _pglast_parse_expression(
                expression["rexpr"], namespace=namespace, query_txt=query_txt
            )

            match expression["kind"]:
                case "AEXPR_OP_ALL":
                    quantifier = QuantifierOperator.All
                case "AEXPR_OP_ANY":
                    quantifier = QuantifierOperator.Any
                case _:
                    raise ParserError(
                        "Unknown quantifier operator: " + str(expression)
                    )

            return BinaryPredicate(
                operation,
                left,
                QuantifierExpression(right, quantifier=quantifier),
            )

        case "BoolExpr":
            expression = pglast_data["BoolExpr"]
            operator = _PglastOperatorMap[expression["boolop"]]
            children = [
                _pglast_parse_predicate(
                    child, namespace=namespace, query_txt=query_txt
                )
                for child in expression["args"]
            ]
            if not isinstance(operator, CompoundOperator):
                raise ParserError(
                    f"Unexpected non-compound in query '{query_txt}'"
                )
            return CompoundPredicate(operator, children)

        case "NullTest":
            expression = pglast_data["NullTest"]
            testexpr = _pglast_parse_expression(
                expression["arg"], namespace=namespace, query_txt=query_txt
            )
            operation = (
                LogicalOperator.Is
                if expression["nulltesttype"] == "IS_NULL"
                else LogicalOperator.IsNot
            )
            return BinaryPredicate(
                operation, testexpr, StaticValueExpression.null()
            )

        case "FuncCall" if (
            "over" not in pglast_data["FuncCall"]
        ):  # normal functions, aggregates and UDFs
            expression: dict = pglast_data["FuncCall"]
            funcname: str = ".".join(
                elem["String"]["sval"] for elem in expression["funcname"]
            )
            funcname = funcname.removeprefix("pg_catalog.")

            distinct = expression.get("agg_distinct", False)
            if expression.get("agg_filter", False):
                filter_expr = _pglast_parse_predicate(
                    expression["agg_filter"],
                    namespace=namespace,
                    query_txt=query_txt,
                )
            else:
                filter_expr = None

            if expression.get("agg_star", False):
                return FunctionExpression(
                    funcname,
                    [StarExpression()],
                    distinct=distinct,
                    filter_where=filter_expr,
                )

            args = [
                _pglast_parse_expression(
                    arg, namespace=namespace, query_txt=query_txt
                )
                for arg in expression.get("args", [])
            ]

            # Sadly, the Postgres parser does not retain keywords from
            # functions. Instead, all arguments are just stuffed into a single
            # list. For example, substring(foo, FROM 3 FOR 42) becomes simply
            # ["foo", 3, 42]. This means that we cannot easily map named
            # arguments from the parser output to keyword arguments in our
            # FunctionExpression. Instead, we need to check for known functions
            # with named parameters and manually extract the keyword arguments.
            # While this is certainly tedious, it is currently the best we can
            # do.
            #
            # To keep the implementation effort manageable, we are going to
            # gradually expand the below match/case with known functions as
            # we go.

            match funcname:
                case "substring":
                    assert len(args) == 3
                    col, start_idx, n_chars = args
                    return FunctionExpression(
                        funcname,
                        [col],
                        keyword_args={"FROM": start_idx, "FOR": n_chars},
                    )
                case _:
                    return FunctionExpression(
                        funcname,
                        args,
                        distinct=distinct,
                        filter_where=filter_expr,
                    )

        case "FuncCall" if (
            "over" in pglast_data["FuncCall"]
        ):  # window functions
            expression: dict = pglast_data["FuncCall"]
            funcname = ".".join(
                elem["String"]["sval"] for elem in expression["funcname"]
            )

            args = [
                _pglast_parse_expression(
                    arg, namespace=namespace, query_txt=query_txt
                )
                for arg in expression.get("args", [])
            ]
            fn = FunctionExpression(funcname, args)

            window_spec: dict = expression["over"]

            if "partitionClause" in window_spec:
                partition = [
                    _pglast_parse_expression(
                        partition, namespace=namespace, query_txt=query_txt
                    )
                    for partition in window_spec["partitionClause"]
                ]
            else:
                partition = None

            if "orderClause" in window_spec:
                order = _pglast_parse_orderby(
                    window_spec["orderClause"],
                    namespace=namespace,
                    query_txt=query_txt,
                )
            else:
                order = None

            if "agg_filter" in expression:
                filter_expr = _pglast_parse_predicate(
                    expression["agg_filter"],
                    namespace=namespace,
                    query_txt=query_txt,
                )
            else:
                filter_expr = None

            return WindowExpression(
                fn,
                partitioning=partition,
                ordering=order,
                filter_condition=filter_expr,
            )

        case "CoalesceExpr":
            expression = pglast_data["CoalesceExpr"]
            args = [
                _pglast_parse_expression(
                    arg, namespace=namespace, query_txt=query_txt
                )
                for arg in expression["args"]
            ]
            return FunctionExpression("coalesce", args)

        case "TypeCast":
            expression: dict = pglast_data["TypeCast"]
            casted_expression = _pglast_parse_expression(
                expression["arg"], namespace=namespace, query_txt=query_txt
            )
            target_type = _pglast_parse_type(expression["typeName"])
            type_params = [
                _pglast_parse_expression(
                    param, namespace=namespace, query_txt=query_txt
                )
                for param in expression["typeName"].get("typmods", [])
            ]
            array_type = "arrayBounds" in expression["typeName"]

            return CastExpression(
                casted_expression,
                target_type,
                type_params=type_params,
                array_type=array_type,
            )

        case "CaseExpr":
            return _pglast_parse_case(
                pglast_data["CaseExpr"],
                namespace=namespace,
                query_txt=query_txt,
            )

        case "SubLink":
            expression = pglast_data["SubLink"]
            sublink_type = expression["subLinkType"]

            subquery = _pglast_parse_query(
                expression["subselect"]["SelectStmt"],
                namespace=namespace.open_nested(source="temporary"),
                query_txt=query_txt,
            )
            if sublink_type == "EXISTS_SUBLINK":
                return UnaryPredicate.exists(subquery)
            elif sublink_type == "EXPR_SUBLINK":
                return SubqueryExpression(subquery)

            testexpr = _pglast_parse_expression(
                expression["testexpr"], namespace=namespace, query_txt=query_txt
            )

            if sublink_type == "ANY_SUBLINK" and "operName" not in expression:
                return InPredicate.subquery(testexpr, subquery)

            if sublink_type == "ANY_SUBLINK":
                operator = _PglastOperatorMap[expression["operName"]]
                subquery_expression = QuantifierExpression.any(subquery)
                return BinaryPredicate(operator, testexpr, subquery_expression)
            elif sublink_type == "ALL_SUBLINK":
                operator = _PglastOperatorMap[expression["operName"]]
                subquery_expression = QuantifierExpression.all(subquery)
                return BinaryPredicate(operator, testexpr, subquery_expression)
            else:
                raise NotImplementedError(
                    "Subquery handling is not yet implemented"
                )

        case "A_Indirection":
            expression: dict = pglast_data["A_Indirection"]
            array_expression = _pglast_parse_expression(
                expression["arg"], namespace=namespace, query_txt=query_txt
            )

            for index_expression in expression["indirection"]:
                index_expression: dict = index_expression["A_Indices"]

                if index_expression.get("is_slice", False):
                    lower = (
                        _pglast_parse_expression(
                            index_expression["lidx"],
                            namespace=namespace,
                            query_txt=query_txt,
                        )
                        if "lidx" in index_expression
                        else None
                    )
                    upper = (
                        _pglast_parse_expression(
                            index_expression["uidx"],
                            namespace=namespace,
                            query_txt=query_txt,
                        )
                        if "uidx" in index_expression
                        else None
                    )
                    array_expression = ArrayAccessExpression(
                        array_expression, lower_idx=lower, upper_idx=upper
                    )
                    continue

                point_index = _pglast_parse_expression(
                    index_expression["uidx"],
                    namespace=namespace,
                    query_txt=query_txt,
                )
                array_expression = ArrayAccessExpression(
                    array_expression, idx=point_index
                )

            return array_expression

        case "A_ArrayExpr":
            expression: dict = pglast_data["A_ArrayExpr"]
            elems = [
                _pglast_parse_expression(
                    elem, namespace=namespace, query_txt=query_txt
                )
                for elem in expression.get("elements", [])
            ]
            return ArrayExpression(elems)

        case _:
            raise ParserError("Unknown expression type: " + str(pglast_data))


def _pglast_parse_values_cte(
    pglast_data: dict, *, namespace: QueryNamespace, query_txt: str
) -> tuple[ValuesList, list[str]]:
    """Handler method to parse a CTE with a *VALUES* expressions.

    Parameters
    ----------
    pglast_data : dict
        JSON encoding of the CTE data. This data is extracted from the pglast data structure.
    namespace: QueryNamespace
        The tables and columns that are available in the current query.
    query_txt : str
        The raw query text that was passed to the parser. This is used to extract information that the PG parser does not
        consider, such as hint blocks.

    Returns
    -------
    tuple[ValuesList, list[str]]
        The parsed *VALUES* expression and their column names.
    """
    values: ValuesList = []
    for row in pglast_data["ctequery"]["SelectStmt"]["valuesLists"]:
        raw_items = row["List"]["items"]
        parsed_items = [
            _pglast_parse_expression(
                item, namespace=namespace, query_txt=query_txt
            )
            for item in raw_items
        ]
        values.append(tuple(parsed_items))

    colnames: list[str] = []
    for raw_colname in pglast_data.get("aliascolnames", []):
        colnames.append(raw_colname["String"]["sval"])

    if colnames:
        namespace.determine_output_shape(colnames)

    return values, colnames


def _pglast_parse_ctes(
    json_data: dict, *, parent_namespace: QueryNamespace, query_txt: str
) -> CommonTableExpression:
    """Handler method to parse the *WITH* clause of a query.

    Parameters
    ----------
    json_data : dict
        JSON enconding of the CTEs, as extracted from the pglast data structure.
    parent_namespace: QueryNamespace
        The tables and columns that are available in the current query.
    query_txt: str
        The raw query text that was passed to the parser. This is used to extract information that the PG parser does not
        consider, such as hint blocks.

    Returns
    -------
    CommonTableExpression
        The parsed CTEs.
    """
    parsed_ctes: list[WithQuery] = []
    for pglast_data in json_data["ctes"]:
        current_cte: dict = pglast_data["CommonTableExpr"]
        target_name = current_cte["ctename"]
        target_table = TableReference.create_virtual(target_name)

        match current_cte.get("ctematerialized", "CTEMaterializeDefault"):
            case "CTEMaterializeDefault":
                force_materialization = None
            case "CTEMaterializeAlways":
                force_materialization = True
            case "CTEMaterializeNever":
                force_materialization = False

        query_data = current_cte["ctequery"]["SelectStmt"]
        child_nsp = parent_namespace.open_nested(
            alias=target_name, source="cte"
        )
        if "targetList" not in query_data and query_data["op"] == "SETOP_NONE":
            # CTE is a VALUES query
            values, columns = _pglast_parse_values_cte(
                current_cte, namespace=child_nsp, query_txt=query_txt
            )
            parsed_cte = ValuesWithQuery(
                values,
                target_name=target_table.identifier(),
                columns=columns,
                materialized=force_materialization,
            )
        else:
            cte_query = _pglast_parse_query(
                current_cte["ctequery"]["SelectStmt"],
                namespace=child_nsp,
                query_txt=query_txt,
            )
            parsed_cte = WithQuery(
                cte_query, target_table, materialized=force_materialization
            )

        parsed_ctes.append(parsed_cte)

    recursive = json_data.get("recursive", False)
    return CommonTableExpression(parsed_ctes, recursive=recursive)


def _pglast_try_select_star(
    target: dict, *, distinct: list[SqlExpression] | bool
) -> Optional[Select]:
    """Attempts to generate a *SELECT(\\*)* representation for a *SELECT* clause.

    If the query is not actually a *SELECT(\\*)* query, this method will return *None*.

    Parameters
    ----------
    target : dict
        JSON encoding of the target entry in the *SELECT* clause. This data is extracted from the pglast data structure
    distinct : list[SqlExpression] | bool
        The parsed *DISTINCT* part of the *SELECT* clause.

    Returns
    -------
    Optional[Select]
        The parsed *SELECT(\\*)* clause, or *None* if this is not a *SELECT(\\*)* query.
    """
    if "ColumnRef" not in target:
        return None
    fields = target["ColumnRef"]["fields"]
    if len(fields) != 1:
        # multiple fields are used for qualified column references. This is definitely not a SELECT * query, so exit
        return None
    colref = fields[0]
    return Select.star(distinct=distinct) if "A_Star" in colref else None


def _pglast_parse_select(
    pglast_data: dict, *, namespace: QueryNamespace, query_txt: str
) -> Select:
    """Handler method to parse the *SELECT* clause of a query.

    This is the only parsing handler that will always be called when parsing a query, since all queries must at least have a
    *SELECT* clause.

    Parameters
    ----------
    pglast_data : dict
        JSON encoding of the entire query. This is required to extract the different projections used in the *SELECT* clause,
        as well as potential required duplicate eliminations via *DISTINCT ON*
    namespace : QueryNamespace
        The tables and columns that are available in the current query.
    query_txt : str
        The raw query text that was passed to the parser. This is used to extract information that the PG parser does not
        consider, such as hint blocks.

    Returns
    -------
    Select
        The parsed *SELECT* clause
    """

    pglast_distinct = pglast_data.get("distinctClause", None)
    if pglast_distinct is None:
        distinct = False  # value not present --> no DISTINCT
    elif pglast_distinct == [
        {}
    ]:  # that is pglasts encoding of a plain DISTINCT
        distinct = True
    elif isinstance(pglast_distinct, list):
        distinct = [
            _pglast_parse_expression(
                expr, namespace=namespace, query_txt=query_txt
            )
            for expr in pglast_distinct
        ]
    else:
        raise ParserError(f"Unknown DISTINCT format: {pglast_distinct}")

    targetlist: list[dict] = pglast_data[
        "targetList"
    ]  # targetlist must always be present, so no need for .get()

    # first, try for SELECT * queries
    if len(targetlist) == 1:
        target = targetlist[0]["ResTarget"]["val"]
        select_star = _pglast_try_select_star(target, distinct=distinct)

        if select_star:
            namespace.determine_output_shape(select_star)
            return select_star
        # if this is not a SELECT * query, we can continue with the regular parsing

    targets: list[BaseProjection] = []
    for target in targetlist:
        expression = _pglast_parse_expression(
            target["ResTarget"]["val"], namespace=namespace, query_txt=query_txt
        )
        alias = target["ResTarget"].get("name", "")
        projection = BaseProjection(expression, alias)
        targets.append(projection)

    clause = Select(targets, distinct=distinct)
    namespace.determine_output_shape(clause)
    return clause


def _pglast_parse_rangevar(rangevar: dict) -> TableReference:
    """Handler method to extract the `TableReference` from a *RangeVar* entry in the *FROM* clause.

    Parameters
    ----------
    rangevar : dict
        JSON encoding of the range variable, as extracted from the pglast data structure.

    Returns
    -------
    TableReference
        The parsed table reference.
    """
    name = rangevar["relname"]
    alias = rangevar["alias"]["aliasname"] if "alias" in rangevar else None
    schema = rangevar.get("schemaname", "")
    return TableReference(name, alias, schema=schema)


def _pglast_is_values_list(pglast_data: dict) -> bool:
    """Checks, whether a pglast subquery representation refers to an actual subquery or a *VALUES* list.

    Parameters
    ----------
    pglast_data : dict
        JSON encoding of the subquery data

    Returns
    -------
    bool
        *True* if the subquery encodes a *VALUES* list, *False* otherwise.
    """
    query = pglast_data["subquery"]["SelectStmt"]
    return "valuesLists" in query


def _pglast_parse_from_entry(
    pglast_data: dict, *, namespace: QueryNamespace, query_txt: str
) -> TableSource:
    """Handler method to parse individual entries in the *FROM* clause.

    Parameters
    ----------
    pglast_data : dict
        JSON enconding of the current entry in the *FROM* clause. This data is extracted from the pglast data structure.
    namespace: QueryNamespace
        The tables and columns that are available in the current query.
    query_txt: str
        The raw query string that was passed to the parser. This is used to extract information that the PG parser does not
        provide, such as hint blocks.

    Returns
    -------
    TableSource
        The parsed table source.
    """
    pglast_data.pop("location", None)
    entry_type = util.dicts.key(pglast_data)

    match entry_type:
        case "RangeVar":
            table = _pglast_parse_rangevar(pglast_data["RangeVar"])

            # If we specified a virtual table in a CTE, we will reference it later in some FROM clause. In this case,
            # we should not create a new table reference, but rather use the existing one.
            # But, if we alias the virtual table, we still need a new reference
            similar_table = namespace.resolve_table(table.full_name)
            if similar_table and similar_table.virtual and not table.alias:
                # a simple reference to the CTE
                namespace.register_table(similar_table)
                return DirectTableSource(similar_table)
            if similar_table and similar_table.virtual and table.alias:
                # an aliased reference to the CTE
                table = table.make_virtual()
                # TODO: should we also update the mapping of the full_name here?

            namespace.register_table(table)
            return DirectTableSource(table)

        case "JoinExpr":
            join_expr: dict = pglast_data["JoinExpr"]
            match join_expr["jointype"]:
                case "JOIN_INNER" if "quals" in join_expr:
                    join_type = JoinType.InnerJoin
                case "JOIN_INNER" if "quals" not in join_expr:
                    join_type = JoinType.CrossJoin
                case "JOIN_LEFT":
                    join_type = JoinType.LeftJoin
                case "JOIN_RIGHT":
                    join_type = JoinType.RightJoin
                case "JOIN_OUTER":
                    join_type = JoinType.OuterJoin
                case "JOIN_FULL":
                    join_type = JoinType.OuterJoin
                case _:
                    raise ParserError(
                        "Unknown join type: " + join_expr["jointype"]
                    )

            left = _pglast_parse_from_entry(
                join_expr["larg"], namespace=namespace, query_txt=query_txt
            )
            right = _pglast_parse_from_entry(
                join_expr["rarg"], namespace=namespace, query_txt=query_txt
            )
            if join_type == JoinType.CrossJoin:
                return JoinTableSource(
                    left, right, join_type=JoinType.CrossJoin
                )

            join_condition = _pglast_parse_predicate(
                join_expr["quals"], namespace=namespace, query_txt=query_txt
            )

            # we do not need to store new tables in available_tables here, since this is already handled by the recursion.
            return JoinTableSource(
                left, right, join_condition=join_condition, join_type=join_type
            )

        case "RangeSubselect" if _pglast_is_values_list(
            pglast_data["RangeSubselect"]
        ):
            values_list = _pglast_parse_values(
                pglast_data["RangeSubselect"],
                parent_namespace=namespace,
                query_txt=query_txt,
            )
            return values_list

        case "RangeSubselect":
            raw_subquery: dict = pglast_data["RangeSubselect"]
            is_lateral = raw_subquery.get("lateral", False)

            if "alias" in raw_subquery:
                alias: str = raw_subquery["alias"]["aliasname"]
            else:
                alias = ""

            child_nsp = namespace.open_nested(alias=alias, source="subquery")
            subquery = _pglast_parse_query(
                raw_subquery["subquery"]["SelectStmt"],
                namespace=child_nsp,
                query_txt=query_txt,
            )

            subquery_source = SubqueryTableSource(
                subquery, target_name=alias, lateral=is_lateral
            )
            return subquery_source

        case "RangeFunction":
            raw_function: dict = pglast_data["RangeFunction"]

            if "alias" in raw_function:
                alias: str = raw_function["alias"]["aliasname"]
            else:
                alias = ""

            function_expr: dict = raw_function["functions"][0]["List"]["items"][
                0
            ]
            parsed_function = _pglast_parse_expression(
                function_expr, namespace=namespace, query_txt=query_txt
            )
            return FunctionTableSource(parsed_function, alias=alias)

        case _:
            raise ParserError("Unknow FROM clause entry: " + str(pglast_data))


def _pglast_parse_values(
    pglast_data: dict, *, parent_namespace: QueryNamespace, query_txt: str
) -> ValuesTableSource:
    """Handler method to parse explicit *VALUES* lists in the *FROM* clause.

    Parameters
    ----------
    pglast_data : dict
        JSON encoding of the actual *VALUES* list. This data is extracted from the pglast data structure and should be akin
        to a subquery.
    parent_namespace : QueryNamespace
        The tables and columns that are available in the current query. This is only used to register the columns of the
        VALUES list
    query_txt : str
        The raw query text that was passed to the parser. This is used to extract information from the query that the PG
        parser ignores, such as hint blocks.

    Returns
    -------
    ValuesTableSource
        The parsed *VALUES* list.
    """

    raw_alias: dict = pglast_data.get("alias", {})
    alias = raw_alias.get("aliasname", "")
    child_nsp = parent_namespace.open_nested(alias=alias, source="values")
    raw_values: list[dict] = pglast_data["subquery"]["SelectStmt"][
        "valuesLists"
    ]

    values: ValuesList = []
    for row in raw_values:
        raw_items = row["List"]["items"]
        parsed_items = [
            _pglast_parse_expression(
                item, namespace=child_nsp, query_txt=query_txt
            )
            for item in raw_items
        ]
        values.append(tuple(parsed_items))

    if not alias:
        return ValuesTableSource(values, alias=alias, columns=[])

    if "colnames" not in raw_alias:
        return ValuesTableSource(values, alias=alias, columns=[])

    colnames = []
    for raw_colname in raw_alias["colnames"]:
        colnames.append(raw_colname["String"]["sval"])
    table_source = ValuesTableSource(values, alias=alias, columns=colnames)
    child_nsp.determine_output_shape(table_source.cols)
    return table_source


def _pglast_parse_from(
    from_clause: list[dict], *, namespace: QueryNamespace, query_txt: str
) -> From:
    """Handler method to parse the *FROM* clause of a query.

    Parameters
    ----------
    from_clause : list[dict]
        The JSON representation of the *FROM* clause, as extracted from the pglast data structure.
    namespace : QueryNamespace
        The tables and columns that are available in the current query.
    query_txt : str
        The raw query text that was passed to the parser. This is used to extract information from the query that the PG
        parser ignores, such as hint blocks.

    Returns
    -------
    From
        The parsed *FROM* clause.
    """
    contains_plain_table = False
    contains_join = False
    contains_mixed = (
        False  # plain tables and explicit JOINs, subqueries or VALUES
    )

    table_sources: list[TableSource] = []
    for entry in from_clause:
        current_table_source = _pglast_parse_from_entry(
            entry, namespace=namespace, query_txt=query_txt
        )
        table_sources.append(current_table_source)

        match current_table_source:
            case DirectTableSource():
                contains_plain_table = True
                if contains_join:
                    contains_mixed = True
            case JoinTableSource():
                contains_join = True
                if contains_plain_table:
                    contains_mixed = True
            case SubqueryTableSource():
                contains_mixed = True
            case ValuesTableSource():
                contains_mixed = True
            case FunctionTableSource():
                contains_mixed = True
            case _:
                raise ParserError(
                    f"Unknown table source type: {type(current_table_source).__name__}"
                )

    if not contains_join and not contains_mixed:
        return ImplicitFromClause(table_sources)
    if contains_join and not contains_mixed:
        return ExplicitFromClause(table_sources)

    return From(table_sources)


def _pglast_parse_predicate(
    pglast_data: dict, *, namespace: QueryNamespace, query_txt: str
) -> AbstractPredicate:
    """Handler method to parse arbitrary predicates in the *WHERE* or *HAVING* clause.

    Parameters
    ----------
    pglast_data : dict
        JSON encoding of the predicate data. This data is extracted from the pglast data structure.
    namespace : QueryNamespace
        The tables and columns that are available in the current query.
    query_txt : str
        The raw query text that was passed to the parser. This is used to extract information from the query that the PG
        parser ignores, such as hint blocks.


    Returns
    -------
    AbstractPredicate
        The parsed predicate.
    """
    predicate = _pglast_parse_expression(
        pglast_data, namespace=namespace, query_txt=query_txt
    )
    return (
        predicate
        if isinstance(predicate, AbstractPredicate)
        else UnaryPredicate(predicate)
    )


def _pglast_parse_where(
    where_clause: dict, *, namespace: QueryNamespace, query_txt: str
) -> Where:
    """Handler method to parse the *WHERE* clause of a query.

    Parameters
    ----------
    where_clause : dict
        The JSON representation of the *WHERE* clause, as extracted from the pglast data structure.
    namespace: QueryNamespace
        The tables and columns that can be referenced by expressions in the query.
    query_txt : str
        The raw query text that was passed to the parser. This is used to extract information from the query that the PG
        parser ignores, such as hint blocks.

    Returns
    -------
    Where
        The parsed *WHERE* clause.
    """
    predicate = _pglast_parse_predicate(
        where_clause, namespace=namespace, query_txt=query_txt
    )
    return Where(predicate)


def _pglast_parse_groupby(
    groupby_clause: list[dict], *, namespace: QueryNamespace, query_txt: str
) -> GroupBy:
    """Handler method to parse the *GROUP BY* clause of a query.

    Parameters
    ----------
    groupby_clause : list[dict]
        The JSON representation of the *GROUP BY* clause, as extracted from the pglast data structure
    namespace: QueryNamespace
        The tables and columns that can be referenced by expressions in the query
    query_txt : str
        The raw query text that was passed to the parser. This is used to extract information from the query that the PG
        parser ignores, such as hint blocks.

    Returns
    -------
    GroupBy
        The parsed *GROUP BY* clause.
    """
    groupings: list[SqlExpression] = []

    for item in groupby_clause:
        if "GroupingSet" in item:
            raise NotImplementedError("Grouping sets are not yet supported")
        group_expression = _pglast_parse_expression(
            item, namespace=namespace, query_txt=query_txt
        )
        groupings.append(group_expression)

    return GroupBy(groupings)


def _pglast_parse_having(
    having_clause: dict, *, namespace: QueryNamespace, query_txt: str
) -> Having:
    """Handler method to parse the *HAVING* clause of a query.

    Parameters
    ----------
    having_clause : dict
        The JSON representation of the *HAVING* clause, as extracted from the pglast data structure.
    namespace: QueryNamespace
        The tables and columns that can be referenced by expressions in the query.
    query_txt : str
        The raw query text that was passed to the parser. This is used to extract information from the query that the PG
        parser ignores, such as hint blocks.

    Returns
    -------
    Having
        The parsed *HAVING* clause.
    """
    predicate = _pglast_parse_predicate(
        having_clause, namespace=namespace, query_txt=query_txt
    )
    return Having(predicate)


def _pglast_parse_orderby(
    order_clause: list[dict], *, namespace: QueryNamespace, query_txt: str
) -> OrderBy:
    """Handler method to parse the *ORDER BY* clause of a query.

    Parameters
    ----------
    order_clause : list[dict]
        The JSON representation of the *ORDER BY* clause, as extracted from the pglast data structure.
    namespace : QueryNamespace
        The tables and columns that can be referenced by expressions in the query.
    query_txt : str
        The raw query text that was passed to the parser. This is used to extract information from the query that the PG
        parser ignores, such as hint blocks.

    Returns
    -------
    OrderBy
        The parsed *ORDER BY* clause.
    """
    orderings: list[OrderByExpression] = []

    for item in order_clause:
        expression = item["SortBy"]
        sort_key = _pglast_parse_expression(
            expression["node"], namespace=namespace, query_txt=query_txt
        )

        match expression["sortby_dir"]:
            case "SORTBY_ASC":
                sort_ascending = True
            case "SORTBY_DESC":
                sort_ascending = False
            case "SORTBY_DEFAULT":
                sort_ascending = None
            case _:
                raise ParserError(
                    "Unknown sort direction: " + expression["sortby_dir"]
                )

        match expression["sortby_nulls"]:
            case "SORTBY_NULLS_FIRST":
                put_nulls_first = True
            case "SORTBY_NULLS_LAST":
                put_nulls_first = False
            case "SORTBY_NULLS_DEFAULT":
                put_nulls_first = None
            case _:
                raise ParserError(
                    "Unknown nulls placement: " + expression["sortby_nulls"]
                )

        order_expression = OrderByExpression(
            sort_key, ascending=sort_ascending, nulls_first=put_nulls_first
        )
        orderings.append(order_expression)

    return OrderBy(orderings)


def _pglast_parse_limit(
    pglast_data: dict, *, namespace: QueryNamespace, query_txt: str
) -> Optional[Limit]:
    """Handler method to parse LIMIT and OFFSET clauses.

    This method assumes that the given query actually contains *LIMIT* or *OFFSET* clauses and will fail otherwise.

    Parameters
    ----------
    pglast_data : dict
        JSON encoding of the entire query. The method takes care of accessing the appropriate keys by itself, no preparation
        of the ``SelectStmt`` is necessary.
    namespace : QueryNamespace
        The tables and columns that can be referenced by expressions in the query.
    query_txt : str
        The raw query text that was passed to the parser. This is used to extract information from the query that the PG
        parser ignores, such as hint blocks.

    Returns
    -------
    Limit
        The limit clause. Can be *None* if no meaningful limit nor a meaningful offset is specified.
    """
    raw_limit: Optional[dict] = pglast_data.get("limitCount", None)
    raw_offset: Optional[dict] = pglast_data.get("limitOffset", None)
    if raw_limit is None and raw_offset is None:
        return None

    if raw_limit is not None:
        # for LIMIT ALL there is no second ival, but instead an "isnull" member that is set to true
        raw_limit = raw_limit["A_Const"]["ival"]
        nrows: int | None = raw_limit["ival"] if "ival" in raw_limit else None
    else:
        nrows = None
    if raw_offset is not None:
        offset: int = raw_offset["A_Const"]["ival"].get("ival", 0)
    else:
        offset = None

    normalized_query = query_txt.lower()
    contains_standard_limit = (
        "limit" in normalized_query or "fetch first" in normalized_query
    )
    if raw_limit is not None and not contains_standard_limit:
        if "fetch next" in normalized_query:
            fetch_direction = "next"
        elif "fetch prior" in normalized_query:
            fetch_direction = "prior"
        elif "fetch last" in normalized_query:
            fetch_direction = "last"
        else:
            raise ParserError("Could not determine LIMIT option")
    else:
        fetch_direction = "first"

    return Limit(limit=nrows, offset=offset, fetch_direction=fetch_direction)


def _pglast_parse_setop(
    pglast_data: dict, *, parent_namespace: QueryNamespace, query_txt: str
) -> SetQuery:
    """Handler method to parse set operations.

    This method assumes that the given query is indeed a set operation and will fail otherwise.

    Parameters
    ----------
    pglast_data : dict
        JSON encoding of the entire query. The method takes care of accessing the appropriate keys by itself, no preparation
        of the ``SelectStmt`` is necessary.
    parent_namespace : QueryNamespace
        The tables and columns that can be referenced by expressions in the query.
    query_txt : str
        The raw query text that was passed to the parser. This is used to extract information that the PG parser ignores, such
        as hint blocks.

    Returns
    -------
    SetOperationClause
        The parsed set clause
    """
    if "withClause" in pglast_data:
        with_clause = _pglast_parse_ctes(
            pglast_data["withClause"],
            parent_namespace=parent_namespace,
            query_txt=query_txt,
        )
    else:
        with_clause = None

    left_query = _pglast_parse_query(
        pglast_data["larg"],
        namespace=parent_namespace.open_nested(source="setop"),
        query_txt=query_txt,
    )
    right_query = _pglast_parse_query(
        pglast_data["rarg"],
        namespace=parent_namespace.open_nested(source="setop"),
        query_txt=query_txt,
    )

    match pglast_data["op"]:
        case "SETOP_UNION":
            operator = (
                SetOperator.UnionAll
                if pglast_data.get("all", False)
                else SetOperator.Union
            )
        case "SETOP_INTERSECT":
            operator = SetOperator.Intersect
        case "SETOP_EXCEPT":
            operator = SetOperator.Except
        case _:
            raise ParserError("Unknown set operation: " + pglast_data["op"])

    if "sortClause" in pglast_data:
        order_clause = _pglast_parse_orderby(
            pglast_data["sortClause"],
            namespace=parent_namespace,
            query_txt=query_txt,
        )
    else:
        order_clause = None

    if pglast_data["limitOption"] == "LIMIT_OPTION_COUNT":
        limit_clause = _pglast_parse_limit(
            pglast_data, namespace=parent_namespace, query_txt=query_txt
        )
    else:
        limit_clause = None

    parent_namespace.determine_output_shape(None)
    return SetQuery(
        left_query,
        right_query,
        set_operation=operator,
        cte_clause=with_clause,
        orderby_clause=order_clause,
        limit_clause=limit_clause,
    )


def _pglast_parse_explain(pglast_data: dict) -> tuple[Optional[Explain], dict]:
    """Handler method to extract the *EXPLAIN* clause from a query.

    Parameters
    ----------
    pglast_data : dict
        JSON encoding of the entire query. The method takes care of accessing the appropriate keys by itself, no preparation
        of the dictionary is necessary.

    Returns
    -------
    tuple[Optional[Explain], dict]
        The parsed explain clause if one exists as well as the wrapped query. The query representation should be used for all
        further parsing steps.
    """
    if "ExplainStmt" not in pglast_data:
        return None, pglast_data

    pglast_data = pglast_data["ExplainStmt"]
    explain_options: list[dict] = pglast_data.get("options", [])

    use_analyze = False
    output_format = "TEXT"
    for option in explain_options:
        definition: dict = option["DefElem"]
        match definition["defname"]:
            case "analyze":
                use_analyze = True
            case "format":
                output_format = definition["arg"]["String"]["sval"]
            case _:
                raise ParserError("Unknown explain option: " + str(definition))

    explain_clause = Explain(use_analyze, output_format)
    return explain_clause, pglast_data["query"]


def _pglast_parse_query(
    stmt: dict, *, namespace: QueryNamespace, query_txt: str
) -> SelectStatement:
    """Main entry point into the parsing logic.

    This function takes a single SQL SELECT query and provides the corresponding `SqlQuery` object.
    While parsing the different expressions, columns are automatically bound to their tables if they use qualified names.
    Otherwise, they are inferred from the database schema if one is given. If no schema is provided, the column will be
    left unbound.

    Parameters
    ----------
    stmt : dict
        The JSON representation of the query. This should be the contents of the ``SelectStmt`` key in the JSON dictionary.
    namespace : QueryNamespace
        The tables and columns that can be referenced by expressions in the query. This is used to register the columns
        that are used in the query.
    query_txt : str
        The raw query text that was passed to the parser. This is used to extract information that the PG parser ignores, such
        as hint blocks.

    Returns
    -------
    SelectStatement
        The parsed query
    """
    if stmt["op"] != "SETOP_NONE":
        return _pglast_parse_setop(
            stmt, parent_namespace=namespace, query_txt=query_txt
        )

    clauses: list[BaseClause] = []

    if "withClause" in stmt:
        with_clause = _pglast_parse_ctes(
            stmt["withClause"], parent_namespace=namespace, query_txt=query_txt
        )
        clauses.append(with_clause)

    if "fromClause" in stmt:
        from_clause = _pglast_parse_from(
            stmt["fromClause"], namespace=namespace, query_txt=query_txt
        )
        clauses.append(from_clause)

    # Each query is guaranteed to have a SELECT clause, so we can just parse it straight away
    select_clause = _pglast_parse_select(
        stmt, namespace=namespace, query_txt=query_txt
    )
    clauses.append(select_clause)

    if "whereClause" in stmt:
        where_clause = _pglast_parse_where(
            stmt["whereClause"], namespace=namespace, query_txt=query_txt
        )
        clauses.append(where_clause)

    if "groupClause" in stmt:
        group_clause = _pglast_parse_groupby(
            stmt["groupClause"], namespace=namespace, query_txt=query_txt
        )
        clauses.append(group_clause)

    if "havingClause" in stmt:
        having_clause = _pglast_parse_having(
            stmt["havingClause"], namespace=namespace, query_txt=query_txt
        )
        clauses.append(having_clause)

    if "sortClause" in stmt:
        order_clause = _pglast_parse_orderby(
            stmt["sortClause"], namespace=namespace, query_txt=query_txt
        )
        clauses.append(order_clause)

    if stmt["limitOption"] == "LIMIT_OPTION_COUNT":
        limit_clause = _pglast_parse_limit(
            stmt, namespace=namespace, query_txt=query_txt
        )
        if limit_clause is None:
            raise ParserError(
                f"Malformed LIMIT/OFFSET clause for query {query_txt}"
            )
        clauses.append(limit_clause)

    return build_query(clauses)


def _pglast_parse_set_commands(
    pglast_data: list[dict],
) -> tuple[list[str], list[dict]]:
    """Handler method to parse all *SET* commands that precede the actual query.

    Parameters
    ----------
    pglast_data : list[dict]
        JSON encoding of the entire query. The method takes care of accessing the appropriate keys by itself, no preparation
        of the dictionary is necessary.

    Returns
    -------
    tuple[list[str], list[dict]]
        The parsed *SET* commands as a list of strings and the remaining query data. The query data is "forwarded" to the first
        encoding that does not represent a *SET* command.
    """
    prep_stmts: list[str] = []

    for i, item in enumerate(pglast_data):
        stmt: dict = item["stmt"]
        if "VariableSetStmt" not in stmt:
            break

        var_set_stmt: dict = stmt["VariableSetStmt"]
        if var_set_stmt["kind"] != "VAR_SET_VALUE":
            raise ParserError(f"Unknown variable set option: {var_set_stmt}")
        var_name = var_set_stmt["name"]
        var_value = var_set_stmt["args"][0]["A_Const"]["sval"]["sval"]

        parsed_stmt = f"SET {var_name} TO '{var_value}';"
        prep_stmts.append(parsed_stmt)

    return prep_stmts, pglast_data[i:]


def _parse_hint_block(
    raw_query: str,
    *,
    set_cmds: list[str],
    _current_hint_text: list[str] | None = None,
) -> Optional[Hint]:
    """Handler method to extract the hint block (i.e. preceding comments) from a query

    Parameters
    ----------
    raw_query : str
        The query text that was passed to the parser. We require access to the raw query, because the PG parser ignores all
        comments and does not represent them in the AST in any way.
    set_cmds: list[str]
        *SET* commands that have already been parsed. These will be added to the hint block.
    _current_hint_text : list[str], optional
        Internal parameter to keep track of the current hint text. This is used because the parsing logic uses a recursive
        implementation.

    Returns
    -------
    Optional[Hint]
        The hint block if any hints were found, or *None* otherwise.
    """
    _current_hint_text = _current_hint_text or []

    raw_query = raw_query.lstrip()
    block_hint = "/*" in raw_query
    line_hint = raw_query.startswith("--")
    if not block_hint and not line_hint:
        prep_stms = "\n".join(set_cmds)
        hints = "\n".join(_current_hint_text)
        return Hint(prep_stms, hints) if prep_stms or hints else None

    if line_hint:
        line_end = raw_query.find("\n")
        if line_end == -1:
            # should never be raised b/c parsing should have failed already at this point
            raise ParserError(f"Unterminated line comment: {raw_query}")

        line_comment = raw_query[:line_end].strip()
        _current_hint_text.append(line_comment)
        return _parse_hint_block(
            raw_query[line_end:],
            set_cmds=set_cmds,
            _current_hint_text=_current_hint_text,
        )

    # must be block hint
    block_start = raw_query.find("/*")
    block_end = raw_query.find("*/")
    if block_end == -1:
        # should never be raised b/c parsing should have failed already at this point
        raise ParserError(f"Unterminated block comment: {raw_query}")

    block_comment = raw_query[block_start : block_end + 2].strip()
    _current_hint_text.append(block_comment)
    return _parse_hint_block(
        raw_query[block_end + 2 :],
        set_cmds=set_cmds,
        _current_hint_text=_current_hint_text,
    )


def _apply_extra_clauses(
    parsed: SelectStatement,
    *,
    hint: Optional[Hint],
    explain_clause: Optional[Explain],
) -> SelectStatement:
    clauses = list(parsed.clauses())
    if hint:
        clauses.append(hint)
    if explain_clause:
        clauses.append(explain_clause)
    return build_query(clauses)


@overload
def parse_query(
    query: str,
    *,
    include_hints: bool = True,
    bind_columns: bool | None = None,
    db_schema: Optional[DatabaseSchema] = None,
) -> SqlQuery: ...


@overload
def parse_query(
    query: str,
    *,
    accept_set_query: Literal[True],
    include_hints: bool = True,
    bind_columns: Optional[bool] = None,
    db_schema: Optional[DatabaseSchema] = None,
) -> SelectStatement: ...


@overload
def parse_query(
    query: str,
    *,
    accept_set_query: Literal[False],
    include_hints: bool = True,
    bind_columns: Optional[bool] = None,
    db_schema: Optional[DatabaseSchema] = None,
) -> SqlQuery: ...


def parse_query(
    query: str,
    *,
    accept_set_query: bool = False,
    include_hints: bool = True,
    bind_columns: Optional[bool] = None,
    db_schema: Optional[DatabaseSchema] = None,
) -> SelectStatement:
    """Parses a query string into a proper `SqlQuery` object.

    During parsing, the appropriate type of SQL query (i.e. with implicit, explicit or mixed *FROM* clause) will be
    inferred automatically. Therefore, this method can potentially return a subclass of `SqlQuery`.

    Once the query has been transformed, a text-based binding process is executed. During this process, the referenced
    tables are normalized such that column references using the table alias are linked to the correct tables that are
    specified in the *FROM* clause (see the module-level documentation for an example). The parsing process can
    optionally also involve a binding process based on the schema of a live database. This is important for all
    remaining columns where the text-based parsing was not possible, e.g. because the column was specified without a
    table alias.

    Parameters
    ----------
    query : str
        The query to parse
    accept_set_query : bool, optional
        Whether set queries are a valid result of the parsing process. If this is *False* (the default), an error will be
        raised if the input query is a set query. This implies that the result of the parsing process is always a `SqlQuery`
        instance. Otherwise, the result can also be a `SetQuery` instance.
    include_hints : bool, optional
        Whether to include hints in the parsed query. If this is *True* (the default), any preceding comments in the query
        text will be parsed as a hint block. Otherwise, these comments are simply ignored.
    bind_columns : bool | None, optional
        Whether to use *live binding*. This does not control the text-based binding, which is always performed. If this
        parameter is *None* (the default), the global `auto_bind_columns` variable will be queried. Depending on its
        value, live binding will be performed or not.
    db_schema : Optional[DatabaseSchema], optional
        For live binding, this indicates the database to use. If this is *None* (the default), the database will be
        tried to extract from the `DatabasePool`

    Returns
    -------
    SqlQuery
        The parsed SQL query.
    """
    if not query:
        raise ParserError("Empty query")

    bind_columns = bind_columns or (bind_columns is None and auto_bind_columns)
    if db_schema is None and bind_columns:
        pool = DatabasePool.get_instance()
        match pool.n_databases():
            case 0:
                warnings.warn(
                    "The database pool does not contain any databases. Did you call connect() on any? "
                    "Having no database available means that the parser will be unable to infer the correct table for "
                    "unqualified column references.",
                    category=ParserWarning,
                    stacklevel=2,
                )
                db_schema = None
            case 1:
                db_schema = pool.current_database().schema()
            case _:
                warnings.warn(
                    "The database pool contains multiple databases. The parser will use an arbitrary one "
                    "for inferring the correct table of unqualified column references. This might lead to incorrect parsing "
                    "results if the databases have different schemas. Consider obtaining private connections to your databases to "
                    "prevent registration in the global database pool.",
                    category=ParserWarning,
                    stacklevel=2,
                )
                db_schema = pool.any_database().schema()

    pglast_data = json.loads(pglast.parser.parse_sql_json(query))
    stmts = pglast_data["stmts"]

    set_cmds, stmts = _pglast_parse_set_commands(stmts)
    if len(stmts) != 1:
        raise ValueError(
            "Parser can only support single-statement queries for now"
        )
    raw_query: dict = stmts[0]["stmt"]

    if "ExplainStmt" in raw_query:
        explain_clause, raw_query = _pglast_parse_explain(raw_query)
    else:
        explain_clause = None

    if "SelectStmt" not in raw_query:
        raise ValueError("Cannot parse non-SELECT queries")
    stmt = raw_query["SelectStmt"]

    parsed_query = _pglast_parse_query(
        stmt,
        namespace=QueryNamespace.empty(db_schema, top_level=True),
        query_txt=query,
    )
    if not accept_set_query and isinstance(query, SetQuery):
        raise ParserError("Input query is a set query")

    hint = (
        _parse_hint_block(query, set_cmds=set_cmds) if include_hints else None
    )
    parsed_query = _apply_extra_clauses(
        parsed_query, hint=hint, explain_clause=explain_clause
    )

    return parsed_query


def load_query(
    path: str | Path,
    *,
    accept_set_query: bool = False,
    include_hints: bool = True,
    bind_columns: Optional[bool] = None,
    db_schema: Optional[DatabaseSchema] = None,
) -> SelectStatement:
    """Loads and parses a single query from a file.

    All parameters other than the query path are forwarded to the `parse_query` method.

    See Also
    --------
    parse_query : For parameter documentation.
    """
    path = Path(path)
    query_text = path.read_text()
    return parse_query(
        query_text,
        accept_set_query=accept_set_query,
        include_hints=include_hints,
        bind_columns=bind_columns,
        db_schema=db_schema,
    )


@overload
def load_table_json(json_data: Literal[None] | Literal[""]) -> None: ...


@overload
def load_table_json(json_data: dict | str) -> TableReference: ...


def load_table_json(json_data):
    """Re-creates a table reference from its JSON encoding.

    Parameters
    ----------
    json_data : dict | str
        Either the JSON dictionary, or a string encoding of the dictionary (which will be parsed by *json.loads*)

    Returns
    -------
    Optional[TableReference]
        The actual table. If the dictionary is empty or otherwise invalid, *None* is returned.
    """
    if not json_data:
        return None
    json_data = (
        json_data if isinstance(json_data, dict) else json.loads(json_data)
    )
    return TableReference(
        json_data.get("full_name", ""),
        json_data.get("alias", ""),
        virtual=json_data.get("virtual", False),
        schema=json_data.get("schemaname", None),
    )


@overload
def load_column_json(json_data: Literal[None] | Literal[""]) -> None: ...


@overload
def load_column_json(json_data: dict | str) -> ColumnReference: ...


def load_column_json(json_data):
    """Re-creates a column reference from its JSON encoding.

    Parameters
    ----------
    json_data : dict | str
        Either the JSON dictionary, or a string encoding of the dictionary (which will be parsed by *json.loads*)

    Returns
    -------
    Optional[ColumnReference]
        The actual column. It the dictionary is empty or otherwise invalid, *None* is returned.
    """
    if not json_data:
        return None
    json_data = (
        json_data if isinstance(json_data, dict) else json.loads(json_data)
    )
    return ColumnReference(
        json_data.get("column"), load_table_json(json_data.get("table", None))
    )


@overload
def load_expression_json(json_data: Literal[None] | Literal[""]) -> None: ...


@overload
def load_expression_json(json_data: dict | str) -> SqlExpression: ...


def load_expression_json(json_data):
    """Re-creates an arbitrary SQL expression from its JSON encoding.

    Parameters
    ----------
    json_data : dict | str
        Either the JSON dictionary, or a string encoding of the dictionary (which will be parsed by *json.loads*)

    Returns
    -------
    Optional[SqlExpression]
        The actual expression. If the dictionary is empty or *None*, *None* is returned. Notice that in case of
        malformed data, errors are raised.
    """
    if not json_data:
        return None
    json_data = (
        json_data if isinstance(json_data, dict) else json.loads(json_data)
    )

    tables = [
        load_table_json(table_data)
        for table_data in json_data.get("tables", [])
    ]
    expression_str = json_data["expression"]
    if not tables:
        emulated_query = f"SELECT {expression_str}"
    else:
        from_clause_str = ", ".join(str(tab) for tab in tables)
        emulated_query = f"SELECT {expression_str} FROM {from_clause_str}"

    parsed_query = parse_query(emulated_query)
    return parsed_query.select_clause.targets[0].expression


@overload
def load_predicate_json(json_data: Literal[None] | Literal[""]) -> None: ...


@overload
def load_predicate_json(json_data: dict | str) -> AbstractPredicate: ...


def load_predicate_json(json_data):
    """Re-creates an arbitrary predicate from its JSON encoding.

    Parameters
    ----------
    json_data : dict | str
        Either the JSON dictionary, or a string encoding of the dictionary (which will be parsed by *json.loads*)

    Returns
    -------
    Optional[AbstractPredicate]
        The actual predicate. If the dictionary is empty or *None*, *None* is returned. Notice that in case of
        malformed data, errors are raised.

    Raises
    ------
    KeyError
        If the encoding does not specify the tables that are referenced in the predicate
    KeyError
        If the encoding does not contain the actual predicate
    """
    if not json_data:
        return None
    json_data = (
        json_data if isinstance(json_data, dict) else json.loads(json_data)
    )

    tables = [
        load_table_json(table_data)
        for table_data in json_data.get("tables", [])
    ]
    if not tables:
        raise KeyError("Predicate needs at least one table!")
    from_clause_str = ", ".join(str(tab) for tab in tables)
    predicate_str = json_data["predicate"]
    emulated_query = f"SELECT * FROM {from_clause_str} WHERE {predicate_str}"
    parsed_query = parse_query(emulated_query)
    if not parsed_query.where_clause:
        raise ParserError(
            f"No predicate found. Inferred query was '{parsed_query}'."
        )
    return parsed_query.where_clause.predicate
