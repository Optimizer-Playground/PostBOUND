"""Contains the basic **query abstraction layer** to conveniently model SQL queries.

The most important features of the qal are:
1. parsing query strings into qal objects
2. providing access to underlying query features such as referenced tables, aliases or predicates
3. converting queries to representations in relational algebra
4. formatting qal objects back to strings

Generally, the qal is structured around 3 fundamental concepts: At the core of the qal are SQL expressions. Such expressions
form the basic building blocks that are re-used by more high-level components. For example, there are expressions that model a
reference to a column, as well as expressions for function calls and expressions for modelling math. The `SqlExpression` acts
as the common base class for all different expression types.

Expressions are used to construct predicates or clauses. Predicates are normally part of clauses (such as a *WHERE* clause or
a *HAVING* clause). Finally, clauses are combined to form the actual SQL queries.

Using these basic building blocks, the `relalg` module provides a simple model of relational algebra, as well as means to
translate a parsed SQL query to an algebraic expression.

A common pattern when working with elements of the qal are the `tables` and `columns` methods (along with some other, more
rarely used ones). These are defined on pretty much all of the qal types and provide access to the tables, respectively the
columns that are referenced within the current element.

Notice that some references in the qal are inherently cyclic: for example, predicates can contain subqueries and
the subqueries in turn contain predicates. This might lead to cyclic import errors in certain corner cases. Such
issues can usually be solved by varying the import sequence slightly.

All concepts in the qal are modelled as data objects that are immutable. In order to modify parts of an SQL query, a
new query has to be constructed. The `transform` module provides some functions to help with that. Traversal of the different
parts of a query can be done using specific visitor implementations.

In order to generate query instances, the `parser` module can be used to read them from strings. Finally, the
`formatter` module can be used to create pretty representations of queries. The `transform`, and `formatter` modules are
available directly from the qal and do not need to be imported explicitly. The same holds for a simple relational algebra
representation in the `relalg` module. The parser provides means for reading an entire query from text, or reading parts of it
from JSON. A `parse_query` helper function is directly available from the qal module.


SQL queries
-----------

The most important type of our query abstraction is the `SqlQuery` class. It focuses on modelling an entire SQL query with all
important concepts. Notice that the focus here really in on modelling - nearly no interactive functionality, no input/output
capabilities and no modification tools are provided. These are handled by dedicated modules (e.g. the `parser` module for
reading queries from text, or the `transform` module for changing existing query objects).

In addition to the pure `SqlQuery`, a number of subclasses exist. These model queries with specific *FROM* clauses. For
example, the `ImplicitSqlQuery` provides an `ImplicitFromClause` that restricts how tables can be referenced in this clause.
For some use-cases, these might be easier to work with than the more general `SqlQuery` class, where much more diverse *FROM*
clauses are permitted.


Predicates
----------

Predicates are the central building block to represent filter conditions for SQL queries.

A predicate is a boolean expression that can be applied to a tuple to determine whether it should be kept in the intermediate
result or thrown away. PostBOUND distinguishes between two kinds of predicates, even though they are both represented by the
same class: there are filter predicates, which - as a rule of thumb - can be applied directly to base table relations.
Furthermore, there are join predicates that access tuples from different relations and determine whether the join of both
tuples should become part of the intermediate result.

PostBOUND's implementation of predicates is structured using a composite-style layout: The `AbstractPredicate` interface
describes all behaviour that is common to the concrete predicate types. There are `BasePredicate`s, which typically contain
different expressions. The `CompoundPredicate` is used to nest different predicates, thereby creating tree-shaped hierarchies.

In addition to the predicate representation, this module also provides a utility for streamlined access to simple predicates
via `SimpleFilter` and `SimpleJoin`.
Likwise, the `QueryPredicates` provide high-level access to all predicates (join and filter) that are specified in a query.
From a user perspective, this is probably the best entry point to work with predicates. Alternatively, the predicate tree can
also be traversed using custom functions.

Lastly, there exists some basic support for equivalence class computation via the `determine_join_equivalence_classes` and
`generate_predicates_for_equivalence_classes` functions.


Clauses
-------

In addition to widely accepted clauses such as the default SPJ-building blocks or grouping clauses (*GROUP BY* and
*HAVING*), some additional clauses are also defined. These include `Explain` clauses that model widely
used *EXPLAIN* queries which provide the query plan instead of optimizing the query. Furthermore, the `Hint` clause
is used to model hint blocks that can be used to pass additional non-standardized information to the database system
and its query optimizer. In real-world contexts this is mostly used to correct mistakes by the optimizer, but PostBOUND
uses this feature to enforce entire query plans. The specific contents of a hint block are not standardized by
PostBOUND and thus remains completely system-specific.

All clauses inherit from `BaseClause`, which specifies the basic common behaviour shared by all concrete clauses.
Furthermore, all clauses are designed as immutable data objects whose content cannot be changed. Any forced
modifications will break the entire query abstraction layer and lead to unpredictable behaviour.


Notes
-----
The immutability enables a very fast hashing of values as well as the caching of complicated computations. Most objects
employ a pattern of determining their hash value during initialization of the object and simply provide that
precomputed value during hashing. This helps to speed up several hot loops at optimization time significantly.
"""

from __future__ import annotations

from ._formatter import format_quick
from ._qal import (
    AbstractPredicate,
    AggregateFunctions,
    ArrayAccessExpression,
    ArrayExpression,
    BaseClause,
    BaseExpression,
    BasePredicate,
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
    DistinctType,
    ExceptClause,
    Explain,
    ExplicitFromClause,
    ExplicitSqlQuery,
    ExpressionCollector,
    From,
    FromClauseType,
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
    LogicalOperator,
    MathExpression,
    MathOperator,
    MixedSqlQuery,
    NoFilterPredicateError,
    NoJoinPredicateError,
    OrderBy,
    OrderByExpression,
    PredicateVisitor,
    QuantifierExpression,
    QuantifierOperator,
    QueryPredicates,
    Select,
    SelectStatement,
    SetOperationClause,
    SetOperator,
    SetQuery,
    SimpleFilter,
    SimpleJoin,
    SqlExpression,
    SqlExpressionVisitor,
    SqlOperator,
    SqlQuery,
    SqlStatement,
    StarExpression,
    StaticValueExpression,
    SubqueryExpression,
    SubqueryTableSource,
    TableSource,
    UnaryPredicate,
    UnarySqlOperators,
    UnionClause,
    UnwrappedFilter,
    ValuesList,
    ValuesTableSource,
    ValuesWithQuery,
    Where,
    WindowExpression,
    WithQuery,
    as_expression,
    as_func_expr,
    as_math_expr,
    as_predicate,
    as_query,
    build_query,
    collect_subqueries_in_expression,
    determine_join_equivalence_classes,
    generate_predicates_for_equivalence_classes,
)

__all__ = [
    "AbstractPredicate",
    "AggregateFunctions",
    "ArrayAccessExpression",
    "ArrayExpression",
    "BaseClause",
    "BaseExpression",
    "BasePredicate",
    "BaseProjection",
    "BetweenPredicate",
    "BinaryPredicate",
    "CaseExpression",
    "CastExpression",
    "ClauseVisitor",
    "ColumnExpression",
    "CommonTableExpression",
    "CompoundOperator",
    "CompoundPredicate",
    "DirectTableSource",
    "DistinctType",
    "ExceptClause",
    "Explain",
    "ExplicitFromClause",
    "ExplicitSqlQuery",
    "ExpressionCollector",
    "From",
    "FromClauseType",
    "FunctionExpression",
    "FunctionTableSource",
    "GroupBy",
    "Having",
    "Hint",
    "ImplicitFromClause",
    "ImplicitSqlQuery",
    "InPredicate",
    "IntersectClause",
    "JoinTableSource",
    "JoinType",
    "Limit",
    "LogicalOperator",
    "MathExpression",
    "MathOperator",
    "MixedSqlQuery",
    "NoFilterPredicateError",
    "NoJoinPredicateError",
    "OrderBy",
    "OrderByExpression",
    "PredicateVisitor",
    "QuantifierExpression",
    "QuantifierOperator",
    "QueryPredicates",
    "Select",
    "SelectStatement",
    "SetOperationClause",
    "SetOperator",
    "SetQuery",
    "SimpleFilter",
    "SimpleJoin",
    "SqlExpression",
    "SqlExpressionVisitor",
    "SqlOperator",
    "SqlQuery",
    "SqlStatement",
    "StarExpression",
    "StaticValueExpression",
    "SubqueryExpression",
    "SubqueryTableSource",
    "TableSource",
    "UnaryPredicate",
    "UnarySqlOperators",
    "UnionClause",
    "UnwrappedFilter",
    "ValuesList",
    "ValuesTableSource",
    "ValuesWithQuery",
    "Where",
    "WindowExpression",
    "WithQuery",
    "as_expression",
    "as_func_expr",
    "as_math_expr",
    "as_predicate",
    "as_query",
    "build_query",
    "collect_subqueries_in_expression",
    "determine_join_equivalence_classes",
    "format_quick",
    "generate_predicates_for_equivalence_classes",
]
