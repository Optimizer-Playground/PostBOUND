"""The **query abstraction layer** (QAL) models SQL queries.

This module provides two core functionalities: representing SQL queries via a composite tree of expressions, and
transforming SQL queries into a (formatted) textual representation.
The QAL is closely related to the following "sibling" modules: The `db` module provides means to execute SQL queries on
a physical database. The `parser` is used to translate text strings into QAL representations. The `transform` module
contains utilities to modify existing queries. Lastly, the `relalg` module offers a simple model of relational algebra
and enables translation of SQL queries into algebraic expressions.

From a high level, the QAL is structured around 3 fundamental concepts: Expressions, clauses, and queries.
At the core of the QAL are SQL expressions. Expressions are used to construct predicates or clauses. Predicates are
part of clauses (such as a *WHERE* clause or a *HAVING* clause). Finally, clauses are combined to form the actual SQL
queries. These components are explained in more detail below.

A typical pattern when working with the elements QAL are the `tables` and `columns` methods. These are defined on
all of the QAL types and provide access to the tables, respectively the columns that are referenced within the current
element. Similarly, `iterchildren` (on expressions) and `iterexpressions` (on clauses and queries) can be used to
recursively process the query tree. To apply arbirary algorithms to the query, different visitors exist (for clauses
and expressions, and one tailored to predicates). Generally speaking, these allow for a more explicit traversal of the
query tree and should be preferred over plain `iterXYZ` access.

All concepts in the QAL are modelled as immutable data objects. In order to modify parts of an SQL query, a new query
has to be constructed. The `transform` module provides some functions to help with that. Traversal of the different
parts of a query can also be done using specific visitor implementations.

SQL queries
-----------

Probably the most important type of the query abstraction is the `SqlQuery` class. It focuses on modelling an entire
SQL query with all related concepts. `SqlQuery` comes in two concrete subclasses: the `SelectStatement` models a plain
*SELECT* query, while the `SetQuery` models queries that combine two input queries via a set operation (*UNION*,
*INTERSECT* or *EXCEPT*). The `is_select_query` and `is_set_query` functions can be used to narrow down the type of an
arbitrary query.

Note that PostBOUND currently does not model *INSERT*, *UPDATE*, *DELETE* queries, etc. since these are typically not
subject to research in query optimization. If demand for such query types exist, they might be added in the future.
For the time being, these queries need to be represented by raw text strings. To future-proof client code that can work
with arbitrary queries, the `SqlStatement` super-type exists. Currently, this type is only used to represent `SqlQuery`
(with emphasis on the query part) instances, but it can be extended in the future to also model other query types.

Expressions
-----------

Expressions form the basic building blocks that are re-used by more high-level components. For example, there are
expressions that model a reference to a column, as well as expressions for function calls and expressions for modelling
math. The `SqlExpression` acts as the common base class for all different expression types.

In addition to the actual expressions, the QAL also provides the core *operators* that are used to construct
predicates, math expressions, etc. These include `BinaryOperator` or `MathOperator` among others. The `SqlOperator`
functions as the operator super-type.

Predicates
----------

Predicates are the central building block to represent filter conditions for SQL queries.

A predicate is a boolean expression that can be applied to a tuple to determine whether it should be kept in the
intermediate result or thrown away. PostBOUND distinguishes between two kinds of predicates, even though they are both
represented by the same class: there are filter predicates, which - as a rule of thumb - can be applied directly to
base table relations.
Furthermore, there are join predicates that access tuples from different relations and determine whether the join of
both tuples should become part of the intermediate result.

PostBOUND's implementation of predicates is structured using a composite-style layout: The `AbstractPredicate`
interface describes all behaviour that is common to the concrete predicate types. There are `BasePredicate`s, which
typically contain different expressions. The `CompoundPredicate` is used to nest different predicates, thereby creating
tree-shaped hierarchies.

In addition to the predicate representation, this module also provides a utility for streamlined access to simple
predicates via `SimpleFilter` and `SimpleJoin`. These are generally more convenient to work with.
Likewise, the `PredicateTree` provide high-level access to all predicates (join and filter) that are specified in a
query. The tree can be retrieved directly from the query.
From a user perspective, this is probably the best entry point to work with predicates. Alternatively, the predicate
hierarchy can also be traversed using custom functions.

Lastly, there exists some basic support for equivalence class computation via the `determine_join_equivalence_classes`
and `generate_predicates_for_equivalence_classes` functions.

Clauses
-------

In addition to widely accepted clauses such as the default SPJ-building blocks or grouping clauses (*GROUP BY* and
*HAVING*), the QAL adds some custom clauses. These include `Explain` clauses that model widely used *EXPLAIN* queries
which provide the query plan instead of optimizing the query. Furthermore, the `Hint` clause is used to model hint
blocks that can be used to pass additional non-standardized information to the database system and its query optimizer.
In real-world contexts this is mostly used to correct mistakes by the optimizer, but PostBOUND uses this feature to
enforce entire query plans. The specific contents of a hint block are not standardized by PostBOUND and thus remains
completely system-specific.

All clauses inherit from `SqlClause`, which specifies the basic common behaviour shared by all concrete clauses.
Based on this common interface, the `BaseClause` is the parent for all clauses that can be found in plain *SELECT*
queries (e.g., *FROM* or *GROUP BY*). The `SetOpClause` is specific to set queries (*UNION*, *INTERSECT*, etc.). Lastly,
the `ModifierClause` subsumes all clauses that can be used in both set and *SELECT* queries, such as *ORDER BY* or
*EXPLAIN*.

Utilities
---------

To create arbitrary queries from their clauses, the `as_query` and `build_query` utilities exist (with slightly
different signatures). Similarly, `as_expression` with its related `as_func_expr` and `as_math_expr` functions can be
used to create simple expressions with "smart" handling of the input data. The `as_predicate` function provides the
same functionality, but specialized for query predicates.

To narrow the type of an arbitrary query, the `is_select_query` and `is_set_query` functions can be used.
Similarly, `all_binary_predicates` can be used to check whether a collection of predicates are all binary predicates.

Transforming queries from their QAL representation back to a text-based representation is done via the `format_quick`
function. This function accepts a *flavor* parameter that can be used to emit different SQL dialects.

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
    AndPredicate,
    ArrayAccessExpression,
    ArrayExpression,
    AutoJoins,
    BaseClause,
    BaseExpression,
    BasePredicate,
    BetweenPredicate,
    BinaryOperator,
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
    ExpressionCollector,
    From,
    FunctionExpression,
    FunctionTableSource,
    GroupBy,
    Having,
    Hint,
    InPredicate,
    IntersectClause,
    JoinTableSource,
    JoinType,
    Limit,
    MathExpression,
    MathOperator,
    ModifierClause,
    NoFilterPredicateError,
    NoJoinPredicateError,
    NotPredicate,
    OrderBy,
    Ordering,
    OrPredicate,
    PredicateTree,
    PredicateVisitor,
    Projection,
    QuantifierExpression,
    QuantifierOperator,
    QueryTypeError,
    Select,
    SelectStatement,
    SetOpClause,
    SetOperator,
    SetQuery,
    SimpleFilter,
    SimpleJoin,
    SqlClause,
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
    TableSourceVisitor,
    UnaryOperator,
    UnaryPredicate,
    UnionClause,
    UnwrappedFilter,
    ValuesList,
    ValuesTableSource,
    ValuesWithQuery,
    Where,
    WindowExpression,
    WithQuery,
    all_binary_predicates,
    as_expression,
    as_func_expr,
    as_math_expr,
    as_predicate,
    as_query,
    build_query,
    collect_subqueries_in_expression,
    determine_join_equivalence_classes,
    generate_predicates_for_equivalence_classes,
    is_select_query,
    is_set_query,
)

__all__ = [
    "AbstractPredicate",
    "AggregateFunctions",
    "AndPredicate",
    "ArrayAccessExpression",
    "ArrayExpression",
    "AutoJoins",
    "BaseClause",
    "BaseExpression",
    "BasePredicate",
    "BetweenPredicate",
    "BinaryOperator",
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
    "ExpressionCollector",
    "From",
    "FunctionExpression",
    "FunctionTableSource",
    "GroupBy",
    "Having",
    "Hint",
    "InPredicate",
    "IntersectClause",
    "JoinTableSource",
    "JoinType",
    "Limit",
    "MathExpression",
    "MathOperator",
    "ModifierClause",
    "NoFilterPredicateError",
    "NoJoinPredicateError",
    "NotPredicate",
    "OrPredicate",
    "OrderBy",
    "Ordering",
    "PredicateTree",
    "PredicateVisitor",
    "Projection",
    "QuantifierExpression",
    "QuantifierOperator",
    "QueryTypeError",
    "Select",
    "SelectStatement",
    "SetOpClause",
    "SetOperator",
    "SetQuery",
    "SimpleFilter",
    "SimpleJoin",
    "SqlClause",
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
    "TableSourceVisitor",
    "UnaryOperator",
    "UnaryPredicate",
    "UnionClause",
    "UnwrappedFilter",
    "ValuesList",
    "ValuesTableSource",
    "ValuesWithQuery",
    "Where",
    "WindowExpression",
    "WithQuery",
    "all_binary_predicates",
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
    "is_select_query",
    "is_set_query",
]
