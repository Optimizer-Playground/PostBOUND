# Changelog

Version numbers are composed of three components, i.e. _major_._minor_._patch_
As a rough guideline, patch releases are just for fixing bugs or adding minor details
(e.g. a new default parameter to some function), minor releases change slightly
larger parts of the framework or add significant new functionality (e.g. a new
optimization pipeline or support for an SQL feature). Major releases fundamentally
shift how the framework is used and indicate stability. Since we are not ready
for the 1.0 release yet, this does not matter right now.

The [history](HISTORY.md) contains the changelogs of older PostBOUND releases.

---

## Version 0.22.1

**Due to the extensive changes of v0.22.0, we combine the changelog from v0.22.0
and v0.22.1 here.**

PostBOUND v0.22.0 is one of the largest releases of the framework so far. With
this release, we modernized key parts of the codebase, especially by eliminating
a lot of surprising or unexpected behavior and by cutting of legacy design decisions
that did not stand the test of time. Sadly, this means that this release brings
a rather large number of breaking changes. However, most of them should have a
pretty straightforward fix. Definitely check out the list below.

In terms of new functionality, highlights include first-class support for DuckDB,
a proper `ResultCache` for expensive SQL queries, `PreciseStatistics` available
across database backends, and better support for set queries (UNION, INTERSECT,
etc.).

## 🐣 New features

- Each installation of PostBOUND now ships with DuckDB dependencies. It is no
  longer necessary to request the backend explicitly.
- Introduced `AndPredicate`, `OrPredicate` and `NotPredicate` as subclasses of
  `CompoundPredicate`. This resolves the issues around different data types of
  the `children` property.
- Introduced a number of _TypeGuard_ checks for the QAL. This includes
  `is_set_query`, `is_select_query`, `all_binary_predicates`, and `all_simple_from`.
- Introduced a `TableSourceVisitor` to traverse FROM item hierarchies.
- The type hierachy of SQL clauses was updated to better reflect the actual SQL
  grammar. Specifically, `SqlClause` is now at the root of the hierarchy. Clauses
  that only apply to plain SELECT queries are `BaseClauses` (a breaking change).
  Clauses that only apply to set queries are `SetOpClause` (another breaking change).
  Clauses that apply to both types of queries (e.g., ORDER BY) are `ModifierClause`.
- A new `CardinalitiesCache` can be used to store cardinalities from expensive estimation
  calls.
- The new `ResultCache` acts as a wrapper around a `Database` and can be used to
  store result sets of expensive queries.
- The `StatisticsCatalog` (formerly `DatabaseStatistics`) now has a `null_frac`
  statistic.

## 💀 Breaking changes

Regarding the **query abstraction layer**:

- `SqlQuery` now functions as a super-type of `SelectStatement`
  (plain SELECT queries) and `SetQuery` (including UNION, INTERSECT, etc.).
- Removed `ImplicitSqlQuery`, `ExplicitSqlQuery` and `MixedSqlQuery` along with
  derived classes (e.g., `ImplicitFromClause`). `SqlQuery` now captures all of
  these cases. FROM clauses have a new `has_simple_from` method and the
  `all_simple_from` check can be used to better narrow the data type.
- The type hierachy of SELECT clauses was updated to better reflect the actual SQL
  grammar. See _New features_ for details.
- WHERE predicates now actually call the root of their predicate tree `root` instead
  of `predicate`.
- INTERSECT clauses no longer have an `input_queries` method.
- `CompoundPredicate` is now abstract. Use `AndPredicate`, `OrPredicate` or
  `NotPredicate` instead. As a consequence, the `PredicateVisitor` no longer receives
  the children as argument.
- Predicate types have been cleaned up, including unary predicates and IN predicates.
  See _Updates_ for details.
- `BasePredicate` no longer exposes an `operation` property. This is now entirely
  dependent on the actual predicate type.
- BETWEEN predicates now call their interval bounds `lower` and `upper` to adapt
  standard community lingo.
- `MathExpression` is now always composed of a left-hand expression and (optionally)
  exactly one right-hand expression. It is no longer possible to represent a math
  expression with more than two children. Practically speaking, instead of representing
  the sum of three expressions as `MathExpression(a, +, [b, c])`, it is now represented
  as `MathExpression(MathExpression(a, +, b), +, c)`.
- `LogicalOperator` has been split into `BinaryOperator` and `UnaryOperator`.
- QAL elements that are composed of two children (e.g., set operations, math
  expressions, or binary predicates) now consistently call their children
  `lhs` and `rhs`.
- `BaseProjection` is now just `Projection`
- `OrderByExpression` is now just `Ordering`
- `QueryPredicates` is now more aptly called `PredicateTree` and can no longer be
  empty. As a side effect, `SqlQuery.predicates()` can now return _None_.

Regarding the **database abstraction**:

- Result caching is no longer mixed with regular database functionality. Instead,
  the new `ResultCache` acts as a wrapper around a `Database`.
- Statistics no longer have an emulation mode. Instead, the new `PreciseStatistics`
  should be used. Emulation is still allowed for unsupported statistics. To make
  these changes more apparent, `DatabaseStatistics` has been renamed to `StatisticsCatalog`.
- Names of the standard backend interfaces have been simplified: `PostgresInterface`
  is now `PostgresDatabase`, `PostgresSchemaInterface` is now `PostgresSchema`,
  `PostgresStatisticsInterface` is now `PostgresStatistics`, `PostgresExplainNode`
  is now `PostgresExplain`, and `PostgresExplainPlan` is now `PostgresPlan`.
  Similarly, the `DuckDBInterface` is now `DuckDBDatabase`.
- The `ParallelQueryExecutor`, `TimeoutQueryExecutor`, and `WorkloadShifter` have
  been removed from the Postgres module. For timeout support, use the `PostgresDatabase`
  directly.

Regarding the **optimization pipelines**:

- `JoinOrderOptimization` is now just `JoinOrdering`
- `PhysicalOperatorSelection` is now just `OperatorSelection`
- `JoinOrdering.optimize_join_order` may no longer return _None_.
- `JoinOperator.IndexNestedLoopJoin` was removed.

Regarding the **optimizer module**:

- The module hierarchy was flattened. `dynprog`, etc. are gone. Their types are
  now available directly in the `opt` module.
- The `noopt` module has been removed.
- The deprecated `presets`, `ues`, and `tonic` modules have been removed.
- `PreciseCardinalities` are now `PerfectCardinalities`. The overall interface and
  performance have been significantly improved.
- `PreComputedCardinalities` are now `OfflineCardinalities`. The overall interface
  and performance have been significantly improved.
- `CardinalityDistortion` has been removed.
- The legacy `JoinGraph` and helper classes have been removed.

Others:

- `Workload` is now an immutable _Mapping_.
- The deprecated `ceb` module has been removed.
- `util.enlist` has been removed.

## 📰 Updates

- Each `OptimizationStage` now checks whether its required hints are available on
  the target database system as a pre-check. As a consequence, subclasses should
  now merge their checks with the base class.
- `InPredicate` can now be used to represent both IN filters as well as NOT IN filters.
- IS NULL and related operations are now proper `UnaryPredicate` instances instead
  of weird cases of `BinaryPredicates`. This aligns with the SQL standard.
- The `DatabasePool` now takes the actual database into account when caching, not
  just the system name. Therefore, it is now possible to open connections to two
  different Postgres databases without setting _private_ or _refresh_ during `connect()`.
- `postgres.connect()` now accepts the _config_file_ as a positional argument.

## 🏥 Fixes

- The string representation of SQL queries now puts parentheses around NOT.
- The SQL parser can now handle NATURAL JOINs properly.
- `transform.extract_subquery` now ignores anything not SELECT, FROM, or WHERE -
  as it was originally intended to do.
- `transform.replace_predicate` can now handle replacements of the root.
- `transform.add_ec_predicates` keeps non-equi joins
- Tons of smaller bug fixes for edge cases in the `transform` module.
- Creating a negative `Cardinality` instance now properly raises an error.
- Infinite `Cardinality` instances now compare properly.
- `PostgresStatistics` now account for the NULL fraction.
- Fixed DuckDB backend not being able to run any EXPLAIN queries.
- Fixed `DuckDBOptimizer.analyze_plan()` crashing when parsing the
  `EXPLAIN ANALYZE` output.
- Scan nodes in DuckDB query plans are now bound to their tables.

## ⚠️ Deprecations

None

## 🪲 Known bugs

- The automatic optimization of the Postgres server configuration as part of the
  Docker installation does not work on MacOS. Currently, this should be considered
  as wontfix.
- The SSB queries can currently not be loaded from the workloads module. The underlying
  data server crashed and we are currently exploring alternative, more reliable
  solutions.

## 🛣 Roadmap

The next months are going to be pretty exciting due to the imminent release of
PostgreSQL 19. As it looks like, this will be the first Postgres version that
provides official support for query hinting via the `pg_plan_advice` extension.
We will definitely add support for pg_plan_advice as a hinting backend, potentially
even making it the standard one and sunsetting development of the `pg_hint_plan`
backend. We will have to wait and see whether pg_hint_plan will be maintained in
the future.

Additionally, the `relalg` module is due for a major overhaul.
