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

# Version 0.21.2

## 🐣 New features

- Added a new (optional) argument to function expressions to represent keyword arguments.
  Specifically, this allows to now create functions like
  `substring(foo FROM 3 FOR 42)`. Note that parser support is still limited and
  will be added on a per-function basis.
- Added a lot of utilities to create QAL objects, specifically SELECT clauses.
- `as_expression()` can now be used to create function expressions.
- Added an `as_query()` utility similar to `as_expression()` or `as_predicate()`.
  It has a slightly nicer API than the existing `build_query()`.
- Improved the creation of `SimpleFilter` and `SimpleJoin` predicates to make double
  checks redundant. `attempt_wrap()` methods are now the preferred entry point into
  the simplification.

## 📰 Updates

- Added jsonize support for optimization pipelines, optimization stages, and databases.

## 🏥 Fixes

- Fixed plan parameterizations not being JSON-serializable.
- Fixed the shorthand method for creating _COUNT(\*)_ select clauses accepting a
  distinct parameter instead of a target name.
- Fixed `as_predicate()` creating star expressions instead of literal star-strings.
  For example, the predicate _foo = '\*'_ was wrongly turned into the (illegal)
  predicate _foo = \*_ (note the missing quotes).
- Fixed database setup scripts for Postgres breaking if the number of CPU cores
  exceeded the number of allowed connections.

---

# Version 0.21.1

## 🐣 New features

- Added a `pre_exec_callback` parameter to the benchmarking utilities. This is
  called right before the query execution.
- Pre-execution and post-execution callbacks for the benchmarking utilities can
  now optionally return a dictionary. The key/value pairs of this dictionary will
  be automatically added to the result data frame as additional columns. This allows
  to incorporate custom measurements into the benchmarking process.
- Added a lot of convenience functions to manually create QAL objects

## 📰 Updates

- Added an `extract_subquery()` transformation as a saner alternative to `extract_query_fragment()`.
- Improved the performance of `PreComputedCardinalities` for large queries

## 🏥 Fixes

- Lifted the restriction of only one parallel hint per plan for Postgres.
  We have been under the impression that Postgres can only run a single subtree
  of a plan in parallel. As it turns out, this is not the case if the two subtrees
  are "independent" of each other, i.e. appear in parallel branches of the plan.
- Fixed hinting a plan for pg_lab messing up the placement of parallel workers.
  The bug occurred when a join computed its outer child in parallel. Instead of
  placing the parallel workers on that child, the hinting process placed them on
  the inner child.
- Fix TableReference `with_alias()` dropping the schema.

## ⚠️ Deprecations

- The old `QueryPrepration` API using _analyze=True_, etc. is now deprecated in
  favor of the more flexible _projection_ and _output_ parameters. However, we
  currently have no plans to remove the old API.
- The _ues_, _tonic_, _presets_, and _experiments_ module are now deprecated and
  will be moved to the separate optimizer repository for version 0.22.0.
- `Workload.read()` is deprecated in favor of `read_workload()`. The old method
  will be removed in version 0.22.0.
  This change unifies the workload API and consistently uses `read_workload_XXX`
  functions for input.
- `CompoundPredicate` will no longer be used to represent NOT predicates from
  version 0.22.0 onwards. Instead, a dedicated `NotPredicate` class will be introduced.
- `MathExpression` will no longer accept sequences of expressions as the second
  argument. To encode operations involving more than 2 operands, a hierarchy of
  binary math expressions should be used instead.
- Databases will no longer support the database cache out-of-the-box. Instead,
  the cache will become a proper high-level component that can be used with any
  database. This change is planned for version 0.22.0
- Renamed the _exec_callback_ parameter in `bench.execute_workload` to `post_exec_callback`.
  This should prevent confusion regarding the precise timing of the callback execution,
  especially since we have now also introduced a _pre_exec_callback_.

---

# Version 0.21.0

This is a rather large release with a lot of new features and improvements. Some highlights include:

- Optimization stages now provide high-level support for learning from different kinds of data, such as workloads,
  databases, etc.
- Lots of usability improvements throughout the framework, e.g. for better retrieval of information from the database
  schema, easier specification of query preparations in benchmarks, etc.
- The usability of the database schema has been improved significantly. You can now iterate over the schema to obtain all
  contained tables, or use dict-style access to obtain more information about specific tables or columns.
- Introduction of new histogram and most common values types for high-level access to these statistics.

While this release does not contain any major breaking changes, we are preparing to clean up some old and unfortunate parts of
the framework. Currently, these are planned for version 0.22.0.

## 🐣 New features

- Optimization stages can now specify whether they require training on data samples or actual query executions. The
  benchmarking tools and optimization pipeline will automatically trigger the training of such stages if they have not been
  trained already. This allows to easily use data-driven and workload-driven optimization stages without any explicit
  action needed by the user.
- The database interface now provides a shortcut `explain()` method to obtain the query plan for a given query. This can be
  used instead of calling the optimizer and it's explain method.
- The database schema now provides a high-level API centered around iteration and dict-style access. This makes the repeated
  calls to different schema methods somewhat redundant.
- Database statistics now also provide histograms.
- The `OptimizerInterface` (e.g. `pg_instance.optimizer()`) now provides a `parse_plan` method to parse an existing
  system-specific query plan to the generalized `QueryPlan`. This can be used as follows:

  ```python
  explain_query = pb.transform.as_explain(query)
  raw_plan = database.execute_query(explain_query)
  plan = database.optimizer().parse_plan(raw_plan)
  ```

- `execute_workload()` now supports many new output formats for writing the progressive output.
  Currently supported are: CSV, Parquet, JSON, HDF
- Workloads now support transformations of their queries, e.g. `workload.map(pb.transform.as_star_query)`
- Added a `fetch_workload()` function to the workloads module. It allows to load a pre-defined workload by name.
- Added a `n_buffered()` and `buffer_state()` methods to the Postgres statistics interface to retrieve the number of
  currently buffered pages of a relation.
- Added an additional `BoundColumnReference` core type. Instances guarantee to be bound to a `TableReference`.
  The new `assert_bound()` serves as a type guard to narrow references. This should prevent constant checks for valid
  table references on columns.
- Added a `joins_tables()` and `joins_columns()` utilities to query predicates.
- Added a `merge_tables()` transformation to rewrite queries for materialized views.
- `util.simplify()` now supports single-item mappings as well.
- `util.to_json()` now supports the standard date-like objects (_datetime_, _date_, _time_, and _timedelta_)
- `util.write_df()` now automatically transforms complex objects in the data frame into their JSON representation before
  writing.
- In IPython and Jupyter sessions, common PostBOUND objects like SQL queries or query plans are now automatically
  pretty-printed. Instead of calling `print(plan.explain())`, one can now simply make `plan` the result of a cell.

## 📰 Updates

- `DatabaseStatistics.most_common_values()` now returns an actual `MostCommonValues` object instead of a list of tuples.
  The `MostCommonValues` can be used as a drop-in replacement for the old tuple-based API. In addition, it provides more
  high-level methods for working with the most common values.
- Enabled the MySQL and DuckDB backends to fall back to emulated statistics if the database does not provide them.
- The Postgres `execute_query()` method now accepts hint parameters and automatically applies them. For example, the
  following can now be done without explicit hinting:

  ```python
  query, plan = ...  # whatever
  pg_instance.execute_query(query, plan=plan)

  # this is equivalent to
  hinted_query = pg_instance.hinting().generate_hints(query, plan)
  pg_instance.execute_query(hinted_query)
  ```

- The Postgres interface now has a `rollback()` method to put connections back into a valid state.
- The Postgres statistics interface now consistently supports table references with a schema.
- Much improved handling of database schemas during query parsing. We now omit clear warnings in case the database pool
  looks weird.
- The `QueryPreparation` API now provides the `projection` and `output` parameters to modify the SELECT\* clause and the
  type of results to gather for all queries in a more flexible and intuitive way (how did _explain=True_ and
  _analyze=True_ interact?).
  The old API using _analyze=True_, etc. is now deprecated in favor of these new parameters.
- Column references now provide a `drop_table_alias()` method to obtain a normalized-ish representation of the column.
  This should be helpful in situations where references to the same column are not consistent in their table references,
  e.g., when one was obtained from the schema and the other was obtained from the query.
- While obtaining a join graph for a query, aliased tables can now be merged into the same node.
- The `PredicateVisitor` can now be started at the query. It will extract the predicates as needed.
- The `extract_query_fragment()` tranformation now supports modifying the output projection.
- The `to_json()` and `to_json_dump()` utilities now support dataclasses out-of-the-box.
- Expose `argmax()` directly in _util_ module

## 🏥 Fixes

- Fixed `n_buffered()` method of the Postgres statistics interface raising an error if no pages of the relation are
  currently buffered. We now return 0 in this case.
- Fixed string representation of `COUNT(DISTINCT ...)` for multiple arguments. We now generate the correct
  `COUNT(DISTINCT (a, b))` instead of `COUNT(DISTINCT a, b)`.
- Fixed `DatabaseSchema.as_graph()` having the assignment of primary key and foreign key columns reversed on join edges.
- Fixed output format of the benchmarking log if additional entries are appended to an existing log. Essentially, we fix
  such entries being escaped twice.
- Fixed the `standard_logger` sometimes logging internal module names.
- Fixed parser for column JSON
- Updated all database setup scripts for Postgres and DuckDB. Since our data server that hosted the raw input data
  crashed once again and broke all download links, we now moved the entire setup to Zenodo. Hopefully, this setup is
  more stable. There are several practical implications of this change:
  1. Instead of distributing raw CSV data, we now provide pre-build database images for Postgres and DuckDB. This should
     lower the setup time significantly
  2. The JOB-complex and JOB-light workloads now use a different indexing scheme. Instead of queries 1, 2, 3, ... we now
     label them similar to Stats, i.e. q-1, q-2, ...
  3. The DuckDB workload-setup.py script was removed - as a consequence of 1., we no longer need to create the database
     files, but distribute them directly.
  4. The SSB queries can currently not be loaded from the workloads module.

## 💀 Breaking changes

- Renamed `PreciseCardinalityHintGenerator` to `PreciseCardinalities` to align with the other pre-defined cardinality
  "estimators".

## ⚠️ Deprecations

- The old `QueryPrepration` API using _analyze=True_, etc. is now deprecated in favor of the more flexible _projection_
  and _output_ parameters. However, we currently have no plans to remove the old API.
- The _ues_, _tonic_, _presets_, and _experiments_ module are now deprecated and will be moved to the separate optimizer
  repository for version 0.22.0.
- `Workload.read()` is deprecated in favor of `read_workload()`. The old method will be removed in version 0.22.0.
  This change unifies the workload API and consistently uses `read_workload_XXX` functions for input.
- `CompoundPredicate` will no longer be used to represent NOT predicates from version 0.22.0 onwards. Instead, a dedicated
  `NotPredicate` class will be introduced.
- Databases will no longer support the database cache out-of-the-box. Instead, the cache will become a proper high-level
  component that can be used with any database. This change is planned for version 0.22.0

## 🪲 Known bugs

- The automatic optimization of the Postgres server configuration as part of the Docker installation does not work
  on MacOS. Currently, this should be considered as wontfix.
- The SSB queries can currently not be loaded from the workloads module. The underlying data server crashed and we are
  currently exploring alternative, more reliable solutions.

---

# 🛣 Roadmap

Currently, we plan to implement the following features in the future (in no particular order):

- Providing a Substrait export for query plans
- Better benchmarking setup, mostly focused on comparing one or multiple optimization pipelines and creating better
  experiment logs and the ability to cancel/resume long-running benchmarks
