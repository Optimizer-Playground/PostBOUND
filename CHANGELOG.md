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

## Version 0.23.2

### 🐣 New features

None

### 💀 Breaking changes

None

### 📰 Updates

- `parse_duckdb_plan()` now accepts the result set of an `EXPLAIN` query directly
  (raw or simplified), in addition to the JSON string, the parsed JSON and the operator
  tree. It determines on its own which representation it receives.
- `QueryPlan`s created from DuckDB now retain their original `node_type` in addition
  to the parsed `operator` variant. This matches the behavior on Postgres.

### 🏥 Fixes

- Fixed `DuckDBOptimizer.parse_plan()` returning `None` (with a "Could not parse
  plan" warning) for every result set of an `EXPLAIN` query.
- Fixed `DuckDBOptimizer.analyze_plan()` and `parse_duckdb_plan()` for the new
  `EXPLAIN ANALYZE` profiling format of DuckDB/`quacklab` v2. Previously, no query
  could be analyzed on DuckDB since the upgrade to `quacklab` v2.
- Fixed schema lookups (e.g. `is_primary_key()`, `is_unique()`, `lookup_column()`,
  foreign keys and indexes) and the column binding of the parser being case-sensitive.
  This broke DuckDB databases that use mixed-case identifiers, such as `users.Id` in
  the Stats database.

### ⚠️ Deprecations

None

### 🪲 Known bugs

- The automatic optimization of the Postgres server configuration as part of the
  Docker installation does not work on MacOS. Currently, this should be considered
  as wontfix.
- The SSB queries can currently not be loaded from the workloads module. The underlying
  data server crashed and we are currently exploring alternative, more reliable
  solutions.

### 🛣 Roadmap

The next months are going to be pretty exciting due to the imminent release of
PostgreSQL 19. As it looks like, this will be the first Postgres version that
provides official support for query hinting via the `pg_plan_advice` extension.
We will definitely add support for pg_plan_advice as a hinting backend, potentially
even making it the standard one and sunsetting development of the `pg_hint_plan`
backend. We will have to wait and see whether pg_hint_plan will be maintained in
the future.

Additionally, the `relalg` module is due for a major overhaul.
