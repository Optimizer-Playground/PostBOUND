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

## Version 0.23.6

### 🐣 New features

- The workload-loading functions (`pb.workloads.stack()`, etc.) now accept an explicit
  `database` argument. This database is used for parsing when binding column references
  to their relations. Structural binding still works as before and workloads like
  JOB do not have the new parameter because they only use structural binding.
  The database pool fallback also works as before.
  The parameter is intended for situations where multiple DB connections need to be
  maintained in the pool.

### 💀 Breaking changes

None

### 📰 Updates

- JSON handling now supports Python's `Decimal`. This type is used by psycopg
  and DuckDB to model `NUMERIC` and actual `DECIMAL`.

### 🏥 Fixes

- Fixed `read_workload` not adding the label prefix as intended. This also
  affected Stack where directory prefixes were silently dropped.

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
