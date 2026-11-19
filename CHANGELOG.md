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

## Version 0.23.5

### 🐣 New features

None

### 💀 Breaking changes

None

### 📰 Updates

- `flatten_and_predicate` is now purely structural: it keeps the conjunct order and duplicates, and also flattens
  conjunctions below OR/NOT.
- `extract_query_fragment` returns `None` for tables that only occur in nested subqueries and for queries whose joins
  cannot be rewritten to implicit form.

### 🏥 Fixes

- `extract_query_fragment`/`extract_subquery` are now deterministic: the FROM clause keeps the order of the original
  query and the WHERE clause of explicit-join queries no longer depends on the Python hash seed.
- `extract_query_fragment` keeps subqueries, VALUES lists and table functions in the FROM clause instead of replacing
  them with a bare table name.

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
